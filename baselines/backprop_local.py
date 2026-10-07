"""Backprop with each linear layer fitted locally to a gradient-descended activation (LocoProp-style).

Not part of the official release. Backprop gives the gradient delta_t at a layer's output y_t = W x_t.
The layer's target is y_t - eta * delta_t, and the layer solves

    min_W  1/2 sum_t |W x_t - target_t|^2 + lambda/2 |W - W_0|^2 .

One gradient step on this problem is a backprop step. Its exact minimizer is

    W = W_0 - eta * G (C + lambda I)^-1 ,   G = sum_t delta_t x_t^T ,   C = sum_t x_t x_t^T ,

the backprop gradient times the inverse second moment of the layer's input on the current batch. This
script uses that minimizer: every nn.Linear gradient G is replaced by G P^-1 before the optimizer step, with

    P = (C/N + damping * m I) / ((1 + damping) * m) ,   m = mean(diag C/N) .

P has mean eigenvalue one, so the step size keeps its meaning across dampings and P tends to the identity,
plain backprop, as the damping grows; --damping inf skips the solve. With --matrix-optimizer sgd the
solution is applied as the update (with momentum); with adamw the preconditioned gradient is passed to
AdamW. Embeddings and residual scalars keep the AdamW settings of baselines/backprop_adam.py.
"""
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys
import time

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dust import GPTConfig, evaluate, load_sequences, make_model

SCHEDULES = {'1M': dict(steps=61, eval_interval=2), '10M': dict(steps=610, eval_interval=10)}


class LocalFit:
    """Records each linear layer's input second moment during the forward pass and applies the local solution."""

    def __init__(self, model, damping):
        self.damping, self.recording, self.moments = damping, False, {}
        self.layers = {name: module for name, module in model.named_modules() if isinstance(module, nn.Linear)}
        for name, module in self.layers.items():
            module.register_forward_pre_hook(self.record(name))

    def record(self, name):
        def hook(module, args):
            if self.recording:
                # Outside autocast: the second moment and the solve need float32.
                with torch.autocast('cuda', enabled=False):
                    x = args[0].detach().reshape(-1, args[0].shape[-1]).float()
                    self.moments[name] = x.T @ x / x.shape[0]
        return hook

    @torch.no_grad()
    def precondition(self):
        if math.isinf(self.damping):
            return
        for name, module in self.layers.items():
            moment = self.moments[name]
            scale = moment.diagonal().mean()
            identity = torch.eye(len(moment), device=moment.device)
            moment = (moment + self.damping * scale * identity) / ((1 + self.damping) * scale)
            gradient = module.weight.grad.float()
            # moment is symmetric, so solving against gradient^T gives (gradient @ moment^-1)^T.
            module.weight.grad = torch.linalg.solve(moment, gradient.T).T.to(module.weight.dtype)


def make_optimizers(model, args):
    matrices, embeddings, scalars = [], [], []
    for name, parameter in model.named_parameters():
        if name == 'transformer.wte.weight' or 'value_embeds' in name:
            embeddings.append(parameter)
        elif 'lambdas' in name:
            scalars.append(parameter)
        else:
            matrices.append(parameter)
    adam = dict(betas=(args.beta1, 0.95), eps=1e-8, weight_decay=0.0)
    others = torch.optim.AdamW([{'params': embeddings, 'lr': args.embedding_lr}, {'params': scalars, 'lr': args.scalar_lr}], **adam)
    if args.matrix_optimizer == 'sgd':
        return [torch.optim.SGD(matrices, lr=args.lr, momentum=args.momentum), others]
    return [torch.optim.AdamW(matrices, lr=args.lr, **adam), others]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--tokens', choices=SCHEDULES, default='10M')
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--matrix-optimizer', choices=['sgd', 'adamw'], default='adamw')
    parser.add_argument('--lr', type=float, default=0.002, help='Step size of the matrices: eta for sgd, the AdamW rate otherwise.')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum of the matrices under sgd.')
    parser.add_argument('--damping', type=float, default=0.1, help='lambda, relative to the mean diagonal of the input second moment; inf is plain backprop.')
    parser.add_argument('--embedding-lr', type=float, default=0.3)
    parser.add_argument('--scalar-lr', type=float, default=0.2)
    parser.add_argument('--beta1', type=float, default=0.8)
    args = parser.parse_args()

    device = torch.device('cuda', 0)
    torch.set_float32_matmul_precision('high')
    config = GPTConfig()
    steps, interval = SCHEDULES[args.tokens]['steps'], SCHEDULES[args.tokens]['eval_interval']
    training = load_sequences(args.data_dir / f'train_{args.tokens}.pt', steps * 8)
    validation = load_sequences(args.data_dir / 'val.pt', 544)
    testing = load_sequences(args.data_dir / 'test.pt', 544)
    order = torch.randperm(len(training), generator=torch.Generator().manual_seed(args.seed + 1))
    training = training[order]

    model = make_model(config, device, args.seed).requires_grad_(True)
    optimizers = make_optimizers(model, args)
    fit = LocalFit(model, args.damping)
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f'Choose an empty output directory: {args.output}')
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    metadata.update(method='backprop-local-fit', model=asdict(config), steps=steps)
    (args.output / 'config.json').write_text(json.dumps(metadata, indent=2) + '\n')

    best = evaluate(model, validation, device)
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    history = [dict(step=0, val_loss=best)]
    start = time.monotonic()
    for step in range(1, steps + 1):
        batch = training[(step - 1) * 8:step * 8].to(device)
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        fit.recording = True
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss = model(batch[:, :-1].contiguous(), batch[:, 1:].contiguous())
        fit.recording = False
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite loss')
        loss.backward()
        fit.precondition()
        for optimizer in optimizers:
            optimizer.step()
        record = dict(step=step, train_loss=loss.item(), elapsed_seconds=time.monotonic() - start)
        if step == 1 or step % interval == 0 or step == steps:
            value = evaluate(model, validation, device)
            record['val_loss'] = value
            if value < best:
                best = value
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        history.append(record)
        with (args.output / 'metrics.jsonl').open('a') as stream:
            stream.write(json.dumps(record) + '\n')

    final_val = history[-1]['val_loss']
    model.load_state_dict(best_state)
    result = dict(best_val=best, final_val=final_val, test_at_best_val=evaluate(model, testing, device), history=history)
    (args.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(f"{args.output.name}: best val {best:.4f}, final val {final_val:.4f}, test {result['test_at_best_val']:.4f}", flush=True)


if __name__ == '__main__':
    main()
