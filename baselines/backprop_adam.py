"""AdamW backpropagation baseline with the same model, data, and evaluation as DUST.

Not part of the official release. Defaults are the paper's selected Adam settings at 1M tokens
(Appendix F, Table 7): matrices and head 0.002, token and value embeddings 0.3, residual scalars
0.2, beta1 0.8, beta2 0.95, eps 1e-8, no weight decay, constant learning rate.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dust import GPTConfig, evaluate, load_sequences, make_model

SCHEDULES = {'1M': dict(steps=61, eval_interval=2), '10M': dict(steps=610, eval_interval=10)}


def make_optimizer(model, args):
    groups = {'matrix': [], 'embedding': [], 'scalar': []}
    for name, parameter in model.named_parameters():
        if name == 'transformer.wte.weight' or 'value_embeds' in name:
            groups['embedding'].append(parameter)
        elif 'lambdas' in name:
            groups['scalar'].append(parameter)
        else:
            groups['matrix'].append(parameter)
    rates = dict(matrix=args.lr, embedding=args.embedding_lr, scalar=args.scalar_lr)
    return torch.optim.AdamW(
        [{'params': ps, 'lr': rates[kind]} for kind, ps in groups.items()],
        betas=(args.beta1, 0.95), eps=1e-8, weight_decay=args.weight_decay,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tokens', choices=SCHEDULES, default='1M')
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lr', type=float, default=0.002)
    parser.add_argument('--embedding-lr', type=float, default=0.3)
    parser.add_argument('--scalar-lr', type=float, default=0.2)
    parser.add_argument('--beta1', type=float, default=0.8)
    parser.add_argument('--weight-decay', type=float, default=0.0)
    parser.add_argument('--train-file', type=Path, help='Train on every sequence of this file instead.')
    parser.add_argument('--steps', type=int, help='Required with --train-file.')
    parser.add_argument('--eval-interval', type=int)
    parser.add_argument('--init', type=Path, help='Checkpoint from --save-final to start from.')
    parser.add_argument('--keep-optimizer', action='store_true', help="Also restore --init's AdamW state.")
    parser.add_argument('--save-final', action='store_true', help='Save final weights and AdamW state.')
    args = parser.parse_args()

    device = torch.device('cuda', 0)
    torch.set_float32_matmul_precision('high')
    config = GPTConfig()
    steps = args.steps or SCHEDULES[args.tokens]['steps']
    interval = args.eval_interval or SCHEDULES[args.tokens]['eval_interval']
    training = load_sequences(args.train_file or args.data_dir / f'train_{args.tokens}.pt', steps * 8)
    validation = load_sequences(args.data_dir / 'val.pt', 544)
    testing = load_sequences(args.data_dir / 'test.pt', 544)
    order = torch.randperm(len(training), generator=torch.Generator().manual_seed(args.seed + 1))
    training = training[order]

    model = make_model(config, device, args.seed).requires_grad_(True)
    optimizer = make_optimizer(model, args)
    if args.init:
        initial = torch.load(args.init, map_location=device, weights_only=True)
        model.load_state_dict(initial['model'])
        if args.keep_optimizer:
            optimizer.load_state_dict(initial['optimizer'])
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f'Choose an empty output directory: {args.output}')
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    metadata.update(method='backprop-adamw', model=asdict(config), steps=steps)
    (args.output / 'config.json').write_text(json.dumps(metadata, indent=2) + '\n')

    best = evaluate(model, validation, device)
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    history = [dict(step=0, val_loss=best)]
    start = time.monotonic()
    for step in range(1, steps + 1):
        batch = training[(step - 1) * 8:step * 8].to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss = model(batch[:, :-1].contiguous(), batch[:, 1:].contiguous())
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite backpropagation loss')
        loss.backward()
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
    if args.save_final:
        torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict()), args.output / 'final.pt')
    model.load_state_dict(best_state)
    result = dict(best_val=best, final_val=final_val, test_at_best_val=evaluate(model, testing, device),
                  history=history)
    (args.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(f"{args.output.name}: best val {best:.4f}, final val {final_val:.4f}, "
          f"test {result['test_at_best_val']:.4f}", flush=True)


if __name__ == '__main__':
    main()
