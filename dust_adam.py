"""DUST under Adam, reconstructed from the paper's appendix. Not part of the official release.

Appendix E, Table 3 (Adam column, K = 16,384): writer pair 4,096, MLP hidden 4,096, attention
output 6,144 and token embedding 8,192 direct draws (11K/8 in total), head 49,152, attention
internals as under SGD, and token-embedding noise scale 0.4. Appendix F, Table 7: matrices and
head 0.0015 (0.002 from 16k), embeddings 300x and residual scalars 100x that rate, beta1 0.7,
beta2 0.95, eps 1e-8, no weight decay.

The paper does not give the Adam ladder's per-block split. WRITERS and HUBS below rescale the
SGD tilt to the Adam totals; that split is our assumption.

DUST_ADAM_LR and DUST_ADAM_BETA1 override the matrix rate and beta1 for tuning sweeps.
"""
import os

import torch

import dust

WRITERS = [20, 16, 10, 6, 4, 4, 2, 2]    # per 256 of population; sums to 64
HUBS = [22, 22, 14, 8, 8, 8, 8, 6]       # sums to 96


def recipe(tokens, population):
    scale = population // 256
    writers = [v * scale for v in WRITERS]
    local = [v * scale for v in [2, 20, 6, 12, 8]]
    return dust.Recipe(
        writers, writers.copy(), [v * scale for v in HUBS], 128 * scale, 3 * population,
        dict(zip(('q', 'v', 'gate', 'k', 've'), local)), dust.DRAW_CHUNK_SIZES[population],
        lr=float(os.environ.get('DUST_ADAM_LR', 0.002 if population >= 16384 else 0.0015)),
        momentum=float(os.environ.get('DUST_ADAM_BETA1', 0.7)),
    )


def make_optimizer(model, settings):
    groups = {1: [], 300: [], 100: []}
    for name, parameter in model.named_parameters():
        if name == 'transformer.wte.weight' or 'value_embeds' in name:
            groups[300].append(parameter)
        elif 'lambdas' in name:
            groups[100].append(parameter)
        else:
            groups[1].append(parameter)
    optimizer = torch.optim.Adam(
        [{'params': ps, 'lr': settings.lr * factor} for factor, ps in groups.items()],
        betas=(settings.momentum, 0.95), eps=1e-8,
    )
    # dust.main logs each group's 'momentum'; Adam ignores the extra key.
    for group in optimizer.param_groups:
        group['momentum'] = settings.momentum
    return optimizer


class DustAdam(dust.DUST):
    def direct_errors(self, sites, draws, sigma):
        if sites == ['transformer.wte']:
            sigma = 0.4
        return super().direct_errors(sites, draws, sigma)


if __name__ == '__main__':
    dust.recipe, dust.make_optimizer, dust.DUST = recipe, make_optimizer, DustAdam
    dust.main()
