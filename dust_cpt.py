"""Continue training a backprop checkpoint with DUST under Adam. Not part of the official release.

Tests whether the estimator still lowers the loss from a model backprop has already trained.
Starts from the weights in DUST_INIT (a final.pt of baselines/backprop_adam.py) with a fresh Adam
state and trains DUST_STEPS steps on data/train_<DUST_DATA>.pt, evaluating every 10 steps.
Run as: DUST_INIT=... DUST_DATA=cpt DUST_STEPS=200 python dust_cpt.py --tokens cpt --population N
"""
import os

import torch

import dust
import dust_adam

make_scratch_model = dust.make_model


def make_model(config, device, seed):
    model = make_scratch_model(config, device, seed)
    initial = torch.load(os.environ['DUST_INIT'], map_location=device, weights_only=True)
    model.load_state_dict(initial['model'])
    return model


if __name__ == '__main__':
    name = os.environ['DUST_DATA']
    dust.DEFAULT_CONFIGS[name] = dict(steps=int(os.environ['DUST_STEPS']), eval_interval=10)
    dust.recipe, dust.make_optimizer, dust.DUST = dust_adam.recipe, dust_adam.make_optimizer, dust_adam.DustAdam
    dust.make_model = make_model
    dust.main()
