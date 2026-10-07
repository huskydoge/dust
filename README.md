# DUST

> **Fork note.** This fork adds an AdamW backprop baseline, Dust under Adam and continuation from backprop
> checkpoints, with a [reproduction report](https://huskydoge.github.io/dust/): see [`reproduction/`](reproduction/README.md).
> The official files are unchanged.

This is a minimal implementation of our research: <https://qlabs.sh/research/dust>

Train a transformer using forward evaluations and SGD, without backpropagation.
This minimal implementation preserves the paper's estimator and tuned defaults
while omitting execution optimizations used in the full experiments.

[`model.py`](model.py) contains the model, and [`dust.py`](dust.py) the algorithm and training loop. A simple
[backpropagation SGD baseline](baselines/backprop.py) uses the same model and data.

## Installation

Requires Linux, Python 3.10+, and an NVIDIA GPU with bfloat16 support.
The following installs PyTorch with CUDA 12.8; use a compatible NVIDIA driver.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

## Data

Prepare the 1M and 10M training sets:

```bash
python prepare_data.py
```

This streams FineWeb from Hugging Face and saves `data/train_1M.pt`,
`data/train_10M.pt`, and shared `data/val.pt` and `data/test.pt` files.
Use `--tokens 1M` or `--tokens 10M` to prepare only one training budget.
Each held-out split contains 544 sequences of 2,048 prediction tokens.

The fixed BPE-4096 tokenizer is included; retraining is optional.
Its training recipe and script are in [`tokenizer/`](tokenizer/README.md).

## DUST

Train with the default population of **16,384** on eight GPUs:

```bash
torchrun --standalone --nproc_per_node=8 dust.py --tokens 1M --output runs/1m
torchrun --standalone --nproc_per_node=8 dust.py --tokens 10M --output runs/10m
```

For a smaller, single-GPU run:

```bash
python dust.py --tokens 1M --population 256 --output runs/1m-p256
```

Supported populations are **256, 1024, 4096, and 16384**, using respectively
**1, 1, 4, and 8 GPUs** in the paper configurations. Each selects its tuned
per-layer draw allocation, credit decay and optimizer settings. Population counts direct-loss
draws across all GPUs; head and local attention draws are additional. Every GPU
processes the same batch and contributes different perturbations.

Both token budgets use an 8-layer, 512-wide transformer, sequence length 2,048,
and batch size 8. Add `--steps 2` for a short check using the selected dataset.

## Backpropagation baseline

Run SGD backpropagation on one GPU:

```bash
python baselines/backprop.py --tokens 1M --output runs/bp-1m
python baselines/backprop.py --tokens 10M --output runs/bp-10m
```

## Outputs

Each run saves its configuration, metrics, a validation-selected `best.pt`, and
`result.json` containing the selected checkpoint's test loss. Use a new output
directory for each run. Data, checkpoints, and logs are stored locally.

## License

Released under the [MIT License](LICENSE).
