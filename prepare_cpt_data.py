"""Prepare continuation data that no existing split contains. Not part of the official release.

Replays prepare_data.py's document stream and checks it against the saved files, then saves:
  train_cpt.pt     pool rows 4880-6479, the 1,600 sequences after the 10M training prefix
  train_100M.pt    48,800 sequences from documents after the test split
  train_cpt100.pt  the 1,600 sequences after those
"""
from pathlib import Path

import torch
from datasets import load_dataset
from tokenizers import Tokenizer

from prepare_data import encode_split

data = Path('data')
corpus = load_dataset('HuggingFaceFW/fineweb', name='sample-10BT', split='train', streaming=True)
tokenizer = Tokenizer.from_file('tokenizer/tokenizer.json')
documents = iter(corpus)
for _ in range(200_000):
    next(documents)

pool = encode_split(documents, tokenizer, 20_500_000)
assert torch.equal(pool[:4880], torch.load(data / 'train_10M.pt', weights_only=True)), 'pool differs'
torch.save(pool[4880:6480].clone(), data / 'train_cpt.pt')
print('train_cpt.pt', flush=True)
for split in ('val', 'test'):
    rows = encode_split(documents, tokenizer, 1_150_000)[:544]
    assert torch.equal(rows, torch.load(data / f'{split}.pt', weights_only=True)), f'{split} differs'
extra = encode_split(documents, tokenizer, 50_400 * 2049)
torch.save(extra[:48800].clone(), data / 'train_100M.pt')
torch.save(extra[48800:50400].clone(), data / 'train_cpt100.pt')
print('train_100M.pt train_cpt100.pt', len(extra), flush=True)
