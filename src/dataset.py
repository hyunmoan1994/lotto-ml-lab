from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from .preprocess import WIN_COLS, numbers_to_multihot


def build_sequence_dataset(df, seq_len: int = 20) -> tuple[np.ndarray, np.ndarray]:
    draws = np.vstack([numbers_to_multihot(row) for row in df[WIN_COLS].to_numpy()])
    xs, ys = [], []
    for i in range(seq_len - 1, len(draws) - 1):
        xs.append(draws[i - seq_len + 1 : i + 1])
        ys.append(draws[i + 1])
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


def latest_sequence(df, seq_len: int = 20) -> np.ndarray:
    draws = np.vstack([numbers_to_multihot(row) for row in df[WIN_COLS].to_numpy()])
    if len(draws) < seq_len:
        pad = np.zeros((seq_len - len(draws), 45), dtype=np.float32)
        draws = np.vstack([pad, draws])
    return draws[-seq_len:].astype(np.float32)[None, :, :]


class LottoSequenceDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


def chronological_split(n: int, train_ratio: float = 0.7, valid_ratio: float = 0.15) -> tuple[slice, slice, slice]:
    train_end = max(1, int(n * train_ratio))
    valid_end = max(train_end + 1, int(n * (train_ratio + valid_ratio)))
    valid_end = min(valid_end, n)
    return slice(0, train_end), slice(train_end, valid_end), slice(valid_end, n)
