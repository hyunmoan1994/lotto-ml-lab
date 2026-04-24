from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .dataset import LottoSequenceDataset, chronological_split
from .evaluate import evaluate_scores
from .model_registry import DL_MODEL_NAMES
from .utils import MODEL_DIR


class RNNModel(nn.Module):
    def __init__(self, hidden: int = 64) -> None:
        super().__init__()
        self.rnn = nn.RNN(45, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 45)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(out[:, -1])


class LSTMModel(nn.Module):
    def __init__(self, hidden: int = 64) -> None:
        super().__init__()
        self.lstm = nn.LSTM(45, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 45)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1])


class TransformerModel(nn.Module):
    def __init__(self, d_model: int = 64, nhead: int = 4, layers: int = 2) -> None:
        super().__init__()
        self.proj = nn.Linear(45, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=128, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.head = nn.Linear(d_model, 45)

    def forward(self, x):
        z = self.proj(x)
        out = self.encoder(z)
        return self.head(out[:, -1])


def _train_one(model: nn.Module, x: np.ndarray, y: np.ndarray, name: str, save_path: Path, epochs: int = 8) -> dict:
    tr, va, te = chronological_split(len(x))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    loader = DataLoader(LottoSequenceDataset(x[tr], y[tr]), batch_size=32, shuffle=False)
    loss_fn = nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    logs: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(xb)
        logs.append({"epoch": epoch, "loss": total / max(1, len(loader.dataset))})
    valid_scores = predict_dl_scores(model, x[va], device)
    test_scores = predict_dl_scores(model, x[te], device) if len(x[te]) else valid_scores
    torch.save({"state_dict": model.state_dict(), "name": name}, save_path)
    return {"name": name, "model": model, "valid_metrics": evaluate_scores(y[va], valid_scores), "test_metrics": evaluate_scores(y[te], test_scores) if len(y[te]) else {}, "logs": logs}


def predict_dl_scores(model: nn.Module, x: np.ndarray, device: torch.device | None = None) -> np.ndarray:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(x, dtype=torch.float32, device=device))
        return torch.sigmoid(logits).cpu().numpy()


def train_all_dl(x_seq: np.ndarray, y_seq: np.ndarray, epochs: int = 8) -> list[dict]:
    return train_selected_dl(x_seq, y_seq, DL_MODEL_NAMES, epochs=epochs)


def train_selected_dl(x_seq: np.ndarray, y_seq: np.ndarray, model_names: list[str], epochs: int = 8) -> list[dict]:
    factories = {
        "RNN": lambda: (RNNModel(), MODEL_DIR / "rnn.pt"),
        "LSTM": lambda: (LSTMModel(), MODEL_DIR / "lstm.pt"),
        "Transformer": lambda: (TransformerModel(), MODEL_DIR / "transformer.pt"),
    }
    results = []
    for name in model_names:
        if name not in factories:
            continue
        model, save_path = factories[name]()
        results.append(_train_one(model, x_seq, y_seq, name, save_path, epochs))
    return results
