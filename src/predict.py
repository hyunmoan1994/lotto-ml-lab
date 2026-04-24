from __future__ import annotations

import numpy as np

from .train_dl import predict_dl_scores
from .train_ml import predict_ml_scores


def collect_latest_scores(dl_results: list[dict], ml_results: list[dict], x_latest, seq_latest: np.ndarray) -> dict[str, np.ndarray]:
    scores: dict[str, np.ndarray] = {}
    for result in dl_results:
        scores[result["name"]] = predict_dl_scores(result["model"], seq_latest)[0]
    for result in ml_results:
        scores[result["name"]] = predict_ml_scores(result["model"], x_latest)
    return scores
