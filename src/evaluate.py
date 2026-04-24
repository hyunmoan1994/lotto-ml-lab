from __future__ import annotations

import numpy as np


def top_k_indices(scores: np.ndarray, k: int = 6) -> np.ndarray:
    return np.argsort(scores)[-k:][::-1]


def evaluate_scores(y_true: np.ndarray, y_score: np.ndarray, k: int = 6) -> dict[str, float]:
    hits, precisions, recalls, overlaps, briers = [], [], [], [], []
    for yt, ys in zip(y_true, y_score):
        pred = top_k_indices(ys, k)
        true = set(np.where(yt > 0.5)[0])
        hit = len(set(pred) & true)
        hits.append(hit)
        precisions.append(hit / k)
        recalls.append(hit / max(1, len(true)))
        overlaps.append(hit / len(set(pred) | true))
        briers.append(float(np.mean((np.clip(ys, 0, 1) - yt) ** 2)))
    return {
        "hit@6_avg": float(np.mean(hits)),
        "precision@6": float(np.mean(precisions)),
        "recall@6": float(np.mean(recalls)),
        "topk_overlap": float(np.mean(overlaps)),
        "brier_score": float(np.mean(briers)),
    }
