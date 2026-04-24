from __future__ import annotations

import numpy as np
import pandas as pd


def top_set(scores: np.ndarray, k: int = 6) -> list[int]:
    return sorted((np.argsort(scores)[-k:] + 1).tolist())


def stochastic_set(scores: np.ndarray, k: int = 6, pool: int = 15, seed: int = 42) -> list[int]:
    rng = np.random.default_rng(seed)
    idx = np.argsort(scores)[-pool:]
    weights = np.clip(scores[idx], 1e-9, None)
    weights = weights / weights.sum()
    picked = rng.choice(idx, size=k, replace=False, p=weights)
    return sorted((picked + 1).tolist())


def build_recommendations(
    model_scores: dict[str, np.ndarray],
    seed: int = 42,
    sets_per_model: dict[str, int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rec_rows, score_rows, seen = [], [], set()
    for offset, (model_name, scores) in enumerate(model_scores.items()):
        scores = np.asarray(scores, dtype=float)
        for n, score in enumerate(scores, start=1):
            score_rows.append({"model": model_name, "number": n, "score": float(score)})
        set_count = max(1, int((sets_per_model or {}).get(model_name, 2)))
        model_sets = [("top_score", top_set(scores))]
        for i in range(1, set_count):
            model_sets.append((f"score_sampling_{i}", stochastic_set(scores, seed=seed + offset * 100 + i)))
        for method, nums in model_sets:
            key = tuple(nums)
            if key in seen:
                adjusted_scores = scores.copy()
                for n in nums:
                    adjusted_scores[n - 1] *= 0.97
                nums = stochastic_set(adjusted_scores, seed=seed + offset + len(seen) + 100)
                key = tuple(nums)
            seen.add(key)
            rec_rows.append({"model": model_name, "method": method, **{f"n{i+1}": nums[i] for i in range(6)}})
    return pd.DataFrame(rec_rows), pd.DataFrame(score_rows)
