from __future__ import annotations

import numpy as np
import pandas as pd

from .preprocess import NUMBERS, WIN_COLS, draw_matrix, numbers_to_multihot


WINDOWS = [5, 10, 30, 50, 100]


def _consecutive_count(nums: np.ndarray) -> int:
    nums = np.sort(nums)
    return int(np.sum(np.diff(nums) == 1))


def build_feature_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    matrix = draw_matrix(df, include_bonus=False)
    bonus_matrix = draw_matrix(df[["draw_no", "draw_date", "bonus"]].rename(columns={"bonus": "n1"}).assign(n2=0, n3=0, n4=0, n5=0, n6=0), False)
    rows: list[dict[str, float | int | str]] = []
    last_seen = np.full(45, -1, dtype=int)
    for i, raw in df.iterrows():
        nums = raw[WIN_COLS].to_numpy(dtype=int)
        hist = matrix[: i + 1]
        row: dict[str, float | int | str] = {"draw_no": int(raw.draw_no), "draw_date": str(raw.draw_date)}
        for w in WINDOWS:
            recent = hist[max(0, i + 1 - w) : i + 1]
            freq = recent.sum(axis=0) / max(1, len(recent))
            for n in NUMBERS:
                row[f"freq_w{w}_{n}"] = float(freq[n - 1])
        for n in NUMBERS:
            row[f"gap_{n}"] = float(i + 1 if last_seen[n - 1] < 0 else i - last_seen[n - 1])
        row["sum"] = float(nums.sum())
        row["mean"] = float(nums.mean())
        row["std"] = float(nums.std())
        row["odd_ratio"] = float(np.mean(nums % 2 == 1))
        row["low_ratio"] = float(np.mean(nums <= 22))
        row["high_ratio"] = float(np.mean(nums >= 23))
        row["consecutive_count"] = _consecutive_count(nums)
        row["bonus_seen_recent_10"] = float(bonus_matrix[max(0, i - 9) : i + 1, int(raw.bonus) - 1].sum())
        for n in NUMBERS:
            series = hist[:, n - 1]
            tail10 = series[max(0, len(series) - 10) :]
            tail30 = series[max(0, len(series) - 30) :]
            row[f"trend_10_30_{n}"] = float(tail10.mean() - tail30.mean())
            row[f"roll_count_20_{n}"] = float(series[max(0, len(series) - 20) :].sum())
            row[f"ewma_20_{n}"] = float(pd.Series(series).ewm(span=20, adjust=False).mean().iloc[-1])
        rows.append(row)
        for n in nums:
            last_seen[n - 1] = i
    return pd.DataFrame(rows)


def build_supervised_tabular(df: pd.DataFrame, features: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    x = features.iloc[:-1].drop(columns=["draw_no", "draw_date"], errors="ignore").copy()
    y = np.vstack([numbers_to_multihot(row) for row in df[WIN_COLS].iloc[1:].to_numpy()])
    return x, y.astype(np.float32)


def latest_feature(features: pd.DataFrame) -> pd.DataFrame:
    return features.tail(1).drop(columns=["draw_no", "draw_date"], errors="ignore").copy()
