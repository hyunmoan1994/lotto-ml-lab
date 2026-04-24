from __future__ import annotations

import numpy as np
import pandas as pd


NUMBERS = list(range(1, 46))
WIN_COLS = ["n1", "n2", "n3", "n4", "n5", "n6"]


def validate_raw_data(df: pd.DataFrame, min_rows: int = 50) -> pd.DataFrame:
    required = {"draw_no", "draw_date", *WIN_COLS, "bonus"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"필수 컬럼 누락: {sorted(missing)}")
    if len(df) < min_rows:
        raise ValueError(f"학습에는 최소 {min_rows}개 회차가 필요합니다.")
    out = df.sort_values("draw_no").reset_index(drop=True).copy()
    for col in WIN_COLS + ["bonus"]:
        if not out[col].between(1, 45).all():
            raise ValueError(f"{col} 컬럼에 1~45 범위를 벗어난 값이 있습니다.")
    return out


def numbers_to_multihot(numbers: list[int] | np.ndarray) -> np.ndarray:
    y = np.zeros(45, dtype=np.float32)
    for n in numbers:
        if 1 <= int(n) <= 45:
            y[int(n) - 1] = 1.0
    return y


def draw_matrix(df: pd.DataFrame, include_bonus: bool = False) -> np.ndarray:
    cols = WIN_COLS + (["bonus"] if include_bonus else [])
    return np.vstack([numbers_to_multihot(row) for row in df[cols].to_numpy()])
