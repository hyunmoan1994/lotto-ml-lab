from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.dataset import build_sequence_dataset
from src.feature_engineering import build_feature_dataframe, build_supervised_tabular
from src.preprocess import validate_raw_data
from src.recommend import build_recommendations


def _sample_df(rows: int = 80) -> pd.DataFrame:
    data = []
    for i in range(rows):
        nums = sorted((((i * 7 + j * 5) % 45) + 1 for j in range(6)))
        data.append({"draw_no": i + 1, "draw_date": "2024-01-01", "n1": nums[0], "n2": nums[1], "n3": nums[2], "n4": nums[3], "n5": nums[4], "n6": nums[5], "bonus": ((i * 11) % 45) + 1})
    return pd.DataFrame(data)


def test_feature_sequence_and_recommendation_pipeline_shapes():
    raw = validate_raw_data(_sample_df(), min_rows=50)
    features = build_feature_dataframe(raw)
    x_tab, y_tab = build_supervised_tabular(raw, features)
    x_seq, y_seq = build_sequence_dataset(raw, seq_len=20)
    assert len(features) == len(raw)
    assert x_tab.shape[0] == y_tab.shape[0] == len(raw) - 1
    assert x_seq.shape[1:] == (20, 45)
    assert y_seq.shape[1] == 45
    rec, scores = build_recommendations(
        {"RNN": y_seq[-1], "LSTM": y_seq[-2], "Transformer": y_seq[-3], "RandomForest": y_seq[-4], "MLP": y_seq[-5]},
        sets_per_model={"RNN": 3, "LSTM": 2, "Transformer": 2, "RandomForest": 2, "MLP": 1},
    )
    assert rec.shape[0] == 10
    assert scores.shape[0] == 225
