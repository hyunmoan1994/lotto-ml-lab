from __future__ import annotations

import argparse

from src.collector import load_or_collect
from src.dataset import build_sequence_dataset, latest_sequence
from src.feature_engineering import build_feature_dataframe, build_supervised_tabular, latest_feature
from src.predict import collect_latest_scores
from src.preprocess import validate_raw_data
from src.recommend import build_recommendations
from src.train_dl import train_all_dl
from src.train_ml import train_all_ml
from src.utils import ensure_dirs, set_seed


def run_pipeline(epochs: int = 4, seq_len: int = 20) -> None:
    ensure_dirs()
    set_seed(42)
    raw = validate_raw_data(load_or_collect(), min_rows=300)
    features = build_feature_dataframe(raw)
    x_tab, y_tab = build_supervised_tabular(raw, features)
    x_seq, y_seq = build_sequence_dataset(raw, seq_len=seq_len)
    dl_results = train_all_dl(x_seq, y_seq, epochs=epochs)
    ml_results = train_all_ml(x_tab, y_tab)
    scores = collect_latest_scores(dl_results, ml_results, latest_feature(features), latest_sequence(raw, seq_len))
    rec_df, _score_df = build_recommendations(scores)
    print("Recommendations")
    print(rec_df.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=20)
    args = parser.parse_args()
    run_pipeline(epochs=args.epochs, seq_len=args.seq_len)
