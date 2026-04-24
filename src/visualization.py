from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .preprocess import WIN_COLS


def frequency_chart(df: pd.DataFrame):
    counts = pd.Series(df[WIN_COLS].to_numpy().ravel()).value_counts().reindex(range(1, 46), fill_value=0)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(counts.index, counts.values, color="#2f6f73")
    ax.set_title("Number Frequency")
    ax.set_xlabel("Number")
    ax.set_ylabel("Count")
    return fig


def gap_chart(features: pd.DataFrame):
    gap_cols = [f"gap_{i}" for i in range(1, 46)]
    latest = features[gap_cols].tail(1).to_numpy().ravel()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(1, 46), latest, color="#b45f4d")
    ax.set_title("Current Gap by Number")
    ax.set_xlabel("Number")
    ax.set_ylabel("Draws Since Last Seen")
    return fig


def top_score_chart(score_df: pd.DataFrame, model: str):
    part = score_df[score_df["model"] == model].sort_values("score", ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(part["number"].astype(str), part["score"], color="#4f6fb5")
    ax.set_title(f"{model} Top-10 Scores")
    ax.set_xlabel("Number")
    ax.set_ylabel("Score")
    return fig


def metrics_chart(metrics_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 4))
    plot_df = metrics_df.set_index("model")[["hit@6_avg", "precision@6", "recall@6"]]
    plot_df.plot(kind="bar", ax=ax)
    ax.set_title("Validation Metrics")
    ax.set_ylabel("Score")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    return fig
