from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .dataset import chronological_split
from .evaluate import evaluate_scores
from .model_registry import ML_MODEL_NAMES
from .utils import MODEL_DIR


def _predict_multi_scores(model, x) -> np.ndarray:
    probas = model.predict_proba(x)
    if isinstance(probas, list):
        classes = getattr(model, "classes_", [None] * len(probas))
        cols = []
        for p, cls in zip(probas, classes):
            if cls is not None and 1 in cls:
                cols.append(p[:, list(cls).index(1)])
            elif p.shape[1] > 1:
                cols.append(p[:, 1])
            elif cls is not None and len(cls) == 1 and int(cls[0]) == 1:
                cols.append(np.ones(len(x)))
            else:
                cols.append(np.zeros(len(x)))
        return np.vstack(cols).T
    return np.asarray(probas)


def train_random_forest(x, y, save_path: Path = MODEL_DIR / "random_forest.joblib") -> dict:
    tr, va, te = chronological_split(len(x))
    model = RandomForestClassifier(n_estimators=160, max_depth=10, min_samples_leaf=2, n_jobs=-1, random_state=42)
    model.fit(x.iloc[tr], y[tr])
    valid_scores = _predict_multi_scores(model, x.iloc[va])
    test_scores = _predict_multi_scores(model, x.iloc[te]) if len(x.iloc[te]) else valid_scores
    joblib.dump(model, save_path)
    return {"name": "RandomForest", "model": model, "valid_metrics": evaluate_scores(y[va], valid_scores), "test_metrics": evaluate_scores(y[te], test_scores) if len(y[te]) else {}}


def train_modern_model(x, y, save_path: Path = MODEL_DIR / "mlp_modern.joblib") -> dict:
    tr, va, te = chronological_split(len(x))
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            ("clf", MultiOutputClassifier(MLPClassifier(hidden_layer_sizes=(96, 48), max_iter=120, random_state=42, early_stopping=True))),
        ]
    )
    model.fit(x.iloc[tr], y[tr])
    valid_scores = _predict_multi_scores(model.named_steps["clf"], model.named_steps["scale"].transform(x.iloc[va]))
    test_scores = _predict_multi_scores(model.named_steps["clf"], model.named_steps["scale"].transform(x.iloc[te])) if len(x.iloc[te]) else valid_scores
    joblib.dump(model, save_path)
    return {"name": "MLP", "model": model, "valid_metrics": evaluate_scores(y[va], valid_scores), "test_metrics": evaluate_scores(y[te], test_scores) if len(y[te]) else {}}


def _train_multioutput_model(name: str, model, x, y, save_path: Path) -> dict:
    tr, va, te = chronological_split(len(x))
    model.fit(x.iloc[tr], y[tr])
    valid_scores = _predict_multi_scores(model, x.iloc[va])
    test_scores = _predict_multi_scores(model, x.iloc[te]) if len(x.iloc[te]) else valid_scores
    joblib.dump(model, save_path)
    return {
        "name": name,
        "model": model,
        "valid_metrics": evaluate_scores(y[va], valid_scores),
        "test_metrics": evaluate_scores(y[te], test_scores) if len(y[te]) else {},
    }


def train_extra_trees(x, y, save_path: Path = MODEL_DIR / "extra_trees.joblib") -> dict:
    model = ExtraTreesClassifier(
        n_estimators=220,
        max_depth=12,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42,
    )
    return _train_multioutput_model("ExtraTrees", model, x, y, save_path)


def train_gradient_boosting(x, y, save_path: Path = MODEL_DIR / "gradient_boosting.joblib") -> dict:
    model = MultiOutputClassifier(
        GradientBoostingClassifier(n_estimators=60, learning_rate=0.05, max_depth=2, random_state=42),
        n_jobs=-1,
    )
    return _train_multioutput_model("GradientBoosting", model, x, y, save_path)


def train_adaboost(x, y, save_path: Path = MODEL_DIR / "adaboost.joblib") -> dict:
    model = MultiOutputClassifier(AdaBoostClassifier(n_estimators=80, learning_rate=0.5, random_state=42), n_jobs=-1)
    return _train_multioutput_model("AdaBoost", model, x, y, save_path)


def train_logistic_regression(x, y, save_path: Path = MODEL_DIR / "logistic_regression.joblib") -> dict:
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "clf",
                MultiOutputClassifier(
                    LogisticRegression(max_iter=400, class_weight="balanced", solver="liblinear", random_state=42),
                    n_jobs=-1,
                ),
            ),
        ]
    )
    return _train_pipeline_model("LogisticRegression", model, x, y, save_path)


def train_knn(x, y, save_path: Path = MODEL_DIR / "knn.joblib") -> dict:
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            ("clf", MultiOutputClassifier(KNeighborsClassifier(n_neighbors=15, weights="distance"), n_jobs=-1)),
        ]
    )
    return _train_pipeline_model("KNN", model, x, y, save_path)


def _train_pipeline_model(name: str, model: Pipeline, x, y, save_path: Path) -> dict:
    tr, va, te = chronological_split(len(x))
    model.fit(x.iloc[tr], y[tr])
    valid_scores = _predict_multi_scores(model.named_steps["clf"], model.named_steps["scale"].transform(x.iloc[va]))
    test_scores = (
        _predict_multi_scores(model.named_steps["clf"], model.named_steps["scale"].transform(x.iloc[te]))
        if len(x.iloc[te])
        else valid_scores
    )
    joblib.dump(model, save_path)
    return {
        "name": name,
        "model": model,
        "valid_metrics": evaluate_scores(y[va], valid_scores),
        "test_metrics": evaluate_scores(y[te], test_scores) if len(y[te]) else {},
    }


def train_selected_ml(x, y, model_names: list[str]) -> list[dict]:
    trainers = {
        "RandomForest": train_random_forest,
        "MLP": train_modern_model,
        "ExtraTrees": train_extra_trees,
        "GradientBoosting": train_gradient_boosting,
        "AdaBoost": train_adaboost,
        "LogisticRegression": train_logistic_regression,
        "KNN": train_knn,
    }
    return [trainers[name](x, y) for name in model_names if name in trainers]


def train_all_ml(x, y) -> list[dict]:
    return train_selected_ml(x, y, ML_MODEL_NAMES)


def predict_ml_scores(model, x_latest) -> np.ndarray:
    if isinstance(model, Pipeline):
        return _predict_multi_scores(model.named_steps["clf"], model.named_steps["scale"].transform(x_latest))[0]
    return _predict_multi_scores(model, x_latest)[0]
