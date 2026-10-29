"""Fit clinical late-fusion models using prepared OOF/averaged image scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import ParameterSampler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from petct.metrics import binary_metrics

from .dataset import CATEGORICAL_FEATURES, NUMERIC_FEATURES, load_pcr_dataset
from .prepare_inputs import (
    EXTERNAL_COHORTS,
    ID_COLUMN,
    LABEL_COLUMN,
    NOTICE,
    PROBABILITY_COLUMN,
    prediction_export,
)


def build_pipeline(name: str, seed: int = 42) -> Pipeline:
    """Fit all preprocessing only on the fusion training cohort."""
    preprocess = ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                        ("scaler", StandardScaler()),
                    ]
                ),
                NUMERIC_FEATURES,
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
    if name == "logistic":
        classifier = LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            max_iter=4000,
            random_state=seed,
        )
    elif name == "rf":
        classifier = RandomForestClassifier(random_state=seed, n_jobs=1)
    elif name == "lgbm":
        from lightgbm import LGBMClassifier

        classifier = LGBMClassifier(
            objective="binary",
            random_state=seed,
            n_jobs=1,
            verbosity=-1,
        )
    elif name == "xgb":
        from xgboost import XGBClassifier

        classifier = XGBClassifier(eval_metric="logloss", random_state=seed, n_jobs=1)
    else:
        raise ValueError(f"Unknown fusion model: {name}")
    return Pipeline([("preprocess", preprocess), ("clf", classifier)])


def parameter_space(name: str) -> dict:
    """Retain the original random-search spaces without their mutable pipeline bug."""
    spaces = {
        "logistic": {
            "C": np.logspace(-3, 1.5, 30),
            "l1_ratio": np.linspace(0.1, 0.9, 9),
            "class_weight": [None, "balanced"],
        },
        "rf": {
            "n_estimators": [200, 400, 600, 800, 1200],
            "max_depth": [2, 3, 4, 5, 6, None],
            "min_samples_leaf": [3, 5, 8, 12, 20],
            "min_samples_split": [5, 10, 20, 40],
            "max_features": ["sqrt", "log2"],
            "class_weight": [None, "balanced"],
        },
        "lgbm": {
            "n_estimators": [300, 500, 800, 1200],
            "learning_rate": [0.005, 0.01, 0.02, 0.05],
            "max_depth": [2, 3, 4, 5, 6],
            "num_leaves": [7, 15, 31, 63],
            "min_child_samples": [5, 10, 20, 40],
            "reg_alpha": [0.0, 0.1, 0.5, 1.0],
            "reg_lambda": [0.5, 1, 3, 5, 10],
            "subsample": [0.6, 0.7, 0.8, 0.9],
            "colsample_bytree": [0.6, 0.7, 0.8, 0.9],
        },
        "xgb": {
            "n_estimators": [300, 500, 800],
            "learning_rate": [0.01, 0.02, 0.05],
            "max_depth": [3, 4, 5],
            "min_child_weight": [5, 8, 12],
            "gamma": [0, 0.3, 1.0],
            "subsample": [0.7, 0.8, 0.9],
            "colsample_bytree": [0.7, 0.8, 0.9],
            "reg_lambda": [1, 3, 5],
        },
    }
    return {f"clf__{key}": values for key, values in spaces[name].items()}


def select_model(pipeline, parameters, x_train, y_train, x_validation, y_validation, top_k):
    """Select without receiving external data; clone each candidate before fitting."""
    candidates = []
    for index, params in enumerate(parameters, 1):
        model = clone(pipeline).set_params(**params)
        model.fit(x_train, y_train)
        train_auc = float(roc_auc_score(y_train, model.predict_proba(x_train)[:, 1]))
        validation_auc = float(roc_auc_score(y_validation, model.predict_proba(x_validation)[:, 1]))
        candidates.append(
            {
                "model": model,
                "trial": index,
                "params": params,
                "training_auc": train_auc,
                "validation_auc": validation_auc,
                "gap": train_auc - validation_auc,
            }
        )
    if not candidates or top_k < 1:
        raise ValueError("At least one trial and top_k >= 1 are required")
    top = sorted(candidates, key=lambda item: -item["validation_auc"])[:top_k]
    selected = min(top, key=lambda item: item["gap"])
    return selected, [
        {key: value for key, value in item.items() if key != "model"} for item in candidates
    ]


def fit_fusion(input_path: Path, output: Path, names, iterations=40, top_k=10, seed=42):
    if output.exists():
        raise FileExistsError("Choose a new output directory; existing results are not overwritten")
    if iterations < 1 or top_k < 1:
        raise ValueError("iterations and top_k must be positive")
    if len(names) != len(set(names)) or not names:
        raise ValueError("Choose distinct model names")
    features, labels, cohort, ids = load_pcr_dataset(input_path, include_patient_id=True)
    if not np.isfinite(features["Predicted_score"]).all():
        raise ValueError("Every fusion case needs a finite image score")
    if not features["Predicted_score"].between(0, 1).all():
        raise ValueError("Image scores must lie within [0, 1]")
    train, validation = cohort.eq("Training"), cohort.eq("Test")
    for mask in (train, validation):
        if labels[mask].nunique() != 2:
            raise ValueError("Training and validation cohorts must both contain two classes")
    # Check optional dependencies before creating any output directories.
    pipelines = {name: build_pipeline(name, seed) for name in names}
    summary_path = input_path.parent / "input_summary.json"
    provenance = (
        json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    )
    result = {
        "seed": seed,
        "features": list(features.columns),
        "notice": NOTICE,
        "selection": {
            "iterations": iterations,
            "top_k": top_k,
            "policy": "highest validation AUC top-k, then minimum train-val gap",
        },
        "input_provenance": provenance,
        "models": {},
    }
    for name, pipeline in pipelines.items():
        selected, trials = select_model(
            pipeline,
            ParameterSampler(parameter_space(name), n_iter=iterations, random_state=seed),
            features[train],
            labels[train],
            features[validation],
            labels[validation],
            top_k,
        )
        model = selected["model"]
        directory = output / name
        directory.mkdir(parents=True)
        metrics = {}
        partitions = {
            "train": train,
            "validation": validation,
            **{key: cohort.eq(value) for key, value in EXTERNAL_COHORTS.items()},
        }
        for partition, mask in partitions.items():
            if not mask.any():
                raise ValueError(f"Missing fusion partition: {partition}")
            probability = model.predict_proba(features[mask])[:, 1]
            metrics[partition] = binary_metrics(labels[mask].to_numpy(), probability)
            predictions = pd.DataFrame(
                {
                    ID_COLUMN: ids[mask],
                    LABEL_COLUMN: labels[mask].to_numpy(),
                    PROBABILITY_COLUMN: probability,
                }
            )
            prediction_export(predictions).to_csv(
                directory / f"predictions_{partition}.csv",
                index=False,
                encoding="utf-8-sig",
            )
        metadata = {key: value for key, value in selected.items() if key != "model"}
        metadata.update({"metrics": metrics, "trials": trials})
        (directory / "metrics.json").write_text(
            json.dumps(metadata, indent=2, default=_json_value, allow_nan=False),
            encoding="utf-8",
        )
        result["models"][name] = metadata
        print(f"{name}: validation AUC={metrics['validation']['auc']:.4f}")
    (output / "summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=_json_value, allow_nan=False),
        encoding="utf-8",
    )
    return result


def _json_value(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["lgbm", "logistic", "rf", "xgb"],
        default=["lgbm", "logistic", "rf", "xgb"],
    )
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    fit_fusion(args.input, args.output, args.models, args.iterations, args.top_k, args.seed)


if __name__ == "__main__":
    main()
