"""Aggregate evaluation, threshold selection, fold summaries, and paired DeLong tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .config import resolve_config_path
from .metrics import binary_metrics, find_optimal_thresholds, paired_delong_test, summarize_folds


def _prediction_columns(config: dict[str, Any]) -> tuple[str, str]:
    return str(config.get("label_column", "label")), str(
        config.get("probability_column", "probability")
    )


def _read(
    path: str | Path, label_column: str, probability_column: str
) -> tuple[Any, Any, list[str] | None]:
    frame = pd.read_csv(path, dtype={"影像组学序列号": str}, keep_default_na=False)
    if label_column == "label" and label_column not in frame and "ground truth" in frame:
        label_column = "ground truth"
    if (
        probability_column == "probability"
        and probability_column not in frame
        and "预测概率" in frame
    ):
        probability_column = "预测概率"
    missing = {label_column, probability_column} - set(frame.columns)
    if missing:
        raise ValueError(f"Prediction file {path} is missing columns: {sorted(missing)}")
    identifiers = None
    if "影像组学序列号" in frame:
        identifiers = frame["影像组学序列号"].tolist()
        if any(not identifier.strip() for identifier in identifiers):
            raise ValueError("Prediction identifiers must not be empty")
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("Prediction identifiers must be unique within a cohort")
    return frame[label_column].to_numpy(), frame[probability_column].to_numpy(), identifiers


def _metric_record(payload: dict[str, Any]) -> dict[str, Any]:
    """Extract validation metrics from either a flat or run-level result file."""

    if isinstance(payload.get("validation"), dict):
        return payload["validation"]
    if isinstance(payload.get("results"), dict) and isinstance(
        payload["results"].get("validation"), dict
    ):
        return payload["results"]["validation"]
    return payload


def run_evaluation(config: dict[str, Any]) -> dict[str, Any]:
    evaluation = config.get("evaluate", {})
    task = str(evaluation.get("task", "metrics"))
    label_column, probability_column = _prediction_columns(evaluation)
    result: dict[str, Any]
    if task in {"metrics", "youden"}:
        labels, probabilities, _ = _read(
            resolve_config_path(config, evaluation["predictions"]), label_column, probability_column
        )
        if task == "youden":
            threshold = float(find_optimal_thresholds(labels, probabilities)["youden"]["threshold"])
        else:
            threshold = float(evaluation.get("threshold", 0.5))
        result = binary_metrics(labels, probabilities, threshold=threshold)
    elif task == "crossval":
        if evaluation.get("crossval_file"):
            payload = json.loads(
                resolve_config_path(config, evaluation["crossval_file"]).read_text(encoding="utf-8")
            )
            result = (
                payload["summary"]
                if isinstance(payload.get("summary"), dict)
                else summarize_folds(payload["folds"])
            )
        else:
            fold_metrics = [
                _metric_record(
                    json.loads(resolve_config_path(config, path).read_text(encoding="utf-8"))
                )
                for path in evaluation["metric_files"]
            ]
            result = summarize_folds(fold_metrics)
    elif task == "delong":
        labels_a, probabilities_a, identifiers_a = _read(
            resolve_config_path(config, evaluation["predictions_a"]),
            label_column,
            probability_column,
        )
        labels_b, probabilities_b, identifiers_b = _read(
            resolve_config_path(config, evaluation["predictions_b"]),
            label_column,
            probability_column,
        )
        if identifiers_a != identifiers_b:
            raise ValueError(
                "Paired DeLong inputs must contain identical IDs in identical row order"
            )
        if len(labels_a) != len(labels_b) or not (labels_a == labels_b).all():
            raise ValueError(
                "Paired DeLong inputs must contain identical labels in identical row order"
            )
        result = paired_delong_test(labels_a, probabilities_a, probabilities_b)
    elif task == "checkpoint":
        import torch

        from .artifacts import save_predictions
        from .data import loader_from_config, prediction_identifiers
        from .engine import build_criterion, evaluate_loader, resolve_device
        from .models import build_model

        device = resolve_device(config.get("device", "auto"))
        model = build_model(config).to(device)
        state_dict = torch.load(
            resolve_config_path(config, evaluation["checkpoint"]),
            map_location=device,
            weights_only=True,
        )
        model.load_state_dict(state_dict)
        criterion = build_criterion(config)
        result = {}
        prediction_directory = resolve_config_path(
            config, evaluation.get("prediction_directory", "evaluation_predictions")
        )
        for cohort_name, split_config in config.get("data", {}).get("external", {}).items():
            loader = loader_from_config(config, split_config, training=False)
            metrics, labels, probabilities = evaluate_loader(model, loader, criterion, device)
            result[cohort_name] = metrics
            if bool(evaluation.get("save_predictions", False)):
                prediction_directory.mkdir(parents=True, exist_ok=True)
                save_predictions(
                    prediction_directory / f"{cohort_name}.csv",
                    labels,
                    probabilities,
                    True,
                    identifiers=prediction_identifiers(
                        loader, bool(evaluation.get("save_prediction_ids", False))
                    ),
                    threshold=float(metrics.get("threshold", 0.5)),
                )
    else:
        raise ValueError("evaluate.task must be metrics, youden, crossval, delong, or checkpoint")

    output = resolve_config_path(config, evaluation.get("output", "evaluation_metrics.json"))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result
