"""Binary classification metrics, bootstrap intervals, and paired DeLong tests."""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    roc_auc_score,
)


def binary_metrics(
    labels: np.ndarray, probabilities: np.ndarray, threshold: float = 0.5
) -> dict[str, float | int | None]:
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    mcc_denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / mcc_denominator if mcc_denominator else 0.0
    auc = float(roc_auc_score(labels, probabilities)) if np.unique(labels).size == 2 else None
    return {
        "auc": auc,
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float((sensitivity + specificity) / 2.0),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "mcc": float(mcc),
        "youden_index": float(sensitivity + specificity - 1.0),
        "threshold": float(threshold),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def bootstrap_interval(
    labels: np.ndarray,
    probabilities: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float],
    *,
    samples: int = 1000,
    confidence: float = 0.95,
    seed: int = 513,
) -> dict[str, float | None]:
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities)
    generator = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(samples):
        indices = generator.integers(0, len(labels), len(labels))
        boot_labels = labels[indices]
        if np.unique(boot_labels).size < 2:
            continue
        values.append(float(metric(boot_labels, probabilities[indices])))
    if not values:
        return {"estimate": None, "lower": None, "upper": None}
    alpha = (1.0 - confidence) / 2.0
    return {
        "estimate": float(metric(labels, probabilities)),
        "lower": float(np.quantile(values, alpha)),
        "upper": float(np.quantile(values, 1.0 - alpha)),
    }


def find_optimal_thresholds(
    labels: np.ndarray, probabilities: np.ndarray, grid_size: int = 1001
) -> dict[str, dict[str, float | dict]]:
    thresholds = np.linspace(0.0, 1.0, grid_size)
    best = {
        "youden": {"value": -math.inf},
        "f1": {"value": -math.inf},
        "balanced": {"value": math.inf},
    }
    for threshold in thresholds:
        metrics = binary_metrics(labels, probabilities, float(threshold))
        if metrics["youden_index"] > best["youden"]["value"]:
            best["youden"] = {
                "value": metrics["youden_index"],
                "threshold": float(threshold),
                "metrics": metrics,
            }
        if metrics["f1"] > best["f1"]["value"]:
            best["f1"] = {
                "value": metrics["f1"],
                "threshold": float(threshold),
                "metrics": metrics,
            }
        difference = abs(float(metrics["sensitivity"]) - float(metrics["specificity"]))
        if difference < best["balanced"]["value"]:
            best["balanced"] = {
                "value": difference,
                "threshold": float(threshold),
                "metrics": metrics,
            }
    return best


def _midrank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    sorted_values = values[order]
    ranks = np.zeros(len(values), dtype=float)
    index = 0
    while index < len(values):
        end = index
        while end < len(values) and sorted_values[end] == sorted_values[index]:
            end += 1
        ranks[index:end] = 0.5 * (index + end - 1) + 1
        index = end
    result = np.empty(len(values), dtype=float)
    result[order] = ranks
    return result


def _fast_delong(
    predictions_sorted: np.ndarray, positive_count: int
) -> tuple[np.ndarray, np.ndarray]:
    classifiers, total = predictions_sorted.shape
    negative_count = total - positive_count
    positive = predictions_sorted[:, :positive_count]
    negative = predictions_sorted[:, positive_count:]
    positive_ranks = np.empty_like(positive, dtype=float)
    negative_ranks = np.empty_like(negative, dtype=float)
    total_ranks = np.empty_like(predictions_sorted, dtype=float)
    for classifier in range(classifiers):
        positive_ranks[classifier] = _midrank(positive[classifier])
        negative_ranks[classifier] = _midrank(negative[classifier])
        total_ranks[classifier] = _midrank(predictions_sorted[classifier])
    aucs = total_ranks[:, :positive_count].sum(axis=1) / (positive_count * negative_count) - (
        positive_count + 1.0
    ) / (2.0 * negative_count)
    v01 = (total_ranks[:, :positive_count] - positive_ranks) / negative_count
    v10 = 1.0 - (total_ranks[:, positive_count:] - negative_ranks) / positive_count
    sx = np.atleast_2d(np.cov(v01, bias=False))
    sy = np.atleast_2d(np.cov(v10, bias=False))
    covariance = sx / positive_count + sy / negative_count
    return aucs, covariance


def paired_delong_test(
    labels: np.ndarray, probabilities_a: np.ndarray, probabilities_b: np.ndarray
) -> dict[str, float]:
    """Perform a paired DeLong test for two ROC AUCs."""

    labels = np.asarray(labels, dtype=int)
    if np.unique(labels).tolist() != [0, 1]:
        raise ValueError("Paired DeLong requires both binary classes")
    probabilities = np.vstack(
        [np.asarray(probabilities_a, dtype=float), np.asarray(probabilities_b, dtype=float)]
    )
    order = np.argsort(-labels)
    positive_count = int(labels.sum())
    aucs, covariance = _fast_delong(probabilities[:, order], positive_count)
    difference = float(aucs[0] - aucs[1])
    contrast = np.array([1.0, -1.0])
    variance = float(contrast @ covariance @ contrast.T)
    if variance <= 0:
        p_value = 1.0 if difference == 0 else 0.0
        z_score = 0.0 if difference == 0 else math.copysign(math.inf, difference)
    else:
        z_score = difference / math.sqrt(variance)
        p_value = float(2.0 * stats.norm.sf(abs(z_score)))
    return {
        "auc_a": float(aucs[0]),
        "auc_b": float(aucs[1]),
        "difference": difference,
        "z_score": float(z_score),
        "p_value": p_value,
    }


def summarize_folds(fold_metrics: list[dict[str, float | int | None]]) -> dict[str, dict]:
    """Compute mean, sample SD, and normal-approximation 95% CI across folds."""

    summary: dict[str, dict] = {}
    numeric_keys = sorted(
        {
            key
            for metrics in fold_metrics
            for key, value in metrics.items()
            if key != "fold" and isinstance(value, (int, float)) and value is not None
        }
    )
    for key in numeric_keys:
        values = np.asarray(
            [metrics[key] for metrics in fold_metrics if metrics.get(key) is not None],
            dtype=float,
        )
        if not values.size:
            continue
        std = float(values.std(ddof=1)) if values.size > 1 else 0.0
        margin = 1.96 * std / math.sqrt(values.size)
        mean = float(values.mean())
        summary[key] = {
            "mean": mean,
            "std": std,
            "ci95_lower": mean - margin,
            "ci95_upper": mean + margin,
            "folds": int(values.size),
        }
    return summary
