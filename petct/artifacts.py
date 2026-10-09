"""Privacy-preserving experiment artifact management."""

from __future__ import annotations

import csv
import json
import logging
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import yaml

from petct.config import public_config, resolve_config_path

if TYPE_CHECKING:
    import torch


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, Path):
        return str(value)
    return value


def prepare_run_directory(config: dict[str, Any], suffix: str | None = None) -> Path:
    output = config.get("output", {})
    root = resolve_config_path(config, output.get("root", "../runs"))
    run_name = output.get("run_name")
    if not run_name:
        run_name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_name = str(run_name)
    if Path(run_name).name != run_name or Path(run_name).is_absolute():
        raise ValueError("output.run_name must be a single directory name")
    if suffix:
        run_name = f"{run_name}_{suffix}"
    directory = root / run_name
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def configure_logging(run_directory: Path) -> logging.Logger:
    logger = logging.getLogger(f"petct.{run_directory}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(run_directory / "run.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def save_resolved_config(config: dict[str, Any], run_directory: Path) -> None:
    with (run_directory / "resolved_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(public_config(config), handle, sort_keys=False, allow_unicode=True)


def save_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(payload), handle, indent=2, ensure_ascii=False)


def save_history(path: Path, history: list[dict[str, Any]]) -> None:
    if not history:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    model_config: dict[str, Any],
    epoch: int,
    validation_metrics: dict[str, Any],
) -> None:
    """Save only the portable model state dict (no private data or pickled model)."""

    import torch

    del model_config, epoch, validation_metrics
    torch.save(model.state_dict(), path)


def save_predictions(
    path: Path,
    labels: np.ndarray,
    probabilities: np.ndarray,
    enabled: bool,
    *,
    identifiers: Sequence[str] | None = None,
    threshold: float = 0.5,
) -> None:
    """Save anonymous predictions or an explicitly requested identified test table.

    Probabilities always refer to class 1 (pCR). Identified tables contain the
    case ID, correctness flag, positive probability, predicted class, and label.
    """

    if not enabled:
        return
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities, dtype=float)
    if labels.ndim != 1 or probabilities.ndim != 1 or len(labels) != len(probabilities):
        raise ValueError("Prediction labels and probabilities must be equally sized 1D arrays")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Prediction labels must be binary 0/1 values")
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Prediction probabilities must be finite values between 0 and 1")
    if not 0 <= threshold <= 1:
        raise ValueError("Prediction threshold must be between 0 and 1")
    labels = labels.astype(int)
    if identifiers is not None:
        identifiers = list(identifiers)
        if len(identifiers) != len(labels):
            raise ValueError("Prediction identifiers must match the number of predictions")
        if any(
            not isinstance(identifier, str) or not identifier.strip() for identifier in identifiers
        ):
            raise ValueError("Prediction identifiers must be non-empty strings")
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("Prediction identifiers must be unique within a cohort")
    encoding = "utf-8-sig" if identifiers is not None else "utf-8"
    with path.open("w", encoding=encoding, newline="") as handle:
        writer = csv.writer(handle)
        if identifiers is None:
            writer.writerow(["label", "probability"])
            writer.writerows(zip(labels, probabilities, strict=True))
        else:
            predictions = (probabilities >= threshold).astype(int)
            correct = (predictions == labels).astype(int)
            writer.writerow(
                ["影像组学序列号", "是否预测成功", "预测概率", "预测结果", "ground truth"]
            )
            writer.writerows(
                zip(identifiers, correct, probabilities, predictions, labels, strict=True)
            )
