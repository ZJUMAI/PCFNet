"""Configurable patient-level stratified manifest generation."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
from sklearn.model_selection import StratifiedKFold

from .config import resolve_config_path
from .data import parse_binary_label


def run_split(config: dict[str, Any]) -> dict[str, Any]:
    settings = config.get("split", {})
    metadata_path = resolve_config_path(config, settings["metadata"])
    suffix = metadata_path.suffix.lower()
    if suffix == ".xlsx":
        frame = pd.read_excel(metadata_path)
    elif suffix == ".csv":
        frame = pd.read_csv(metadata_path)
    else:
        raise ValueError("split.metadata must be a CSV or XLSX file")
    id_column = str(settings["id_column"])
    label_column = str(settings["label_column"])
    missing = {id_column, label_column} - set(frame.columns)
    if missing:
        raise ValueError(f"Metadata is missing columns: {sorted(missing)}")
    frame = frame[[id_column, label_column]].copy()
    frame[id_column] = frame[id_column].astype(str).str.strip()
    if frame[id_column].duplicated().any():
        raise ValueError("Metadata contains duplicate patient IDs")
    labels = frame[label_column].map(parse_binary_label).to_numpy()
    identifiers = frame[id_column].to_numpy()
    folds = int(settings.get("folds", 10))
    splitter = StratifiedKFold(
        n_splits=folds, shuffle=True, random_state=int(settings.get("seed", 42))
    )
    output = resolve_config_path(config, settings.get("output_directory", "dataset/splits"))
    output.mkdir(parents=True, exist_ok=True)
    pattern = str(settings.get("filename_pattern", "fold_{fold}_{split}.txt"))
    for fold, (train_indices, validation_indices) in enumerate(
        splitter.split(identifiers, labels), start=1
    ):
        for split, indices in (("train", train_indices), ("validation", validation_indices)):
            path = output / pattern.format(fold=fold, split=split)
            path.write_text("\n".join(identifiers[indices]) + "\n", encoding="utf-8")
    summary = {
        "cases": int(len(frame)),
        "folds": folds,
        "positive": int(labels.sum()),
        "negative": int((labels == 0).sum()),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
