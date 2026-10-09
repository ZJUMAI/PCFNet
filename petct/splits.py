"""Configurable patient-level stratified manifest generation."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split

from .config import ConfigError, resolve_config_path
from .metadata import load_metadata, parse_binary_label


def run_split(config: dict[str, Any]) -> dict[str, Any]:
    settings = config.get("split", {})
    metadata_path = resolve_config_path(config, settings["metadata"])
    id_column = str(settings["id_column"])
    label_column = str(settings["label_column"])
    frame = load_metadata(metadata_path, id_column, label_column)
    source_manifests = settings.get("source_manifests")
    if source_manifests is not None:
        if not isinstance(source_manifests, list) or not source_manifests:
            raise ConfigError("split.source_manifests must be a non-empty list")
        identifiers = []
        seen = set()
        for manifest in source_manifests:
            path = resolve_config_path(config, manifest)
            for identifier in path.read_text(encoding="utf-8").splitlines():
                identifier = identifier.strip()
                if identifier and identifier not in seen:
                    identifiers.append(identifier)
                    seen.add(identifier)
        if not identifiers:
            raise ConfigError("split.source_manifests contain no case IDs")
        missing = seen - set(frame.index)
        if missing:
            raise ConfigError(
                f"{len(missing)} source-manifest cases are absent from the label table"
            )
        frame = frame.loc[identifiers]
    labels = frame[label_column].map(parse_binary_label).to_numpy()
    identifiers = frame.index.to_numpy()
    mode = str(settings.get("mode", "kfold"))
    seed = int(settings.get("seed", 42))
    summary = {
        "cases": int(len(frame)),
        "positive": int(labels.sum()),
        "negative": int((labels == 0).sum()),
    }
    if mode == "holdout":
        validation_fraction = float(settings.get("validation_fraction", 0.2))
        if not 0.0 < validation_fraction < 1.0:
            raise ValueError("split.validation_fraction must be between 0 and 1")
        if np.unique(labels).size != 2:
            raise ValueError("Stratified holdout requires both binary classes")
        train_indices, validation_indices = train_test_split(
            np.arange(len(frame)),
            test_size=validation_fraction,
            random_state=seed,
            shuffle=True,
            stratify=labels,
        )
        splits = [(train_indices, validation_indices)]
        pattern = str(settings.get("filename_pattern", "{split}.txt"))
        summary.update({"mode": mode, "seed": seed, "validation_fraction": validation_fraction})
        for name, indices in (("train", train_indices), ("validation", validation_indices)):
            summary[name] = {
                "cases": int(len(indices)),
                "positive": int(labels[indices].sum()),
                "negative": int((labels[indices] == 0).sum()),
            }
    elif mode == "kfold":
        folds = int(settings.get("folds", 10))
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        splits = list(splitter.split(identifiers, labels))
        pattern = str(settings.get("filename_pattern", "fold_{fold}_{split}.txt"))
        summary["folds"] = folds
    else:
        raise ValueError("split.mode must be 'holdout' or 'kfold'")
    output = resolve_config_path(config, settings.get("output_directory", "dataset/splits"))
    output.mkdir(parents=True, exist_ok=True)
    for fold, (train_indices, validation_indices) in enumerate(splits, start=1):
        for split, indices in (("train", train_indices), ("validation", validation_indices)):
            path = output / pattern.format(fold=fold, split=split)
            path.write_text("\n".join(identifiers[indices]) + "\n", encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
