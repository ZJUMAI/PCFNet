"""CNN feature extraction followed by a gradient-boosted tree classifier."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .artifacts import (
    configure_logging,
    prepare_run_directory,
    save_json,
    save_predictions,
    save_resolved_config,
)
from .data import loader_from_config
from .engine import fit, resolve_device, set_seed
from .metrics import binary_metrics
from .models import build_model


@torch.inference_mode()
def _extract(
    model: torch.nn.Module, loader: Any, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    features: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for batch in loader:
        ct = batch["ct"].to(device, non_blocking=True)
        pet = batch["pet"].to(device, non_blocking=True)
        feature = model.extract_features(ct, pet)
        features.append(feature.cpu().numpy())
        labels.append(batch["label"].numpy())
    return np.concatenate(features), np.concatenate(labels).astype(int)


def _tree_classifier(config: dict[str, Any]) -> Any:
    tree = config.get("tree", {})
    name = str(tree.get("name", "lightgbm")).lower()
    parameters = dict(tree.get("parameters", {}))
    parameters.setdefault("random_state", int(config.get("seed", 42)))
    if name == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
        except ImportError as error:
            raise RuntimeError(
                "Install two-stage dependencies with: pip install .[two-stage]"
            ) from error
        return LGBMClassifier(**parameters)
    if name == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as error:
            raise RuntimeError(
                "Install two-stage dependencies with: pip install .[two-stage]"
            ) from error
        parameters.setdefault("eval_metric", "logloss")
        return XGBClassifier(**parameters)
    raise ValueError("tree.name must be 'lightgbm' or 'xgboost'")


def run_two_stage(config: dict[str, Any]) -> dict[str, Any]:
    seed = int(config.get("seed", 42))
    set_seed(seed, bool(config.get("deterministic", True)))
    device = resolve_device(config.get("device", "auto"))
    run_directory = prepare_run_directory(config, suffix="two_stage")
    save_resolved_config(config, run_directory)
    logger = configure_logging(run_directory)

    data = config["data"]
    train_loader = loader_from_config(config, data["train"], training=True)
    train_feature_loader = loader_from_config(config, data["train"], training=False)
    validation_loader = loader_from_config(config, data["validation"], training=False)

    model_config = dict(config)
    model_config["model"] = dict(config.get("model", {}))
    model_config["model"]["name"] = "feature_extractor"
    model = build_model(model_config)
    training_result = fit(
        model,
        train_loader,
        validation_loader,
        model_config,
        run_directory,
        logger,
    )
    model = training_result["model"]

    train_features, train_labels = _extract(model, train_feature_loader, device)
    validation_features, validation_labels = _extract(model, validation_loader, device)
    variable_mask = np.ptp(train_features, axis=0) > 0
    if not np.any(variable_mask):
        raise RuntimeError("All extracted CNN features are constant")
    train_features = train_features[:, variable_mask]
    validation_features = validation_features[:, variable_mask]

    classifier = _tree_classifier(config)
    classifier.fit(train_features, train_labels)
    validation_probability = classifier.predict_proba(validation_features)[:, 1]
    validation_metrics = binary_metrics(validation_labels, validation_probability)
    result: dict[str, Any] = {"validation": validation_metrics}

    output = config.get("output", {})
    save_predictions(
        run_directory / "predictions_validation.csv",
        validation_labels,
        validation_probability,
        bool(output.get("save_predictions", False)),
    )

    # The external cohort is constructed and accessed only after both stages are fixed.
    external_results: dict[str, Any] = {}
    for cohort_name, split_config in data.get("external", {}).items():
        external_loader = loader_from_config(config, split_config, training=False)
        external_features, external_labels = _extract(model, external_loader, device)
        external_probability = classifier.predict_proba(external_features[:, variable_mask])[:, 1]
        external_results[cohort_name] = binary_metrics(external_labels, external_probability)
        save_predictions(
            run_directory / f"predictions_{cohort_name}.csv",
            external_labels,
            external_probability,
            bool(output.get("save_predictions", False)),
        )
    if external_results:
        result["external"] = external_results

    try:
        import joblib
    except ImportError as error:
        raise RuntimeError("Saving the tree model requires joblib") from error
    joblib.dump(
        {"model": classifier, "feature_mask": variable_mask}, run_directory / "tree_model.joblib"
    )
    save_json(run_directory / "metrics.json", result)
    return result
