"""Single-run and cross-validation experiment orchestration."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from petct.artifacts import (
    configure_logging,
    prepare_run_directory,
    save_json,
    save_predictions,
    save_resolved_config,
)
from petct.config import ConfigError, external_selection_mode, require_sections
from petct.data import loader_from_config, prediction_identifiers
from petct.engine import evaluate_loader, fit, set_seed
from petct.metrics import summarize_folds
from petct.models import build_model
from petct.selection import ExternalEpochSelector


def _evaluate_external(
    model,
    criterion,
    device,
    config: dict[str, Any],
    run_directory: Path,
) -> dict[str, Any]:
    external_results: dict[str, Any] = {}
    external_config = config.get("data", {}).get("external", {})
    output = config.get("output", {})
    save_private = bool(output.get("save_predictions", False))
    for cohort_name, split_config in external_config.items():
        loader = loader_from_config(config, split_config, training=False)
        metrics, labels, probabilities = evaluate_loader(model, loader, criterion, device)
        external_results[cohort_name] = metrics
        save_predictions(
            run_directory / f"predictions_{cohort_name}.csv",
            labels,
            probabilities,
            save_private,
            identifiers=prediction_identifiers(
                loader, save_private and bool(output.get("save_prediction_ids", False))
            ),
            threshold=float(metrics.get("threshold", 0.5)),
        )
    return external_results


def run_training(
    config: dict[str, Any],
    *,
    run_directory: Path | None = None,
    evaluate_external: bool = True,
    epoch_callback=None,
) -> dict[str, Any]:
    """Run one train/validation experiment and optional final external evaluation."""

    require_sections(config, "data", "model", "training", "output")
    data = config["data"]
    if "train" not in data or "validation" not in data:
        raise ConfigError("data.train and data.validation are required")
    set_seed(int(config.get("seed", 513)), bool(config.get("deterministic", True)))
    run_directory = run_directory or prepare_run_directory(config)
    run_directory.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(run_directory)
    save_resolved_config(config, run_directory)
    train_loader = loader_from_config(config, data["train"], training=True)
    validation_loader = loader_from_config(config, data["validation"], training=False)
    model = build_model(config)
    selection = external_selection_mode(config)
    evaluate_external = evaluate_external and bool(
        config.get("evaluation", {}).get("external_after_training", True)
    )
    selector = None
    fit_options = {}
    if evaluate_external and selection == "per_cohort_best_auc":
        logger.warning(
            "External test labels are used for per-cohort epoch selection. "
            "These metrics are exploratory, not unbiased final test results; "
            "The best model and early stopping remain validation-only."
        )
        selector = ExternalEpochSelector()
        external_loaders = {
            name: loader_from_config(config, source, training=False)
            for name, source in data.get("external", {}).items()
        }
        output = config.get("output", {})
        export_ids = bool(output.get("save_predictions", False)) and bool(
            output.get("save_prediction_ids", False)
        )
        identifiers = {
            name: prediction_identifiers(loader, export_ids)
            for name, loader in external_loaders.items()
        }

        def observe_epoch(epoch, current_model, criterion, device):
            for name, loader in external_loaders.items():
                metrics, labels, probabilities = evaluate_loader(
                    current_model, loader, criterion, device
                )
                selector.update(name, epoch, metrics, labels, probabilities, identifiers[name])
                logger.info(
                    "epoch=%d external_cohort=%s auc=%s",
                    epoch,
                    name,
                    metrics["auc"],
                )

        fit_options["epoch_observer"] = observe_epoch
    fitted = fit(
        model,
        train_loader,
        validation_loader,
        config,
        run_directory,
        logger,
        epoch_callback,
        **fit_options,
    )
    results: dict[str, Any] = {
        "best_epoch": fitted["best_epoch"],
        "validation": fitted["validation"],
    }
    if selector is not None:
        results["external"] = selector.selected_metrics()
        for name, prediction in selector.best.items():
            save_predictions(
                run_directory / f"predictions_{name}.csv",
                prediction.labels,
                prediction.probabilities,
                bool(config["output"].get("save_predictions", False)),
                identifiers=prediction.identifiers,
                threshold=float(prediction.metrics.get("threshold", 0.5)),
            )
    elif evaluate_external:
        results["external"] = _evaluate_external(
            fitted["model"],
            fitted["criterion"],
            fitted["device"],
            config,
            run_directory,
        )
    save_json(run_directory / "metrics.json", results)
    return {"run_directory": run_directory, "results": results, **fitted}


def _format_fold_value(value: Any, fold: int) -> Any:
    if isinstance(value, str):
        return value.replace("{fold}", str(fold))
    if isinstance(value, dict):
        return {key: _format_fold_value(item, fold) for key, item in value.items()}
    if isinstance(value, list):
        return [_format_fold_value(item, fold) for item in value]
    return value


def run_cross_validation(config: dict[str, Any]) -> dict[str, Any]:
    """Run the registered model through identical sequential folds."""

    crossval = config.get("cross_validation", {})
    folds = int(crossval.get("folds", 5))
    root_directory = prepare_run_directory(config)
    save_resolved_config(config, root_directory)
    fold_results: list[dict[str, Any]] = []
    for fold in range(1, folds + 1):
        fold_config = _format_fold_value(copy.deepcopy(config), fold)
        fold_directory = root_directory / f"fold_{fold}"
        result = run_training(fold_config, run_directory=fold_directory)
        result_metrics = {
            "fold": fold,
            **{
                f"validation_{key}": value for key, value in result["results"]["validation"].items()
            },
        }
        for cohort, metrics in result["results"].get("external", {}).items():
            result_metrics.update({f"{cohort}_{key}": value for key, value in metrics.items()})
        fold_results.append(result_metrics)
    payload = {
        "folds": fold_results,
        "summary": summarize_folds(fold_results),
    }
    save_json(root_directory / "cross_validation.json", payload)
    return payload
