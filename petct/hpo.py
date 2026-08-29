"""In-process Optuna hyperparameter optimization."""

from __future__ import annotations

import copy
import json
import shutil
from typing import Any

import yaml

from .artifacts import prepare_run_directory, save_resolved_config
from .config import public_config
from .workflows import run_training


def _set_nested(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    node = config
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        child = node.setdefault(part, {})
        if not isinstance(child, dict):
            raise ValueError(f"Cannot set {dotted_key!r}: {part!r} is not a mapping")
        node = child
    node[parts[-1]] = value


def _suggest(trial: Any, name: str, spec: dict[str, Any]) -> Any:
    kind = spec.get("type")
    if kind == "float":
        return trial.suggest_float(
            name,
            float(spec["low"]),
            float(spec["high"]),
            log=bool(spec.get("log", False)),
            step=spec.get("step"),
        )
    if kind == "int":
        return trial.suggest_int(
            name,
            int(spec["low"]),
            int(spec["high"]),
            step=int(spec.get("step", 1)),
            log=bool(spec.get("log", False)),
        )
    if kind == "categorical":
        choices = spec.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError(f"Categorical search parameter {name!r} needs non-empty choices")
        return trial.suggest_categorical(name, choices)
    raise ValueError(f"Unsupported search-space type {kind!r} for {name!r}")


def run_hpo(config: dict[str, Any]) -> dict[str, Any]:
    """Optimize validation AUC without constructing or reading an external loader."""
    try:
        import optuna
    except ImportError as error:
        raise RuntimeError("HPO requires the optional dependency: pip install .[hpo]") from error

    hpo_config = config.get("hpo", {})
    search_space = hpo_config.get("search_space", {})
    if not isinstance(search_space, dict) or not search_space:
        raise ValueError("hpo.search_space must be a non-empty mapping")

    run_directory = prepare_run_directory(config, suffix="hpo")
    save_resolved_config(config, run_directory)
    storage = hpo_config.get("storage")
    if storage is None:
        storage = f"sqlite:///{(run_directory / 'study.db').as_posix()}"

    sampler = optuna.samplers.TPESampler(seed=int(hpo_config.get("seed", 42)))
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=int(hpo_config.get("pruner_startup_trials", 5)),
        n_warmup_steps=int(hpo_config.get("pruner_warmup_steps", 1)),
    )
    study = optuna.create_study(
        study_name=str(hpo_config.get("study_name", "petct")),
        direction="maximize",
        storage=str(storage),
        load_if_exists=True,
        sampler=sampler,
        pruner=pruner,
    )

    keep_trial_outputs = bool(hpo_config.get("keep_trial_outputs", False))

    def objective(trial: Any) -> float:
        trial_config = copy.deepcopy(config)
        for key, specification in search_space.items():
            if not isinstance(specification, dict):
                raise ValueError(f"Search-space entry {key!r} must be a mapping")
            _set_nested(trial_config, key, _suggest(trial, key, specification))

        trial_directory = run_directory / "trials" / f"trial_{trial.number:04d}"

        def report(epoch: int, validation_metrics: dict[str, Any]) -> None:
            auc = validation_metrics.get("auc")
            if auc is None:
                return
            trial.report(float(auc), epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        try:
            result = run_training(
                trial_config,
                run_directory=trial_directory,
                evaluate_external=False,
                epoch_callback=report,
            )
            score = result["results"]["validation"]["auc"]
            if score is None:
                raise optuna.TrialPruned("Validation AUC is undefined")
            for key, value in result["results"]["validation"].items():
                if value is not None:
                    trial.set_user_attr(f"validation_{key}", value)
            return float(score)
        finally:
            if not keep_trial_outputs and trial_directory.exists():
                shutil.rmtree(trial_directory)

    study.optimize(objective, n_trials=int(hpo_config.get("trials", 20)))
    completed = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        raise RuntimeError("The study completed without a successful trial")

    best_config = copy.deepcopy(config)
    for key, value in study.best_params.items():
        _set_nested(best_config, key, value)
    best_config.pop("hpo", None)
    best_path = run_directory / "best_config.yaml"
    best_path.write_text(
        yaml.safe_dump(public_config(best_config), sort_keys=False), encoding="utf-8"
    )

    study.trials_dataframe().to_csv(run_directory / "trials.csv", index=False)
    summary = {
        "study_name": study.study_name,
        "best_value": float(study.best_value),
        "best_params": study.best_params,
        "completed_trials": sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials),
        "pruned_trials": sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials),
    }
    (run_directory / "study_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary
