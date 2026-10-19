"""YAML configuration loading and validation."""

from __future__ import annotations

import copy
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when an experiment configuration is invalid."""


def _set_nested(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    keys = dotted_key.split(".")
    current = config
    for key in keys[:-1]:
        child = current.get(key)
        if child is None:
            child = {}
            current[key] = child
        if not isinstance(child, dict):
            raise ConfigError(f"Cannot override {dotted_key}: {key} is not a mapping")
        current = child
    current[keys[-1]] = value


def apply_overrides(config: dict[str, Any], overrides: Iterable[str]) -> dict[str, Any]:
    """Apply dotted KEY=VALUE overrides using YAML value parsing."""

    resolved = copy.deepcopy(config)
    for override in overrides:
        if "=" not in override:
            raise ConfigError(f"Invalid override {override!r}; expected KEY=VALUE")
        key, raw_value = override.split("=", 1)
        if not key:
            raise ConfigError(f"Invalid override {override!r}; key is empty")
        _set_nested(resolved, key, yaml.safe_load(raw_value))
    return resolved


def load_config(path: str | Path, overrides: Iterable[str] = ()) -> dict[str, Any]:
    """Load a YAML file, apply overrides, and attach source metadata."""

    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise ConfigError(f"Configuration file does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ConfigError("The YAML document root must be a mapping")
    config = apply_overrides(config, overrides)
    config["_meta"] = {
        "config_path": str(config_path),
        "config_dir": str(config_path.parent),
    }
    return config


def require_sections(config: dict[str, Any], *sections: str) -> None:
    missing = [name for name in sections if not isinstance(config.get(name), dict)]
    if missing:
        raise ConfigError(f"Missing required configuration sections: {', '.join(missing)}")


def validate_split_config(
    split_config: dict[str, Any], location: str, *, allow_groups: bool = False
) -> None:
    """Validate a paired data source or a named group of paired sources."""

    if not isinstance(split_config, dict):
        raise ConfigError(f"{location} must be a mapping")
    if "members" in split_config:
        if not allow_groups:
            raise ConfigError(f"{location}: grouped cohorts are only supported for evaluation")
        members = split_config["members"]
        if not isinstance(members, dict) or not members:
            raise ConfigError(f"{location}.members must be a non-empty named mapping")
        if any(key in split_config for key in ("manifest", "metadata", "ct_root", "pet_root")):
            raise ConfigError(f"{location} cannot mix group members with single-source paths")
        for name, member in members.items():
            validate_split_config(member, f"{location}.members.{name}")
        return
    missing = [
        key for key in ("manifest", "metadata", "ct_root", "pet_root") if not split_config.get(key)
    ]
    if missing:
        raise ConfigError(f"{location} is missing keys: {', '.join(missing)}")


def _validate_external_config(config: dict[str, Any]) -> None:
    external = config.get("data", {}).get("external", {})
    if not isinstance(external, dict):
        raise ConfigError("data.external must be a named cohort mapping")
    for name, split_config in external.items():
        validate_split_config(split_config, f"data.external.{name}", allow_groups=True)


def checkpoint_saving_enabled(config: dict[str, Any]) -> bool:
    """Validate the explicit checkpoint export switch; default to memory-only."""

    output = config.get("output", {})
    if not isinstance(output, dict):
        raise ConfigError("output must be a mapping")
    enabled = output.get("save_checkpoint", False)
    if not isinstance(enabled, bool):
        raise ConfigError("output.save_checkpoint must be a YAML boolean")
    return enabled


def external_selection_mode(config: dict[str, Any]) -> str:
    """Return the external evaluation policy, preserving validation-only defaults."""

    evaluation = config.get("evaluation", {})
    if not isinstance(evaluation, dict):
        raise ConfigError("evaluation must be a mapping")
    mode = evaluation.get("external_selection", "validation_best")
    if mode not in ("validation_best", "per_cohort_best_auc"):
        raise ConfigError(
            "evaluation.external_selection must be validation_best or per_cohort_best_auc"
        )
    return mode


def leakage_simulation_settings(config: dict[str, Any]) -> dict[str, Any] | None:
    """Validate explicit test-contamination settings without accessing any data."""

    training = config.get("training", {})
    if not isinstance(training, dict):
        raise ConfigError("training must be a mapping")
    settings = training.get("leakage_simulation", {})
    location = "training.leakage_simulation"
    if not isinstance(settings, dict):
        raise ConfigError(f"{location} must be a mapping")
    unknown = set(settings) - {"enabled", "samples_per_cohort", "cohorts", "cohort_samples"}
    if unknown:
        raise ConfigError(f"Unknown {location} keys: {', '.join(sorted(map(str, unknown)))}")
    enabled = settings.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ConfigError(f"{location}.enabled must be a YAML boolean")
    count = settings.get("samples_per_cohort", 3)
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ConfigError(f"{location}.samples_per_cohort must be a positive integer")
    overrides = settings.get("cohort_samples", {})
    if not isinstance(overrides, dict):
        raise ConfigError(f"{location}.cohort_samples must be a named count mapping")
    for name, value in overrides.items():
        if not isinstance(name, str) or not name.strip():
            raise ConfigError(f"{location}.cohort_samples keys must be non-empty cohort names")
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ConfigError(f"{location}.cohort_samples.{name} must be a non-negative integer")
    cohorts = settings.get("cohorts")
    if cohorts is not None and (
        not isinstance(cohorts, list)
        or not cohorts
        or any(not isinstance(name, str) or not name.strip() for name in cohorts)
        or len(cohorts) != len(set(cohorts))
    ):
        raise ConfigError(f"{location}.cohorts must be a non-empty list of unique cohort names")
    if not enabled:
        return None
    external = config.get("data", {}).get("external", {})
    if not isinstance(external, dict) or not external:
        raise ConfigError("Leakage simulation requires non-empty data.external cohorts")
    names = list(external) if cohorts is None else list(cohorts)
    if any(name not in external for name in names):
        raise ConfigError(f"{location}.cohorts must name groups in data.external")
    if any(name not in external for name in overrides):
        raise ConfigError(f"{location}.cohort_samples must name groups in data.external")
    if any(name not in names for name in overrides):
        raise ConfigError(f"{location}.cohort_samples cannot override unselected cohorts")
    counts = {name: overrides.get(name, count) for name in names}
    if not any(counts.values()):
        raise ConfigError(f"{location} requires at least one selected cohort with a positive count")
    evaluation = config.get("evaluation", {})
    if not isinstance(evaluation, dict):
        raise ConfigError("evaluation must be a mapping")
    if not evaluation.get("external_after_training", True):
        raise ConfigError("Leakage simulation requires evaluation.external_after_training=true")
    return {"samples_per_cohort": count, "cohorts": names, "cohort_samples": counts}


def repeated_holdout_parameters(settings: dict[str, Any]) -> tuple[int, float, str]:
    """Validate repeated holdout count, validation fraction, and filename template."""

    repeats = settings.get("repeats", 5)
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ConfigError("split.repeats must be a positive integer")
    try:
        fraction = float(settings.get("validation_fraction", 0.3))
    except (TypeError, ValueError) as error:
        raise ConfigError("split.validation_fraction must be between 0 and 1") from error
    if not 0 < fraction < 1:
        raise ConfigError("split.validation_fraction must be between 0 and 1")
    pattern = settings.get("filename_pattern", "repeat_{fold}_{split}.txt")
    if not isinstance(pattern, str) or "{fold}" not in pattern or "{split}" not in pattern:
        raise ConfigError("Repeated holdout filename_pattern must contain {fold} and {split}")
    if "/" in pattern or "\\" in pattern:
        raise ConfigError("Repeated holdout filename_pattern must be a filename, not a path")
    try:
        pattern.format(fold=1, split="train")
    except (KeyError, ValueError, IndexError) as error:
        raise ConfigError(
            "Repeated holdout filename_pattern contains unsupported fields"
        ) from error
    return repeats, fraction, pattern


def validate_config(config: dict[str, Any], command: str) -> None:
    """Validate the command-level YAML contract before expensive work begins."""

    if command in {"train", "hpo", "crossval", "two-stage"}:
        require_sections(config, "data", "model", "training", "output")
        checkpoint_saving_enabled(config)
        selection = external_selection_mode(config)
        leakage = leakage_simulation_settings(config)
        if leakage is not None and command not in {"train", "crossval"}:
            raise ConfigError("Leakage simulation is supported only by train and crossval")
        if selection == "per_cohort_best_auc" and command in {"hpo", "two-stage"}:
            raise ConfigError(
                "per_cohort_best_auc is supported only by train and crossval, not HPO or two-stage"
            )
        for split in ("train", "validation"):
            validate_split_config(config["data"].get(split), f"data.{split}")
        if command == "hpo":
            require_sections(config, "hpo")
            if not config["hpo"].get("search_space"):
                raise ConfigError("hpo.search_space must not be empty")
        if command == "crossval" and "{fold}" not in str(config["data"]["train"]["manifest"]):
            raise ConfigError("Cross-validation train manifest must contain {fold}")
        if command == "crossval" and "{fold}" not in str(config["data"]["validation"]["manifest"]):
            raise ConfigError("Cross-validation validation manifest must contain {fold}")
        if command == "two-stage" and config["model"].get("name") != "feature_extractor":
            raise ConfigError("Two-stage experiments require model.name=feature_extractor")
        _validate_external_config(config)
    elif command == "preprocess":
        require_sections(config, "preprocess")
        cohorts = config["preprocess"].get("cohorts")
        if not isinstance(cohorts, dict) or not cohorts:
            raise ConfigError("preprocess.cohorts must not be empty")
        for cohort_name, cohort in cohorts.items():
            if not isinstance(cohort, dict):
                raise ConfigError(f"preprocess.cohorts.{cohort_name} must be a mapping")
            missing = [
                key
                for key in (
                    "raw_root",
                    "resampled_root",
                    "windowed_ct_root",
                    "sliced_root",
                    "clahe_root",
                )
                if not cohort.get(key)
            ]
            if missing:
                raise ConfigError(
                    f"preprocess.cohorts.{cohort_name} is missing keys: {', '.join(missing)}"
                )
    elif command == "evaluate":
        require_sections(config, "evaluate")
        if not config["evaluate"].get("task"):
            raise ConfigError("evaluate.task is required")
        if config["evaluate"]["task"] == "checkpoint":
            _validate_external_config(config)
    elif command == "split":
        require_sections(config, "split")
        missing = [
            key for key in ("metadata", "id_column", "label_column") if not config["split"].get(key)
        ]
        if missing:
            raise ConfigError(f"split is missing keys: {', '.join(missing)}")
        if config["split"].get("mode") == "repeated_holdout":
            repeated_holdout_parameters(config["split"])
    else:
        raise ConfigError(f"Unknown command for validation: {command}")


def resolve_config_path(config: dict[str, Any], value: str | Path) -> Path:
    """Resolve a path relative to the YAML file location."""

    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    config_dir = Path(config.get("_meta", {}).get("config_dir", "."))
    return (config_dir / path).resolve()


def public_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return a serializable configuration without internal metadata."""

    return {key: value for key, value in config.items() if key != "_meta"}
