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


def validate_config(config: dict[str, Any], command: str) -> None:
    """Validate the command-level YAML contract before expensive work begins."""

    if command in {"train", "hpo", "crossval", "two-stage"}:
        require_sections(config, "data", "model", "training", "output")
        for split in ("train", "validation"):
            split_config = config["data"].get(split)
            if not isinstance(split_config, dict):
                raise ConfigError(f"data.{split} must be a mapping")
            missing = [
                key
                for key in ("manifest", "metadata", "ct_root", "pet_root")
                if not split_config.get(key)
            ]
            if missing:
                raise ConfigError(f"data.{split} is missing keys: {', '.join(missing)}")
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
        external = config["data"].get("external", {})
        if not isinstance(external, dict):
            raise ConfigError("data.external must be a named cohort mapping")
        for cohort_name, split_config in external.items():
            if not isinstance(split_config, dict):
                raise ConfigError(f"data.external.{cohort_name} must be a mapping")
            missing = [
                key
                for key in ("manifest", "metadata", "ct_root", "pet_root")
                if not split_config.get(key)
            ]
            if missing:
                raise ConfigError(
                    f"data.external.{cohort_name} is missing keys: {', '.join(missing)}"
                )
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
    elif command == "split":
        require_sections(config, "split")
        missing = [
            key for key in ("metadata", "id_column", "label_column") if not config["split"].get(key)
        ]
        if missing:
            raise ConfigError(f"split is missing keys: {', '.join(missing)}")
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
