"""Shared metadata validation without image-processing or PyTorch dependencies."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

from petct.config import ConfigError

DEFAULT_POSITIVE_VALUES = {"1", "1.0", "yes", "true", "是"}
DEFAULT_NEGATIVE_VALUES = {"0", "0.0", "no", "false", "否"}


def normalize_identifier(value: Any) -> str:
    """Normalize spreadsheet identifiers without changing meaningful text."""

    if pd.isna(value):
        raise ConfigError("Encountered an empty sample identifier in metadata")
    text = str(value).strip()
    if not text:
        raise ConfigError("Encountered an empty sample identifier in metadata")
    return text[:-2] if re.fullmatch(r"\d+\.0", text) else text


def load_metadata(path: Path, id_column: str, label_column: str) -> pd.DataFrame:
    """Load private metadata from CSV or Excel and validate its schema."""

    if not path.is_file():
        raise ConfigError(f"Metadata file does not exist: {path}")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        frame = pd.read_csv(path)
    elif suffix == ".xlsx":
        frame = pd.read_excel(path)
    else:
        raise ConfigError(f"Unsupported metadata format {suffix}; use CSV or XLSX")
    missing = [column for column in (id_column, label_column) if column not in frame.columns]
    if missing:
        raise ConfigError(f"Metadata is missing required columns: {', '.join(missing)}")
    frame = frame[[id_column, label_column]].copy()
    frame[id_column] = frame[id_column].map(normalize_identifier)
    if frame[id_column].duplicated().any():
        duplicated = int(frame[id_column].duplicated().sum())
        raise ConfigError(f"Metadata contains {duplicated} duplicate sample identifiers")
    return frame.set_index(id_column)


def parse_binary_label(
    value: Any,
    positive_values: set[str] | None = None,
    negative_values: set[str] | None = None,
) -> int:
    """Map an explicit binary value to 0/1 and reject unknown labels."""

    normalized = str(value).strip().lower()
    positives = positive_values or DEFAULT_POSITIVE_VALUES
    negatives = negative_values or DEFAULT_NEGATIVE_VALUES
    if normalized in positives:
        return 1
    if normalized in negatives:
        return 0
    raise ConfigError(f"Unsupported binary label value: {value!r}")
