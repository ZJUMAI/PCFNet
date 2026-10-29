"""Clinical variables and a precomputed image score for late fusion."""

from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_CLINICAL = Path(__file__).resolve().parent / "PET_all_radiomics-2026-1-22.xlsx"
CLINICAL_COLUMNS = {
    "性别": "Gender",
    "就诊年龄": "Age",
    "是否吸烟": "Smoking",
    "Drug": "Drug",
    "病理诊断": "Pathology",
    "最长径（淋巴结为最短经）": "Tumor_size",
    "基线期T分期": "T_stage",
    "基线期N分期": "N_stage",
    "基线期分期": "Stage",
}
SCORE_COLUMN = "Predicted_score_3Dresnet"
NUMERIC_FEATURES = ["Age", "Tumor_size", "Predicted_score"]
CATEGORICAL_FEATURES = [
    "Gender",
    "Smoking",
    "Drug",
    "Pathology",
    "T_stage",
    "N_stage",
    "Stage",
]
COHORT_COLUMN = "Cohort_final"
PATIENT_COLUMN = "检查编号"


def read_table(path: str | Path) -> pd.DataFrame:
    """Read private clinical inputs without modifying their source."""
    path = Path(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, dtype={PATIENT_COLUMN: str}, encoding="utf-8-sig")
    return pd.read_excel(path, dtype={PATIENT_COLUMN: str})


def binary_labels(values: pd.Series) -> pd.Series:
    """Reject missing or unsupported labels instead of assuming negative cases."""
    result = (
        values.astype(str)
        .str.strip()
        .map(
            {
                "是": 1,
                "否": 0,
                "1": 1,
                "0": 0,
                "1.0": 1,
                "0.0": 0,
            }
        )
    )
    if result.isna().any():
        raise ValueError("Labels must contain only 是/否 or 0/1, with no missing values")
    return result.astype(int)


def load_pcr_dataset(xlsx_path: str | Path = DEFAULT_CLINICAL, include_patient_id: bool = False):
    """Load nine clinical variables plus the image score; never Cycles or PDL1."""
    frame = read_table(xlsx_path)
    required = [*CLINICAL_COLUMNS, SCORE_COLUMN, "pCR", COHORT_COLUMN]
    if include_patient_id:
        required.append(PATIENT_COLUMN)
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing fusion columns: {sorted(missing)}")
    selected = frame[required].rename(
        columns={
            **CLINICAL_COLUMNS,
            SCORE_COLUMN: "Predicted_score",
        }
    )
    labels = binary_labels(selected["pCR"])
    features = selected[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    for column in NUMERIC_FEATURES:
        features[column] = pd.to_numeric(features[column], errors="coerce")
    features = features.replace([np.inf, -np.inf], np.nan)
    for column in CATEGORICAL_FEATURES:
        features[column] = features[column].astype("string").fillna("Missing").astype(str)
    cohort = selected[COHORT_COLUMN]
    if include_patient_id:
        return features, labels, cohort, selected[PATIENT_COLUMN].to_numpy()
    return features, labels, cohort
