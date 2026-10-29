"""Prepare OOF internal scores and equal-weight external fold means for fusion."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .dataset import (
    CLINICAL_COLUMNS,
    COHORT_COLUMN,
    DEFAULT_CLINICAL,
    PATIENT_COLUMN,
    SCORE_COLUMN,
    binary_labels,
    read_table,
)

ID_COLUMN = "影像组学序列号"
PROBABILITY_COLUMN = "预测概率"
LABEL_COLUMN = "ground truth"
EXTERNAL_COHORTS = {
    "multi_hospital": "External",
    "france": "Toulouse",
    "SPH_test": "SPH_perspective",
}
NOTICE = (
    "OOF scores are taken from validation CSV files, not in-sample training predictions. "
    "OOF construction and averaging do not remove upstream leakage, validation-based "
    "checkpoint selection, test-based epoch selection, or test-based seed selection bias. "
    "Original Training/Test clinical-table assignments are retained for fusion fitting/tuning."
)


def normalize_ids(values: pd.Series) -> pd.Series:
    """Preserve case strings while normalizing Excel's integral float representation."""
    if values.isna().any():
        raise ValueError("Missing case identifiers")
    normalized = values.astype(str).str.strip()
    normalized = normalized.str.replace(r"^([0-9]+)\.0+$", r"\1", regex=True)
    if normalized.eq("").any():
        raise ValueError("Empty case identifiers")
    return normalized


def read_predictions(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
    required = {ID_COLUMN, PROBABILITY_COLUMN, LABEL_COLUMN}
    if missing := required - set(frame.columns):
        raise ValueError(f"Missing prediction columns in {path.name}: {sorted(missing)}")
    frame = frame[[ID_COLUMN, LABEL_COLUMN, PROBABILITY_COLUMN]].copy()
    if frame.empty:
        raise ValueError(f"Empty prediction file: {path.name}")
    frame[ID_COLUMN] = normalize_ids(frame[ID_COLUMN])
    if frame[ID_COLUMN].duplicated().any():
        raise ValueError(f"Duplicate case identifiers in {path.name}")
    frame[LABEL_COLUMN] = binary_labels(frame[LABEL_COLUMN])
    probability = pd.to_numeric(frame[PROBABILITY_COLUMN], errors="raise")
    if not (np.isfinite(probability).all() and probability.between(0, 1).all()):
        raise ValueError(f"Probabilities must be finite and within [0, 1]: {path.name}")
    frame[PROBABILITY_COLUMN] = probability
    return frame


def fold_predictions(run: Path, cohort: str, folds: int) -> list[pd.DataFrame]:
    return [
        read_predictions(run / f"fold_{fold}" / f"predictions_{cohort}.csv")
        for fold in range(1, folds + 1)
    ]


def average_external(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Align by identifier, reject case/label drift, and average probabilities, not AUCs."""
    reference = frames[0].set_index(ID_COLUMN)
    aligned = []
    for frame in frames:
        current = frame.set_index(ID_COLUMN)
        if set(current.index) != set(reference.index):
            raise ValueError("External case sets differ between folds")
        current = current.loc[reference.index]
        if not current[LABEL_COLUMN].equals(reference[LABEL_COLUMN]):
            raise ValueError("External labels differ between folds")
        aligned.append(current[PROBABILITY_COLUMN].to_numpy())
    result = reference.copy()
    result[PROBABILITY_COLUMN] = np.mean(aligned, axis=0)
    return result.reset_index()


def collect_oof(frames: list[pd.DataFrame]) -> pd.DataFrame:
    result = pd.concat(
        [frame.assign(source_fold=fold) for fold, frame in enumerate(frames, 1)],
        ignore_index=True,
    )
    if result[ID_COLUMN].duplicated().any():
        raise ValueError("OOF requires exactly one validation prediction per case")
    return result


def prediction_export(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame[[ID_COLUMN, PROBABILITY_COLUMN, LABEL_COLUMN]].copy()
    result["预测结果"] = result[PROBABILITY_COLUMN].ge(0.5).astype(int)
    result["是否预测成功"] = result["预测结果"].eq(result[LABEL_COLUMN]).astype(int)
    return result[[ID_COLUMN, "是否预测成功", PROBABILITY_COLUMN, "预测结果", LABEL_COLUMN]]


def join_clinical(
    scores: pd.DataFrame, clinical: pd.DataFrame, label_source: str = "strict"
) -> tuple[pd.DataFrame, dict]:
    """Match within a cohort: internal SPH and France may share numeric case IDs."""
    required = [PATIENT_COLUMN, COHORT_COLUMN, "pCR", *CLINICAL_COLUMNS]
    if missing := set(required) - set(clinical.columns):
        raise ValueError(f"Missing clinical columns: {sorted(missing)}")
    selected = clinical.loc[
        clinical[COHORT_COLUMN].isin(["Training", "Test", *EXTERNAL_COHORTS.values()]),
        required,
    ].copy()
    selected[ID_COLUMN] = normalize_ids(selected[PATIENT_COLUMN])
    selected["_cohort"] = selected[COHORT_COLUMN].replace(
        {
            "Training": "internal",
            "Test": "internal",
            **{value: key for key, value in EXTERNAL_COHORTS.items()},
        }
    )
    keys = ["_cohort", ID_COLUMN]
    if selected.duplicated(keys).any():
        raise ValueError("Duplicate clinical identifiers within a cohort")
    merged = scores.merge(selected, on=keys, how="outer", validate="one_to_one", indicator=True)
    if not merged["_merge"].eq("both").all():
        missing_counts = merged["_merge"].value_counts().to_dict()
        raise ValueError(f"Clinical/prediction case coverage differs: {missing_counts}")
    clinical_labels = binary_labels(merged["pCR"])
    mismatched = clinical_labels.ne(merged[LABEL_COLUMN].astype(int))
    disagreements = merged.loc[mismatched, "_cohort"].value_counts().to_dict()
    if label_source not in {"strict", "predictions", "clinical"}:
        raise ValueError("label_source must be strict, predictions, or clinical")
    if mismatched.any() and label_source == "strict":
        raise ValueError("Prediction ground truth disagrees with the clinical table")
    merged["clinical_pCR_original"] = clinical_labels
    if label_source == "predictions":
        merged["pCR"] = merged[LABEL_COLUMN].astype(int)
    merged[SCORE_COLUMN] = merged[PROBABILITY_COLUMN]
    merged[PATIENT_COLUMN] = merged[ID_COLUMN]
    result = merged[
        [
            PATIENT_COLUMN,
            ID_COLUMN,
            COHORT_COLUMN,
            "pCR",
            *CLINICAL_COLUMNS,
            SCORE_COLUMN,
            "source_fold",
            "clinical_pCR_original",
        ]
    ]
    return result, disagreements


def prepare_run(run: Path, clinical: pd.DataFrame, folds: int = 5, label_source: str = "strict"):
    if folds < 2:
        raise ValueError("At least two folds are required")
    internal = collect_oof(fold_predictions(run, "validation", folds))
    outputs = {"oof_internal.csv": prediction_export(internal)}
    scores = [internal.assign(_cohort="internal")]
    counts = {"internal_oof": len(internal)}
    for cohort in EXTERNAL_COHORTS:
        averaged = average_external(fold_predictions(run, cohort, folds))
        outputs[f"predictions_{cohort}_mean.csv"] = prediction_export(averaged)
        scores.append(averaged.assign(_cohort=cohort, source_fold=0))
        counts[cohort] = len(averaged)
    fused, disagreements = join_clinical(
        pd.concat(scores, ignore_index=True), clinical, label_source
    )
    outputs["fusion_input.csv"] = fused
    source_hashes = {}
    for cohort in ["validation", *EXTERNAL_COHORTS]:
        for fold in range(1, folds + 1):
            relative = f"fold_{fold}/predictions_{cohort}.csv"
            source_hashes[relative] = hashlib.sha256((run / relative).read_bytes()).hexdigest()
    # Audit inconsistent legacy CSV/JSON validation exports, without replacing the CSV scores.
    inconsistencies = []
    for fold in range(1, folds + 1):
        metrics_path = run / f"fold_{fold}" / "metrics.json"
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))["validation"]
            current = internal.loc[internal["source_fold"].eq(fold)]
            positive = current.loc[current[LABEL_COLUMN].eq(1), PROBABILITY_COLUMN].to_numpy()
            negative = current.loc[current[LABEL_COLUMN].eq(0), PROBABILITY_COLUMN].to_numpy()
            if len(positive) and len(negative):
                difference = positive[:, None] - negative
                auc = float(np.mean((difference > 0) + 0.5 * (difference == 0)))
                if abs(auc - metrics["auc"]) > 1e-10:
                    inconsistencies.append(
                        {"fold": fold, "csv_auc": auc, "json_auc": metrics["auc"]}
                    )
    summary = {
        "source_run": str(run.resolve()),
        "folds": folds,
        "internal_score_source": "one validation CSV score per case; never train CSV",
        "external_score_source": "equal-weight arithmetic mean after case-ID alignment",
        "counts": counts,
        "fusion_split_counts": fused[COHORT_COLUMN].value_counts().to_dict(),
        "clinical_features": list(CLINICAL_COLUMNS.values()),
        "excluded_features": ["Cycles", "PDL1"],
        "label_source": label_source,
        "clinical_prediction_label_disagreements": disagreements,
        "validation_csv_json_inconsistencies": inconsistencies,
        "source_sha256": source_hashes,
        "notice": NOTICE,
    }
    return outputs, summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--clinical", type=Path, default=DEFAULT_CLINICAL)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--label-source",
        choices=["strict", "predictions", "clinical"],
        default="strict",
        help="Reject label disagreements by default; select an authority only after review",
    )
    args = parser.parse_args(argv)
    if args.output_root.exists():
        raise FileExistsError(
            "Choose a new output directory; existing results are never overwritten"
        )
    if len({run.name for run in args.runs}) != len(args.runs):
        raise ValueError("Source runs must have distinct directory names")
    clinical = read_table(args.clinical)
    prepared = [
        (run, *prepare_run(run, clinical, args.folds, args.label_source)) for run in args.runs
    ]
    for run, frames, summary in prepared:
        directory = args.output_root / run.name
        directory.mkdir(parents=True)
        summary["clinical_source"] = str(args.clinical.resolve())
        summary["clinical_sha256"] = hashlib.sha256(args.clinical.read_bytes()).hexdigest()
        for filename, frame in frames.items():
            frame.to_csv(directory / filename, index=False, encoding="utf-8-sig")
        (directory / "input_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"{run.name}: {summary['counts']}; exported to {directory}")
        if summary["validation_csv_json_inconsistencies"]:
            print("WARNING: validation CSV/JSON disagree; see input_summary.json. CSV scores used.")


if __name__ == "__main__":
    main()
