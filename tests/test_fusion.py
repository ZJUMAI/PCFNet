"""Offline clinical-fusion input preparation and fitting without private fixtures."""

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin

from fusion.dataset import (
    CATEGORICAL_FEATURES,
    CLINICAL_COLUMNS,
    COHORT_COLUMN,
    PATIENT_COLUMN,
    load_pcr_dataset,
)
from fusion.prepare_inputs import (
    EXTERNAL_COHORTS,
    ID_COLUMN,
    LABEL_COLUMN,
    PROBABILITY_COLUMN,
    average_external,
    collect_oof,
    main,
    prepare_run,
    read_predictions,
)
from fusion.run_fusion import fit_fusion, select_model


def predictions(ids, labels, probabilities):
    return pd.DataFrame(
        {
            ID_COLUMN: ids,
            LABEL_COLUMN: labels,
            PROBABILITY_COLUMN: probabilities,
        }
    )


@pytest.fixture
def synthetic_run(tmp_path):
    run = tmp_path / "source"
    clinical = []
    for fold in range(1, 6):
        directory = run / f"fold_{fold}"
        directory.mkdir(parents=True)
        ids = [str(2 * fold - 2), str(2 * fold - 1)]
        predictions(ids, [0, 1], [0.1 + fold * 0.01, 0.7 + fold * 0.01]).to_csv(
            directory / "predictions_validation.csv",
            index=False,
        )
        # Poison in-sample training scores: these must never be loaded.
        (directory / "predictions_train.csv").write_text("DO NOT READ", encoding="utf-8")
        for index, identifier in enumerate(ids):
            clinical.append(
                {
                    PATIENT_COLUMN: identifier,
                    COHORT_COLUMN: "Training" if fold < 4 else "Test",
                    "pCR": "是" if index else "否",
                }
            )
        for cohort in EXTERNAL_COHORTS:
            frame = predictions(["0", "1"], [0, 1], [0.1 * fold, 0.5 + 0.05 * fold])
            if fold % 2 == 0:
                frame = frame.iloc[::-1]
            frame.to_csv(directory / f"predictions_{cohort}.csv", index=False)
    for cohort in EXTERNAL_COHORTS.values():
        for index in range(2):
            clinical.append(
                {PATIENT_COLUMN: str(index), COHORT_COLUMN: cohort, "pCR": "是" if index else "否"}
            )
    clinical = pd.DataFrame(clinical)
    for column in CLINICAL_COLUMNS:
        clinical[column] = np.arange(len(clinical)) % 3 + 1
    clinical["病例完成治疗的周期数"] = 999
    clinical["PDL1_expression"] = "must not be used"
    return run, clinical


def test_average_aligns_ids_and_averages_probabilities():
    first = predictions(["a", "b"], [0, 1], [0.1, 0.6])
    second = predictions(["b", "a"], [1, 0], [0.8, 0.3])
    result = average_external([first, second])
    assert result[ID_COLUMN].tolist() == ["a", "b"]
    np.testing.assert_allclose(result[PROBABILITY_COLUMN], [0.2, 0.7])


@pytest.mark.parametrize("change", ["missing", "label"])
def test_average_rejects_case_or_label_drift(change):
    first = predictions(["a", "b"], [0, 1], [0.1, 0.6])
    second = first.iloc[:1] if change == "missing" else first.assign(**{LABEL_COLUMN: [1, 0]})
    with pytest.raises(ValueError):
        average_external([first, second])


@pytest.mark.parametrize("bad_probability", [-0.1, 1.1, np.inf, np.nan])
def test_prediction_reader_rejects_invalid_probability(tmp_path, bad_probability):
    path = tmp_path / "input.csv"
    predictions(["a"], [0], [bad_probability]).to_csv(path, index=False)
    with pytest.raises(ValueError):
        read_predictions(path)


def test_oof_rejects_duplicate_validation_cases():
    frame = predictions(["a"], [0], [0.2])
    with pytest.raises(ValueError, match="exactly one"):
        collect_oof([frame, frame])


def test_prepare_excludes_features_and_only_uses_oof(synthetic_run, tmp_path):
    run, clinical = synthetic_run
    outputs, summary = prepare_run(run, clinical)
    combined = outputs["fusion_input.csv"]
    assert len(combined) == 16
    assert not {"病例完成治疗的周期数", "PDL1_expression"} & set(combined.columns)
    assert summary["counts"]["internal_oof"] == 10
    assert summary["clinical_prediction_label_disagreements"] == {}
    # Identical numeric IDs in different hospitals must not cross-match.
    assert combined[COHORT_COLUMN].value_counts().to_dict() == {
        "Training": 6,
        "Test": 4,
        "External": 2,
        "Toulouse": 2,
        "SPH_perspective": 2,
    }
    external = outputs["predictions_multi_hospital_mean.csv"]
    np.testing.assert_allclose(external[PROBABILITY_COLUMN], [0.3, 0.65])
    path = tmp_path / "prepared.csv"
    combined.to_csv(path, index=False)
    x, y, cohort, _ = load_pcr_dataset(path, include_patient_id=True)
    assert x.shape == (16, 10)
    assert not {"Cycles", "PDL1", "source_fold", "clinical_pCR_original"} & set(x.columns)
    assert all(x[column].map(type).eq(str).all() for column in CATEGORICAL_FEATURES)
    assert y.nunique() == 2
    assert len(cohort) == 16


def test_label_conflicts_require_explicit_authority(synthetic_run):
    run, clinical = synthetic_run
    target = clinical[COHORT_COLUMN].eq("SPH_perspective") & clinical[PATIENT_COLUMN].eq("0")
    clinical.loc[target, "pCR"] = "是"
    with pytest.raises(ValueError, match="ground truth"):
        prepare_run(run, clinical)
    outputs, summary = prepare_run(run, clinical, label_source="predictions")
    fusion = outputs["fusion_input.csv"]
    selected = fusion[COHORT_COLUMN].eq("SPH_perspective") & fusion[PATIENT_COLUMN].eq("0")
    assert fusion.loc[selected, "pCR"].item() == 0
    assert fusion.loc[selected, "clinical_pCR_original"].item() == 1
    assert summary["clinical_prediction_label_disagreements"] == {"SPH_test": 1}
    outputs, _ = prepare_run(run, clinical, label_source="clinical")
    assert outputs["fusion_input.csv"].loc[selected, "pCR"].item() == "是"


def test_prepare_rejects_duplicate_missing_clinical_or_bad_labels(synthetic_run):
    run, clinical = synthetic_run
    with pytest.raises(ValueError, match="Duplicate clinical"):
        prepare_run(run, pd.concat([clinical, clinical.iloc[:1]]))
    with pytest.raises(ValueError, match="coverage"):
        prepare_run(run, clinical.iloc[1:])
    clinical.loc[0, "pCR"] = "unknown"
    with pytest.raises(ValueError, match="Labels"):
        prepare_run(run, clinical)


def test_prepare_fails_on_missing_fold(synthetic_run):
    run, clinical = synthetic_run
    with pytest.raises(FileNotFoundError):
        prepare_run(run, clinical, folds=6)


def test_cli_preserves_sources_and_refuses_overwrite(synthetic_run, tmp_path):
    run, clinical = synthetic_run
    metadata = tmp_path / "clinical.csv"
    clinical.to_csv(metadata, index=False)
    sources = {path: path.read_bytes() for path in run.rglob("*.csv")}
    destination = tmp_path / "output"
    args = ["--runs", str(run), "--clinical", str(metadata), "--output-root", str(destination)]
    main(args)
    assert all(path.read_bytes() == data for path, data in sources.items())
    assert len(list((destination / run.name).glob("*.csv"))) == 5
    summary = json.loads((destination / run.name / "input_summary.json").read_text())
    assert len(summary["source_sha256"]) == 20
    with pytest.raises(FileExistsError):
        main(args)


class DirectionClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, direction=1):
        self.direction = direction

    def fit(self, x, y):
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, x):
        probability = np.asarray(x).ravel() if self.direction == 1 else 1 - np.asarray(x).ravel()
        return np.column_stack([1 - probability, probability])


def test_search_keeps_independent_candidate_models():
    x, y = np.array([[0.1], [0.9]]), np.array([0, 1])
    original = DirectionClassifier()
    selected, trials = select_model(
        original,
        [{"direction": 1}, {"direction": -1}],
        x,
        y,
        x,
        y,
        top_k=1,
    )
    assert selected["model"].direction == 1
    assert selected["validation_auc"] == 1
    assert trials[1]["validation_auc"] == 0
    assert not hasattr(original, "classes_")


def test_logistic_fusion_smoke(synthetic_run, tmp_path):
    run, clinical = synthetic_run
    frames, _ = prepare_run(run, clinical)
    source = tmp_path / "fusion_input.csv"
    frames["fusion_input.csv"].to_csv(source, index=False)
    destination = tmp_path / "models"
    result = fit_fusion(source, destination, ["logistic"], iterations=2, top_k=1)
    assert len(result["features"]) == 10
    assert set(result["models"]["logistic"]["metrics"]) == {
        "train",
        "validation",
        "multi_hospital",
        "france",
        "SPH_test",
    }
    assert len(list((destination / "logistic").glob("predictions_*.csv"))) == 5
    with pytest.raises(FileExistsError):
        fit_fusion(source, destination, ["logistic"])
