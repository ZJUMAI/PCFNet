import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")

from petct.artifacts import save_predictions
from petct.config import ConfigError
from petct.evaluation import _read
from petct.metrics import binary_metrics
from petct.selection import ExternalEpochSelector

LABELS = np.array([0, 0, 1, 1])
SCORES = {
    1.0: np.array([0.1, 0.2, 0.8, 0.9]),
    0.75: np.array([0.1, 0.7, 0.5, 0.9]),
    0.5: np.array([0.1, 0.9, 0.5, 0.7]),
    0.25: np.array([0.4, 0.9, 0.3, 0.8]),
}


def test_three_cohorts_select_independent_epochs_and_matching_predictions(tmp_path):
    selector = ExternalEpochSelector()
    curves = {
        "multi_hospital": [1.0, 0.75, 0.5],
        "france": [0.25, 1.0, 0.75],
        "SPH_test": [0.5, 0.75, 1.0],
    }
    identifiers = ["003", "001", "004", "002"]
    for epoch in range(1, 4):
        for cohort, scores in curves.items():
            probabilities = SCORES[scores[epoch - 1]]
            metrics = dict(binary_metrics(LABELS, probabilities), loss=epoch / 10)
            assert metrics["auc"] == scores[epoch - 1]
            selector.update(cohort, epoch, metrics, LABELS, probabilities, identifiers)
    selected = selector.selected_metrics()
    for epoch, cohort in enumerate(curves, start=1):
        prediction = selector.best[cohort]
        assert prediction.epoch == epoch
        assert selected[cohort]["auc"] == 1.0
        assert selected[cohort]["loss"] == epoch / 10
        assert set(selected[cohort]) == set(binary_metrics(LABELS, SCORES[1.0])) | {"loss"}
        path = tmp_path / f"{cohort}.csv"
        save_predictions(
            path,
            prediction.labels,
            prediction.probabilities,
            True,
            identifiers=prediction.identifiers,
        )
        labels, probabilities, ids = _read(path, "label", "probability")
        assert ids == identifiers
        assert binary_metrics(labels, probabilities)["auc"] == selected[cohort]["auc"]
        np.testing.assert_array_equal(probabilities, SCORES[1.0])


def test_selector_keeps_first_epoch_on_auc_ties():
    selector = ExternalEpochSelector()
    first = SCORES[1.0]
    second = np.array([0.2, 0.3, 0.6, 0.7])
    assert selector.update("site", 1, binary_metrics(LABELS, first), LABELS, first)
    assert not selector.update("site", 2, binary_metrics(LABELS, second), LABELS, second)
    assert selector.best["site"].epoch == 1
    np.testing.assert_array_equal(selector.best["site"].probabilities, first)


def test_selector_snapshots_arrays_ids_and_metrics():
    selector = ExternalEpochSelector()
    labels, probabilities = LABELS.copy(), SCORES[1.0].copy()
    metrics = binary_metrics(labels, probabilities)
    identifiers = ["003", "001", "004", "002"]
    selector.update("site", 1, metrics, labels, probabilities, identifiers)
    labels[:] = 1
    probabilities[:] = 0
    metrics["auc"] = 0
    identifiers[0] = "changed"
    selected_metrics = selector.selected_metrics()
    assert selected_metrics["site"]["auc"] == 1
    selected_metrics["site"]["auc"] = 0
    assert selector.best["site"].metrics["auc"] == 1
    assert selector.best["site"].identifiers[0] == "003"
    np.testing.assert_array_equal(selector.best["site"].labels, LABELS)
    np.testing.assert_array_equal(selector.best["site"].probabilities, SCORES[1.0])


@pytest.mark.parametrize("auc", [None, float("nan"), float("inf")])
def test_selector_rejects_undefined_auc(auc):
    with pytest.raises(ConfigError, match="AUC is undefined"):
        ExternalEpochSelector().update("site", 1, {"auc": auc}, LABELS, SCORES[1.0])


@pytest.mark.parametrize("change", ["labels", "identifiers"])
def test_selector_rejects_changed_case_order(change):
    selector = ExternalEpochSelector()
    identifiers = ["003", "001", "004", "002"]
    metrics = binary_metrics(LABELS, SCORES[1.0])
    selector.update("site", 1, metrics, LABELS, SCORES[1.0], identifiers)
    labels = LABELS[::-1] if change == "labels" else LABELS
    ids = identifiers[::-1] if change == "identifiers" else identifiers
    with pytest.raises(ConfigError, match="changed across epochs"):
        selector.update("site", 2, metrics, labels, SCORES[1.0], ids)


@pytest.mark.parametrize("enabled", [False, True])
def test_selector_exports_remain_opt_in(tmp_path, enabled):
    selector = ExternalEpochSelector()
    probabilities = SCORES[1.0]
    selector.update("site", 1, binary_metrics(LABELS, probabilities), LABELS, probabilities)
    prediction = selector.best["site"]
    path = tmp_path / "predictions.csv"
    save_predictions(path, prediction.labels, prediction.probabilities, enabled)
    assert path.is_file() == enabled
    if enabled:
        assert path.read_text(encoding="utf-8").splitlines()[0] == "label,probability"
