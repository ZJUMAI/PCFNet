import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")
pytest.importorskip("sklearn")

from petct.metrics import binary_metrics, paired_delong_test


def test_single_class_metrics_do_not_crash() -> None:
    result = binary_metrics(np.zeros(4), np.array([0.1, 0.2, 0.3, 0.4]))
    assert result["auc"] is None
    assert result["tn"] == 4


def test_identical_predictions_have_delong_p_one() -> None:
    labels = np.array([0, 0, 0, 1, 1, 1])
    probabilities = np.array([0.1, 0.4, 0.3, 0.7, 0.8, 0.9])
    result = paired_delong_test(labels, probabilities, probabilities)
    assert result["difference"] == pytest.approx(0.0)
    assert result["p_value"] == pytest.approx(1.0)


def test_delong_is_symmetric_when_models_are_swapped() -> None:
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    first = np.array([0.1, 0.2, 0.8, 0.4, 0.6, 0.7, 0.9, 0.5])
    second = np.array([0.3, 0.1, 0.5, 0.2, 0.9, 0.4, 0.8, 0.7])
    forward = paired_delong_test(labels, first, second)
    reverse = paired_delong_test(labels, second, first)
    assert forward["difference"] == pytest.approx(-reverse["difference"])
    assert forward["p_value"] == pytest.approx(reverse["p_value"])
