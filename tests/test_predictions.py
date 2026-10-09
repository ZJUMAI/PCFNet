import csv
from pathlib import Path

import numpy as np
import pytest

from petct.artifacts import save_predictions
from petct.config import load_config
from petct.evaluation import run_evaluation

HEADERS = ["影像组学序列号", "是否预测成功", "预测概率", "预测结果", "ground truth"]


def _table(path: Path, identifiers=("003", "001", "002")) -> None:
    save_predictions(
        path,
        np.array([0, 1, 0]),
        np.array([0.2, 0.5, 0.8]),
        True,
        identifiers=identifiers,
    )


def test_identified_predictions_have_exact_schema_and_aligned_values(tmp_path: Path) -> None:
    path = tmp_path / "predictions.csv"
    _table(path)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == HEADERS
        rows = list(reader)
    assert [row[HEADERS[0]] for row in rows] == ["003", "001", "002"]
    assert [int(row["是否预测成功"]) for row in rows] == [1, 1, 0]
    assert [float(row["预测概率"]) for row in rows] == [0.2, 0.5, 0.8]
    assert [int(row["预测结果"]) for row in rows] == [0, 1, 1]
    assert [int(row["ground truth"]) for row in rows] == [0, 1, 0]


def test_probability_precision_is_not_rounded(tmp_path: Path) -> None:
    probability = float(np.float32(0.123456789))
    path = tmp_path / "predictions.csv"
    save_predictions(path, np.array([0]), np.array([probability]), True, identifiers=["001"])
    with path.open(encoding="utf-8-sig", newline="") as handle:
        row = next(csv.DictReader(handle))
    assert float(row["预测概率"]) == probability


def test_disabled_identified_export_writes_nothing(tmp_path: Path) -> None:
    path = tmp_path / "predictions.csv"
    save_predictions(path, np.array([0]), np.array([0.2]), False, identifiers=["private_case"])
    assert not path.exists()


@pytest.mark.parametrize("identifiers", [["one"], ["same", "same"], ["", "two"], [1, 2]])
def test_invalid_identifiers_fail_before_creating_file(tmp_path: Path, identifiers) -> None:
    path = tmp_path / "predictions.csv"
    with pytest.raises(ValueError, match="identifiers"):
        save_predictions(
            path, np.array([0, 1]), np.array([0.2, 0.8]), True, identifiers=identifiers
        )
    assert not path.exists()


@pytest.mark.parametrize(
    ("labels", "probabilities"),
    [([0], [0.1, 0.2]), ([2], [0.2]), ([0], [float("nan")]), ([0], [1.1]), ([0], [-0.1])],
)
def test_invalid_predictions_fail_before_creating_file(
    tmp_path: Path, labels, probabilities
) -> None:
    path = tmp_path / "predictions.csv"
    with pytest.raises(ValueError):
        save_predictions(path, np.array(labels), np.array(probabilities), True)
    assert not path.exists()


@pytest.mark.parametrize("filename", ["train.yaml", "two_stage.yaml", "evaluate.yaml"])
def test_final_test_configs_enable_identified_exports(filename: str) -> None:
    config = load_config(Path(__file__).resolve().parents[1] / "configs" / filename)
    settings = config["evaluate" if filename == "evaluate.yaml" else "output"]
    assert settings["save_predictions"] is True
    assert settings["save_prediction_ids"] is True


def test_metric_evaluation_reads_identified_table_without_column_overrides(tmp_path: Path) -> None:
    _table(tmp_path / "predictions.csv")
    config = {
        "_meta": {"config_dir": str(tmp_path)},
        "evaluate": {
            "task": "metrics",
            "predictions": "predictions.csv",
            "output": "metrics.json",
        },
    }
    result = run_evaluation(config)
    assert result["accuracy"] == pytest.approx(2 / 3)
    assert result["tp"] == result["tn"] == result["fp"] == 1
    assert result["fn"] == 0
    assert "003" not in (tmp_path / "metrics.json").read_text(encoding="utf-8")


def test_metric_evaluation_still_reads_anonymous_predictions(tmp_path: Path) -> None:
    save_predictions(
        tmp_path / "predictions.csv", np.array([0, 1]), np.array([0.2, 0.8]), True
    )
    result = run_evaluation(
        {
            "_meta": {"config_dir": str(tmp_path)},
            "evaluate": {
                "task": "metrics",
                "predictions": "predictions.csv",
                "output": "metrics.json",
            },
        }
    )
    assert result["accuracy"] == result["auc"] == 1.0


def test_paired_delong_checks_case_order_even_when_labels_match(tmp_path: Path) -> None:
    _table(tmp_path / "first.csv")
    _table(tmp_path / "second.csv", identifiers=("002", "001", "003"))
    config = {
        "_meta": {"config_dir": str(tmp_path)},
        "evaluate": {
            "task": "delong",
            "predictions_a": "first.csv",
            "predictions_b": "second.csv",
            "output": "delong.json",
        },
    }
    with pytest.raises(ValueError, match="identical IDs"):
        run_evaluation(config)
    assert not (tmp_path / "delong.json").exists()
    for filename in ("first.csv", "second.csv"):
        save_predictions(
            tmp_path / filename,
            np.array([0, 0, 0, 1, 1, 1]),
            np.array([0.1, 0.4, 0.3, 0.7, 0.8, 0.9]),
            True,
            identifiers=[f"case_{index}" for index in range(6)],
        )
    assert run_evaluation(config)["p_value"] == pytest.approx(1.0)
