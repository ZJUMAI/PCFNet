"""Original training scores use the restored best model without augmentation."""

import json
from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("torch")
pytest.importorskip("torchio")

import petct.workflows as workflows


@pytest.mark.parametrize("identified", [False, True])
@pytest.mark.parametrize("enabled", [None, False, True])
def test_original_training_export_uses_best_model_and_separate_evaluation_loader(
    tmp_path, monkeypatch, identified, enabled
):
    original_split = {"name": "train", "manifest": "original_train.txt"}
    config = {
        "data": {"train": original_split, "validation": {"name": "validation"}},
        "model": {},
        "training": {},
        "evaluation": {"external_after_training": False},
        "output": {
            "save_validation_predictions": False,
            "save_predictions": False,
            "save_prediction_ids": identified,
        },
    }
    if enabled is not None:
        config["output"]["save_training_predictions"] = enabled
    calls = []
    training_loader = object()
    validation_loader = object()
    export_loader = object()
    model = SimpleNamespace(epoch=3)
    labels = np.array([1, 0, 1])
    probabilities = np.array([0.8, 0.1, 0.4])

    def loader(root, split, *, training):
        calls.append((split["name"], training))
        if split["name"] == "validation":
            assert not training
            return validation_loader
        assert split is original_split
        return training_loader if training else export_loader

    def fit(current_model, train, validation, *args, **kwargs):
        assert train is training_loader and validation is validation_loader
        current_model.epoch = 2
        return dict(
            model=current_model,
            criterion=None,
            device="cpu",
            best_epoch=2,
            validation={"auc": 0.8, "threshold": 0.5},
        )

    def evaluate(current_model, loader, *args):
        assert current_model is model and current_model.epoch == 2
        assert loader is export_loader and loader is not training_loader
        return {}, labels, probabilities

    def identifiers(loader, enabled):
        assert loader is export_loader
        return ["003", "001", "002"] if enabled else None

    monkeypatch.setattr(workflows, "set_seed", lambda *args: None)
    monkeypatch.setattr(workflows, "loader_from_config", loader)
    monkeypatch.setattr(workflows, "build_model", lambda config: model)
    monkeypatch.setattr(workflows, "fit", fit)
    monkeypatch.setattr(workflows, "evaluate_loader", evaluate)
    monkeypatch.setattr(workflows, "prediction_identifiers", identifiers)
    result = workflows.run_training(config, run_directory=tmp_path)
    assert result["results"] == {"best_epoch": 2, "validation": {"auc": 0.8, "threshold": 0.5}}
    assert json.loads((tmp_path / "metrics.json").read_text()) == result["results"]
    path = tmp_path / "predictions_train.csv"
    if enabled is False:
        assert calls == [("train", True), ("validation", False)]
        assert not path.exists()
    else:
        assert calls == [("train", True), ("validation", False), ("train", False)]
        frame = pd.read_csv(path, dtype={"影像组学序列号": str}, float_precision="round_trip")
        if identified:
            assert frame.columns.tolist() == [
                "影像组学序列号",
                "是否预测成功",
                "预测概率",
                "预测结果",
                "ground truth",
            ]
            assert frame["影像组学序列号"].tolist() == ["003", "001", "002"]
            assert frame["预测结果"].tolist() == [1, 0, 0]
            assert frame["是否预测成功"].tolist() == [1, 1, 0]
            np.testing.assert_array_equal(frame["ground truth"], labels)
            np.testing.assert_array_equal(frame["预测概率"], probabilities)
        else:
            assert frame.columns.tolist() == ["label", "probability"]
            np.testing.assert_array_equal(frame["label"], labels)
            np.testing.assert_array_equal(frame["probability"], probabilities)
    assert not (tmp_path / "best.pt").exists()
