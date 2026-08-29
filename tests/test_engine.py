from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sklearn")

import petct.engine as engine
import petct.workflows as workflows
from petct.artifacts import save_predictions


def test_fit_selects_highest_validation_auc_not_lowest_loss(tmp_path: Path, monkeypatch) -> None:
    model = torch.nn.Linear(1, 2)
    validation = iter(
        [
            ({"auc": 0.50, "loss": 0.30}, None, None),
            ({"auc": 0.80, "loss": 0.20}, None, None),
            ({"auc": 0.70, "loss": 0.10}, None, None),
        ]
    )
    monkeypatch.setattr(
        engine,
        "train_epoch",
        lambda *args: ({"auc": 0.5, "loss": 1.0}, None, None),
    )
    monkeypatch.setattr(engine, "evaluate_loader", lambda *args: next(validation))
    monkeypatch.setattr(
        engine,
        "save_checkpoint",
        lambda path, current, *args: torch.save(current.state_dict(), path),
    )
    logger = type("Logger", (), {"info": lambda *args, **kwargs: None})()
    result = engine.fit(
        model,
        [],
        [],
        {"device": "cpu", "training": {"epochs": 3, "early_stopping_patience": 0}},
        tmp_path,
        logger,
    )
    assert result["best_epoch"] == 2
    assert result["validation"]["auc"] == pytest.approx(0.8)


def test_external_loader_is_not_built_when_final_evaluation_is_disabled(
    tmp_path: Path, monkeypatch
) -> None:
    accesses: list[str] = []
    config = {
        "data": {
            "train": {"name": "train"},
            "validation": {"name": "validation"},
            "external": {"site": {"name": "external"}},
        },
        "model": {},
        "training": {},
        "output": {},
        "evaluation": {"external_after_training": False},
    }
    monkeypatch.setattr(workflows, "set_seed", lambda *args: None)
    monkeypatch.setattr(workflows, "configure_logging", lambda *args: object())
    monkeypatch.setattr(workflows, "save_resolved_config", lambda *args: None)
    monkeypatch.setattr(workflows, "save_json", lambda *args: None)
    monkeypatch.setattr(
        workflows,
        "loader_from_config",
        lambda root, split, training: accesses.append(split["name"]) or split["name"],
    )
    monkeypatch.setattr(workflows, "build_model", lambda config: object())
    monkeypatch.setattr(
        workflows,
        "fit",
        lambda *args: {
            "model": object(),
            "criterion": object(),
            "device": "cpu",
            "best_epoch": 1,
            "validation": {"auc": 0.8},
        },
    )
    workflows.run_training(config, run_directory=tmp_path)
    assert accesses == ["train", "validation"]


def test_predictions_are_opt_in(tmp_path: Path) -> None:
    import numpy as np

    path = tmp_path / "predictions.csv"
    save_predictions(path, np.array([0, 1]), np.array([0.2, 0.8]), False)
    assert not path.exists()
    save_predictions(path, np.array([0, 1]), np.array([0.2, 0.8]), True)
    assert path.read_text(encoding="utf-8").splitlines()[0] == "label,probability"
    assert "patient" not in path.read_text(encoding="utf-8").lower()
