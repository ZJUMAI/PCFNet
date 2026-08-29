from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("yaml")
pytest.importorskip("optuna")

import petct.hpo as hpo


def test_optuna_objective_persistence_and_best_yaml(tmp_path: Path, monkeypatch) -> None:
    def fake_training(config, *, epoch_callback, **kwargs):
        score = 1.0 - abs(float(config["training"]["learning_rate"]) - 0.01)
        epoch_callback(1, {"auc": score, "loss": 1.0 - score})
        return {"results": {"validation": {"auc": score, "loss": 1.0 - score}}}

    monkeypatch.setattr(hpo, "run_training", fake_training)
    config = {
        "_meta": {"config_dir": str(tmp_path)},
        "training": {"learning_rate": 0.1},
        "hpo": {
            "study_name": "test-study",
            "trials": 2,
            "seed": 7,
            "pruner_startup_trials": 10,
            "search_space": {
                "training.learning_rate": {"type": "float", "low": 0.001, "high": 0.1, "log": True}
            },
        },
        "output": {"root": ".", "run_name": "search"},
    }
    first = hpo.run_hpo(config)
    second = hpo.run_hpo(config)
    run_directory = tmp_path / "search_hpo"
    assert first["completed_trials"] == 2
    assert second["completed_trials"] == 4
    assert (run_directory / "study.db").is_file()
    assert (run_directory / "best_config.yaml").is_file()
    assert (run_directory / "trials.csv").is_file()
