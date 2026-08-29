from pathlib import Path

import numpy as np

from petct.artifacts import save_predictions


def test_predictions_are_disabled_and_anonymous_by_default(tmp_path: Path) -> None:
    path = tmp_path / "predictions.csv"
    labels = np.array([0, 1])
    probabilities = np.array([0.2, 0.8])
    save_predictions(path, labels, probabilities, False)
    assert not path.exists()

    save_predictions(path, labels, probabilities, True)
    contents = path.read_text(encoding="utf-8")
    assert contents.splitlines()[0] == "label,probability"
    assert "patient" not in contents.lower()
