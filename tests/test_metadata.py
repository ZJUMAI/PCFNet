from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("yaml")

from petct.config import ConfigError, load_config
from petct.metadata import (
    DEFAULT_NEGATIVE_VALUES,
    DEFAULT_POSITIVE_VALUES,
    load_metadata,
    parse_binary_label,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("是", 1),
        ("否", 0),
        (" 是 ", 1),
        (" 否 ", 0),
        (1, 1),
        (0, 0),
        (1.0, 1),
        (0.0, 0),
        ("1", 1),
        ("0", 0),
        ("1.0", 1),
        ("0.0", 0),
        ("YES", 1),
        ("NO", 0),
        (True, 1),
        (False, 0),
    ],
)
def test_default_binary_label_mapping(value, expected: int) -> None:
    assert parse_binary_label(value) == expected


@pytest.mark.parametrize("value", ["unknown", "", None, float("nan"), 2])
def test_unknown_binary_labels_remain_errors(value) -> None:
    with pytest.raises(ConfigError, match="Unsupported binary label"):
        parse_binary_label(value)


@pytest.mark.parametrize(
    "filename", ["train.yaml", "compare.yaml", "hpo.yaml", "two_stage.yaml", "evaluate.yaml"]
)
def test_example_configs_accept_chinese_and_numeric_labels(filename: str) -> None:
    config = load_config(Path(__file__).resolve().parents[1] / "configs" / filename)
    data = config["data"]
    positive = {
        str(value).strip().lower()
        for value in (data.get("positive_values") or DEFAULT_POSITIVE_VALUES)
    }
    negative = {
        str(value).strip().lower()
        for value in (data.get("negative_values") or DEFAULT_NEGATIVE_VALUES)
    }
    for value, expected in (("是", 1), ("否", 0), (1.0, 1), (0.0, 0)):
        assert parse_binary_label(value, positive, negative) == expected


@pytest.mark.parametrize("suffix", [".csv", ".xlsx"])
def test_metadata_preserves_text_ids_with_leading_zeros(tmp_path, suffix):
    frame = pd.DataFrame({"id": ["003", "001", "12.0"], "label": [0, 1, 0]})
    path = tmp_path / f"labels{suffix}"
    if suffix == ".csv":
        frame.to_csv(path, index=False)
    else:
        pytest.importorskip("openpyxl")
        frame.to_excel(path, index=False)
    metadata = load_metadata(path, "id", "label")
    assert metadata.index.tolist() == ["003", "001", "12"]
    assert metadata["label"].tolist() == [0, 1, 0]
