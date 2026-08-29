from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
Image = pytest.importorskip("PIL.Image")
pytest.importorskip("torch")
pytest.importorskip("torchio")

import torchio as tio

from petct.config import ConfigError
from petct.data import PETCTDataset, natural_key, parse_binary_label


def _case(root: Path, identifier: str, *, slices: int = 64) -> None:
    directory = root / identifier
    directory.mkdir(parents=True)
    for index in range(slices):
        array = np.full((64, 64), index, dtype=np.uint8)
        Image.fromarray(array).save(directory / f"{index}.png")


def _inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    ct, pet = tmp_path / "ct", tmp_path / "pet"
    _case(ct, "27")
    _case(pet, "27")
    manifest = tmp_path / "manifest.txt"
    manifest.write_text("27\n", encoding="utf-8")
    metadata = tmp_path / "metadata.csv"
    pd.DataFrame({"patient": [27], "outcome": [1]}).to_csv(metadata, index=False)
    return manifest, metadata, ct, pet


def test_strict_label_mapping() -> None:
    assert parse_binary_label("yes") == 1
    assert parse_binary_label(0) == 0
    with pytest.raises(ConfigError):
        parse_binary_label("unknown")


def test_natural_sort() -> None:
    names = [Path("10.png"), Path("2.png"), Path("1.png")]
    assert [path.name for path in sorted(names, key=natural_key)] == ["1.png", "2.png", "10.png"]


def test_dataset_returns_fixed_volume(tmp_path: Path) -> None:
    manifest, metadata, ct, pet = _inputs(tmp_path)
    dataset = PETCTDataset(manifest, metadata, ct, pet, id_column="patient", label_column="outcome")
    sample = dataset[0]
    assert tuple(sample["ct"].shape) == (64, 64, 64)
    assert tuple(sample["pet"].shape) == (64, 64, 64)
    assert sample["label"].item() == 1


def test_missing_modality_is_rejected(tmp_path: Path) -> None:
    manifest, metadata, ct, pet = _inputs(tmp_path)
    for path in (pet / "27").glob("*.png"):
        path.unlink()
    (pet / "27").rmdir()
    with pytest.raises(ConfigError, match="Missing paired"):
        PETCTDataset(manifest, metadata, ct, pet, id_column="patient", label_column="outcome")


def test_spatial_augmentation_is_synchronized(tmp_path: Path) -> None:
    manifest, metadata, ct, pet = _inputs(tmp_path)
    dataset = PETCTDataset(
        manifest,
        metadata,
        ct,
        pet,
        id_column="patient",
        label_column="outcome",
        training=True,
        augment_probability=1.0,
    )
    dataset.transform = tio.RandomFlip(axes=(0, 1, 2), flip_probability=1.0)
    sample = dataset[0]
    assert np.array_equal(sample["ct"].numpy(), sample["pet"].numpy())
