from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
Image = pytest.importorskip("PIL.Image")
sitk = pytest.importorskip("SimpleITK")
pytest.importorskip("cv2")

from petct.preprocessing import run_preprocessing


def _write(path: Path, array: np.ndarray, *, mask: bool = False) -> None:
    image = sitk.GetImageFromArray(array.astype(np.uint8 if mask else np.float32))
    image.SetSpacing((2.0, 2.0, 2.0))
    sitk.WriteImage(image, str(path))


def test_complete_preprocessing_pipeline(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    for name in ("ct", "pet", "ct_mask", "pet_mask"):
        (raw / name).mkdir(parents=True)
    shape = (20, 20, 20)
    mask = np.zeros(shape)
    mask[7:13, 6:14, 5:15] = 1
    _write(raw / "ct" / "27.nii.gz", np.linspace(-1400, 500, np.prod(shape)).reshape(shape))
    _write(raw / "pet" / "27.nii.gz", np.linspace(0, 12, np.prod(shape)).reshape(shape))
    _write(raw / "ct_mask" / "27.nii.gz", mask, mask=True)
    _write(raw / "pet_mask" / "27.nii.gz", mask, mask=True)
    config = {
        "_meta": {"config_dir": str(tmp_path)},
        "preprocess": {
            "summary_directory": "summaries",
            "resize": {"spacing": [1.0, 1.0, 1.0]},
            "window": {"width": 1500, "level": -600},
            "slice": {"depth": 64, "size": 64},
            "clahe": {"clip_limit": 2.0, "tile_grid_size": [8, 8]},
            "cohorts": {
                "synthetic": {
                    "raw_root": "raw",
                    "resampled_root": "resampled",
                    "windowed_ct_root": "windowed",
                    "sliced_root": "sliced",
                    "clahe_root": "clahe",
                }
            },
        },
    }
    run_preprocessing(config, "all")
    resampled = sitk.ReadImage(str(tmp_path / "resampled" / "ct" / "27.nii.gz"))
    assert resampled.GetSpacing() == pytest.approx((1.0, 1.0, 1.0))
    oriented = sitk.DICOMOrient(sitk.ReadImage(str(raw / "ct" / "27.nii.gz")), "RAI")
    assert resampled.GetDirection() == pytest.approx(oriented.GetDirection())
    resampled_array = sitk.GetArrayFromImage(resampled)
    windowed_array = sitk.GetArrayFromImage(
        sitk.ReadImage(str(tmp_path / "windowed" / "27.nii.gz"))
    )
    expected_window = np.clip((resampled_array + 1350.0) / 1500.0 * 255.0, 0, 255)
    assert windowed_array == pytest.approx(expected_window, abs=1.0)
    for root in (tmp_path / "sliced", tmp_path / "clahe"):
        for modality in ("ct", "pet"):
            files = sorted((root / modality / "27").glob("*.png"))
            assert len(files) == 64
            assert files[0].name == "000.png"
            assert files[-1].name == "063.png"
            assert Image.open(files[0]).size == (64, 64)
    assert "27" not in (tmp_path / "summaries" / "slice.json").read_text(encoding="utf-8")


def test_missing_preprocessing_pair_fails_with_anonymous_summary(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    for name in ("ct", "pet", "ct_mask", "pet_mask"):
        (raw / name).mkdir(parents=True)
    array = np.ones((4, 4, 4))
    _write(raw / "ct" / "private-case.nii.gz", array)
    _write(raw / "ct_mask" / "private-case.nii.gz", array, mask=True)
    _write(raw / "pet_mask" / "private-case.nii.gz", array, mask=True)
    config = {
        "_meta": {"config_dir": str(tmp_path)},
        "preprocess": {
            "summary_directory": "summaries",
            "cohorts": {
                "synthetic": {
                    "raw_root": "raw",
                    "resampled_root": "resampled",
                    "windowed_ct_root": "windowed",
                    "sliced_root": "sliced",
                    "clahe_root": "clahe",
                }
            },
        },
    }
    with pytest.raises(RuntimeError, match="failed"):
        run_preprocessing(config, "resize")
    summary = (tmp_path / "summaries" / "resize.json").read_text(encoding="utf-8")
    assert "private-case" not in summary
    assert '"failed": 1' in summary
