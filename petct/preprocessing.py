"""Validated PET/CT preprocessing pipeline.

Pipeline order: resampling -> CT windowing -> paired slicing -> CLAHE.
"""

from __future__ import annotations

import json
import shutil
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .config import resolve_config_path


class PairingError(ValueError):
    """Raised when a preprocessing cohort has incomplete modality pairs."""

    def __init__(self, total: int, incomplete: int):
        super().__init__(f"Found {incomplete} cases with missing paired inputs")
        self.total = total
        self.incomplete = incomplete


def _sitk() -> Any:
    try:
        import SimpleITK as sitk
    except ImportError as error:
        raise RuntimeError("Preprocessing requires SimpleITK") from error
    return sitk


def _volume_id(path: Path) -> str:
    return path.name[:-7] if path.name.endswith(".nii.gz") else path.stem


def _volumes(directory: Path) -> dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Volume directory does not exist: {directory}")
    paths = list(directory.glob("*.nii.gz")) + list(directory.glob("*.nrrd"))
    result: dict[str, Path] = {}
    for path in paths:
        identifier = _volume_id(path)
        if identifier in result:
            raise ValueError(f"Multiple volumes found for case {identifier!r} in {directory}")
        result[identifier] = path
    if not result:
        raise ValueError(f"No supported volumes found in {directory}")
    return result


def _paired_cases(directories: dict[str, Path]) -> list[tuple[str, dict[str, Path]]]:
    mappings = {name: _volumes(path) for name, path in directories.items()}
    key_sets = {name: set(mapping) for name, mapping in mappings.items()}
    union = set().union(*key_sets.values())
    incomplete = [
        identifier
        for identifier in union
        if any(identifier not in ids for ids in key_sets.values())
    ]
    if incomplete:
        raise PairingError(len(union), len(incomplete))
    return [
        (identifier, {name: mapping[identifier] for name, mapping in mappings.items()})
        for identifier in sorted(union)
    ]


def _same_geometry(image: Any, mask: Any, tolerance: float = 1e-4) -> bool:
    return (
        image.GetSize() == mask.GetSize()
        and np.allclose(image.GetSpacing(), mask.GetSpacing(), atol=tolerance)
        and np.allclose(image.GetOrigin(), mask.GetOrigin(), atol=tolerance)
        and np.allclose(image.GetDirection(), mask.GetDirection(), atol=tolerance)
    )


def _resample(image: Any, spacing: tuple[float, float, float], is_mask: bool) -> Any:
    sitk = _sitk()
    size = [
        max(1, int(round(old_size * old_spacing / new_spacing)))
        for old_size, old_spacing, new_spacing in zip(
            image.GetSize(), image.GetSpacing(), spacing, strict=True
        )
    ]
    return sitk.Resample(
        image,
        size,
        sitk.Transform(),
        sitk.sitkNearestNeighbor if is_mask else sitk.sitkLinear,
        image.GetOrigin(),
        spacing,
        image.GetDirection(),
        0.0,
        image.GetPixelID(),
    )


def _cohorts(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    preprocessing = config.get("preprocess", {})
    cohorts = preprocessing.get("cohorts")
    if not isinstance(cohorts, dict) or not cohorts:
        raise ValueError("preprocess.cohorts must be a non-empty mapping")
    return cohorts


def _path(config: dict[str, Any], value: str | Path) -> Path:
    return resolve_config_path(config, value)


def _run_cases(
    cases: list[tuple[str, dict[str, Path]]],
    process: Callable[[str, dict[str, Path]], None],
) -> dict[str, Any]:
    failures: Counter[str] = Counter()
    succeeded = 0
    for identifier, paths in cases:
        try:
            process(identifier, paths)
            succeeded += 1
        except Exception as error:  # keep processing the cohort, but never hide failures
            failures[type(error).__name__] += 1
    return {
        "total": len(cases),
        "succeeded": succeeded,
        "failed": len(cases) - succeeded,
        "failure_types": dict(failures),
    }


def _pair_or_summarize(
    directories: dict[str, Path], summary: dict[str, Any], cohort_name: str
) -> list[tuple[str, dict[str, Path]]] | None:
    try:
        return _paired_cases(directories)
    except PairingError as error:
        summary[cohort_name] = {
            "total": error.total,
            "succeeded": 0,
            "failed": error.total,
            "failure_types": {"PairingError": error.total},
        }
        return None
    except ValueError:
        summary[cohort_name] = {
            "total": 0,
            "succeeded": 0,
            "failed": 1,
            "failure_types": {"InputError": 1},
        }
        return None


def _write_summary(config: dict[str, Any], step: str, summary: dict[str, Any]) -> None:
    output = _path(
        config, config.get("preprocess", {}).get("summary_directory", "preprocess_summaries")
    )
    output.mkdir(parents=True, exist_ok=True)
    (output / f"{step}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _raise_on_failures(step: str, summary: dict[str, Any]) -> None:
    failed = sum(int(cohort["failed"]) for cohort in summary.values())
    if failed:
        raise RuntimeError(f"{step} failed for {failed} cases; see the aggregate summary")


def run_resize(config: dict[str, Any]) -> dict[str, Any]:
    """Orient paired inputs to RAI and resample images/masks to common spacing."""
    sitk = _sitk()
    settings = config["preprocess"].get("resize", {})
    spacing = tuple(float(value) for value in settings.get("spacing", [1.0, 1.0, 1.0]))
    summary: dict[str, Any] = {}
    for cohort_name, cohort in _cohorts(config).items():
        raw = _path(config, cohort["raw_root"])
        names = cohort.get("raw_directories", {})
        directories = {
            "ct": raw / names.get("ct", "ct"),
            "pet": raw / names.get("pet", "pet"),
            "ct_mask": raw / names.get("ct_mask", "ct_mask"),
            "pet_mask": raw / names.get("pet_mask", "pet_mask"),
        }
        cases = _pair_or_summarize(directories, summary, cohort_name)
        if cases is None:
            continue
        output = _path(config, cohort["resampled_root"])
        for modality in directories:
            (output / modality).mkdir(parents=True, exist_ok=True)

        def process(
            identifier: str,
            paths: dict[str, Path],
            output_directory: Path = output,
        ) -> None:
            images = {
                name: sitk.DICOMOrient(sitk.ReadImage(str(path)), "RAI")
                for name, path in paths.items()
            }
            if not _same_geometry(images["ct"], images["ct_mask"]):
                raise ValueError("CT image/mask geometry mismatch")
            if not _same_geometry(images["pet"], images["pet_mask"]):
                raise ValueError("PET image/mask geometry mismatch")
            for name, image in images.items():
                result = _resample(image, spacing, name.endswith("mask"))
                sitk.WriteImage(result, str(output_directory / name / f"{identifier}.nii.gz"))

        summary[cohort_name] = _run_cases(cases, process)
    _write_summary(config, "resize", summary)
    _raise_on_failures("resize", summary)
    return summary


def run_window(config: dict[str, Any]) -> dict[str, Any]:
    """Apply the configured lung window to CT only."""
    sitk = _sitk()
    settings = config["preprocess"].get("window", {})
    width = float(settings.get("width", 1500))
    level = float(settings.get("level", -600))
    lower, upper = level - width / 2, level + width / 2
    summary: dict[str, Any] = {}
    for cohort_name, cohort in _cohorts(config).items():
        input_directory = _path(config, cohort["resampled_root"]) / "ct"
        output = _path(config, cohort["windowed_ct_root"])
        output.mkdir(parents=True, exist_ok=True)
        cases = [
            (identifier, {"ct": path})
            for identifier, path in sorted(_volumes(input_directory).items())
        ]

        def process(
            identifier: str,
            paths: dict[str, Path],
            output_directory: Path = output,
        ) -> None:
            image = sitk.ReadImage(str(paths["ct"]))
            windowed = sitk.IntensityWindowing(image, lower, upper, 0.0, 255.0)
            windowed = sitk.Cast(windowed, sitk.sitkUInt8)
            sitk.WriteImage(windowed, str(output_directory / f"{identifier}.nii.gz"))

        summary[cohort_name] = _run_cases(cases, process)
    _write_summary(config, "window", summary)
    _raise_on_failures("window", summary)
    return summary


def _uint8(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    minimum, maximum = float(array.min()), float(array.max())
    if maximum <= minimum:
        return np.zeros(array.shape, dtype=np.uint8)
    return np.rint((array - minimum) / (maximum - minimum) * 255).astype(np.uint8)


def _center(mask: np.ndarray) -> tuple[int, int, int]:
    coordinates = np.argwhere(mask > 0)
    if coordinates.size == 0:
        raise ValueError("Mask is empty")
    low = coordinates.min(axis=0)
    high = coordinates.max(axis=0)
    return tuple(((low + high) // 2).tolist())  # z, y, x


def _cube(volume: np.ndarray, center: tuple[int, int, int], depth: int, size: int) -> np.ndarray:
    output = np.zeros((depth, size, size), dtype=volume.dtype)
    starts = (center[0] - depth // 2, center[1] - size // 2, center[2] - size // 2)
    targets = (depth, size, size)
    source_slices = []
    destination_slices = []
    for start, target, available in zip(starts, targets, volume.shape, strict=True):
        source_start = max(0, start)
        source_end = min(available, start + target)
        destination_start = max(0, -start)
        destination_end = destination_start + max(0, source_end - source_start)
        source_slices.append(slice(source_start, source_end))
        destination_slices.append(slice(destination_start, destination_end))
    output[tuple(destination_slices)] = volume[tuple(source_slices)]
    return output


def _replace_case(directory: Path, cube: np.ndarray) -> None:
    directory.parent.mkdir(parents=True, exist_ok=True)
    if directory.exists():
        resolved = directory.resolve()
        if resolved.parent != directory.parent.resolve():
            raise RuntimeError(f"Unsafe output directory: {directory}")
        shutil.rmtree(directory)
    directory.mkdir()
    for index, image in enumerate(cube):
        Image.fromarray(image, mode="L").save(directory / f"{index:03d}.png")


def run_slice(config: dict[str, Any]) -> dict[str, Any]:
    """Create paired, fixed-size CT/PET cubes centered on their respective masks."""
    sitk = _sitk()
    settings = config["preprocess"].get("slice", {})
    depth = int(settings.get("depth", 64))
    size = int(settings.get("size", 64))
    if depth <= 0 or size <= 0:
        raise ValueError("slice.depth and slice.size must be positive")
    summary: dict[str, Any] = {}
    for cohort_name, cohort in _cohorts(config).items():
        resampled = _path(config, cohort["resampled_root"])
        cases = _pair_or_summarize(
            {
                "ct": _path(config, cohort["windowed_ct_root"]),
                "pet": resampled / "pet",
                "ct_mask": resampled / "ct_mask",
                "pet_mask": resampled / "pet_mask",
            },
            summary,
            cohort_name,
        )
        if cases is None:
            continue
        output = _path(config, cohort["sliced_root"])

        def process(
            identifier: str,
            paths: dict[str, Path],
            output_directory: Path = output,
        ) -> None:
            images = {name: sitk.ReadImage(str(path)) for name, path in paths.items()}
            if not _same_geometry(images["ct"], images["ct_mask"]):
                raise ValueError("CT image/mask geometry mismatch")
            if not _same_geometry(images["pet"], images["pet_mask"]):
                raise ValueError("PET image/mask geometry mismatch")
            arrays = {name: sitk.GetArrayFromImage(image) for name, image in images.items()}
            ct_uint8 = np.clip(arrays["ct"], 0, 255).astype(np.uint8)
            ct_cube = _cube(ct_uint8, _center(arrays["ct_mask"]), depth, size)
            pet_cube = _cube(_uint8(arrays["pet"]), _center(arrays["pet_mask"]), depth, size)
            _replace_case(output_directory / "ct" / identifier, ct_cube)
            _replace_case(output_directory / "pet" / identifier, pet_cube)

        summary[cohort_name] = _run_cases(cases, process)
    _write_summary(config, "slice", summary)
    _raise_on_failures("slice", summary)
    return summary


def run_clahe(config: dict[str, Any]) -> dict[str, Any]:
    """Apply CLAHE to paired CT and PET PNG slices."""
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("CLAHE requires opencv-python-headless") from error
    settings = config["preprocess"].get("clahe", {})
    clip_limit = float(settings.get("clip_limit", 3.0))
    grid = tuple(int(value) for value in settings.get("tile_grid_size", [8, 8]))
    summary: dict[str, Any] = {}
    for cohort_name, cohort in _cohorts(config).items():
        source = _path(config, cohort["sliced_root"])
        ct_cases = {path.name: path for path in (source / "ct").iterdir() if path.is_dir()}
        pet_cases = {path.name: path for path in (source / "pet").iterdir() if path.is_dir()}
        if set(ct_cases) != set(pet_cases):
            raise ValueError(f"Cohort {cohort_name!r} has unpaired sliced CT/PET cases")
        cases = [
            (identifier, {"ct": ct_cases[identifier], "pet": pet_cases[identifier]})
            for identifier in sorted(ct_cases)
        ]
        output = _path(config, cohort["clahe_root"])

        def process(
            identifier: str,
            paths: dict[str, Path],
            output_directory: Path = output,
        ) -> None:
            modality_cubes: dict[str, list[np.ndarray]] = {}
            for modality, directory in paths.items():
                files = sorted(directory.glob("*.png"))
                if len(files) != int(config["preprocess"].get("slice", {}).get("depth", 64)):
                    raise ValueError("Unexpected slice count")
                clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid)
                images = []
                for path in files:
                    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                    if image is None:
                        raise ValueError("Unreadable PNG slice")
                    images.append(clahe.apply(image))
                modality_cubes[modality] = images
            for modality, images in modality_cubes.items():
                _replace_case(output_directory / modality / identifier, np.stack(images))

        summary[cohort_name] = _run_cases(cases, process)
    _write_summary(config, "clahe", summary)
    _raise_on_failures("clahe", summary)
    return summary


def run_preprocessing(config: dict[str, Any], step: str = "all") -> dict[str, Any]:
    functions = {"resize": run_resize, "window": run_window, "slice": run_slice, "clahe": run_clahe}
    if step == "all":
        return {name: function(config) for name, function in functions.items()}
    if step not in functions:
        raise ValueError(f"Unknown preprocessing step: {step}")
    return functions[step](config)
