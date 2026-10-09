"""Strict paired PET/CT dataset and data-loader construction."""

from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchio as tio
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from petct.config import ConfigError, resolve_config_path
from petct.metadata import (
    DEFAULT_NEGATIVE_VALUES,
    DEFAULT_POSITIVE_VALUES,
    load_metadata,
    normalize_identifier,
    parse_binary_label,
)

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def _seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def natural_key(path: Path) -> list[int | str]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


class PETCTDataset(Dataset):
    """Paired CT/PET PNG volumes with strict metadata and shape validation."""

    def __init__(
        self,
        manifest_path: str | Path,
        metadata_path: str | Path,
        ct_root: str | Path,
        pet_root: str | Path,
        *,
        id_column: str = "patient_id",
        label_column: str = "label",
        depth: int = 64,
        image_size: int = 64,
        training: bool = False,
        augment_probability: float = 0.0,
        strict_slice_count: bool = True,
        seed: int = 513,
        positive_values: list[Any] | None = None,
        negative_values: list[Any] | None = None,
    ):
        self.manifest_path = Path(manifest_path)
        self.ct_root = Path(ct_root)
        self.pet_root = Path(pet_root)
        self.depth = int(depth)
        self.image_size = int(image_size)
        self.training = training
        self.augment_probability = float(augment_probability)
        self.strict_slice_count = strict_slice_count
        self.positive_values = {
            str(value).strip().lower() for value in (positive_values or DEFAULT_POSITIVE_VALUES)
        }
        self.negative_values = {
            str(value).strip().lower() for value in (negative_values or DEFAULT_NEGATIVE_VALUES)
        }

        for path, description in (
            (self.manifest_path, "manifest"),
            (self.ct_root, "CT root"),
            (self.pet_root, "PET root"),
        ):
            if not path.exists():
                raise ConfigError(f"{description} does not exist: {path}")

        metadata = load_metadata(Path(metadata_path), id_column, label_column)
        identifiers = [
            line.strip()
            for line in self.manifest_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not identifiers:
            raise ConfigError(f"Manifest is empty: {self.manifest_path}")
        if len(identifiers) != len(set(identifiers)):
            raise ConfigError(f"Manifest contains duplicate identifiers: {self.manifest_path}")

        self.samples: list[tuple[Path, Path, int]] = []
        for identifier in identifiers:
            normalized_id = normalize_identifier(identifier)
            if normalized_id not in metadata.index:
                raise ConfigError(f"Sample {normalized_id!r} is absent from metadata")
            ct_dir = self.ct_root / identifier
            pet_dir = self.pet_root / identifier
            if not ct_dir.is_dir() or not pet_dir.is_dir():
                raise ConfigError(f"Missing paired CT/PET directory for sample {identifier!r}")
            label = parse_binary_label(
                metadata.at[normalized_id, label_column],
                self.positive_values,
                self.negative_values,
            )
            self.samples.append((ct_dir, pet_dir, label))

        self.transform = tio.Compose(
            [
                tio.RandomFlip(axes=(0, 1, 2), flip_probability=0.5),
                tio.RandomAffine(scales=(0.9, 1.1), degrees=10, translation=2),
            ]
        )
        self.seed = int(seed)

    def __len__(self) -> int:
        return len(self.samples)

    def _load_volume(self, directory: Path) -> torch.Tensor:
        files = sorted(
            [path for path in directory.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES],
            key=natural_key,
        )
        if self.strict_slice_count and len(files) != self.depth:
            raise ConfigError(f"Expected {self.depth} slices in {directory}, found {len(files)}")
        if not files:
            raise ConfigError(f"No image slices found in {directory}")

        slices: list[np.ndarray] = []
        for path in files:
            with Image.open(path) as image:
                array = np.asarray(image.convert("L"), dtype=np.float32)
            if array.shape != (self.image_size, self.image_size):
                raise ConfigError(
                    f"Expected {self.image_size}x{self.image_size} image at {path}, "
                    f"found {array.shape}"
                )
            slices.append(array)
        volume = np.stack(slices, axis=0)
        if len(slices) < self.depth:
            pad = self.depth - len(slices)
            volume = np.pad(volume, ((0, pad), (0, 0), (0, 0)))
        elif len(slices) > self.depth:
            start = (len(slices) - self.depth) // 2
            volume = volume[start : start + self.depth]
        return torch.from_numpy(volume / 255.0)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        ct_dir, pet_dir, label = self.samples[index]
        ct = self._load_volume(ct_dir)
        pet = self._load_volume(pet_dir)
        if self.training and torch.rand(()).item() < self.augment_probability:
            subject = tio.Subject(
                ct=tio.ScalarImage(tensor=ct.unsqueeze(0)),
                pet=tio.ScalarImage(tensor=pet.unsqueeze(0)),
            )
            transformed = self.transform(subject)
            ct = transformed.ct.data.squeeze(0)
            pet = transformed.pet.data.squeeze(0)
        return {
            "ct": ct.float(),
            "pet": pet.float(),
            "label": torch.tensor(label, dtype=torch.long),
        }


def dataset_from_config(
    root_config: dict[str, Any], split_config: dict[str, Any], *, training: bool
) -> PETCTDataset:
    """Construct a dataset from a split configuration."""

    data_defaults = root_config.get("data", {})
    kwargs = {
        "id_column": split_config.get("id_column", data_defaults.get("id_column", "patient_id")),
        "label_column": split_config.get(
            "label_column", data_defaults.get("label_column", "label")
        ),
        "depth": data_defaults.get("depth", 64),
        "image_size": data_defaults.get("image_size", 64),
        "training": training,
        "augment_probability": (data_defaults.get("augment_probability", 0.0) if training else 0.0),
        "strict_slice_count": data_defaults.get("strict_slice_count", True),
        "seed": root_config.get("seed", 513),
        "positive_values": data_defaults.get("positive_values"),
        "negative_values": data_defaults.get("negative_values"),
    }
    return PETCTDataset(
        resolve_config_path(root_config, split_config["manifest"]),
        resolve_config_path(root_config, split_config["metadata"]),
        resolve_config_path(root_config, split_config["ct_root"]),
        resolve_config_path(root_config, split_config["pet_root"]),
        **kwargs,
    )


def loader_from_config(
    config: dict[str, Any], split_config: dict[str, Any], *, training: bool
) -> DataLoader:
    dataset = dataset_from_config(config, split_config, training=training)
    training_config = config.get("training", {})
    seed = int(config.get("seed", 513))

    return DataLoader(
        dataset,
        batch_size=int(training_config.get("batch_size", 8)),
        shuffle=training,
        num_workers=int(training_config.get("num_workers", 0)),
        pin_memory=bool(training_config.get("pin_memory", False)),
        drop_last=False,
        generator=torch.Generator().manual_seed(seed),
        worker_init_fn=_seed_worker,
    )
