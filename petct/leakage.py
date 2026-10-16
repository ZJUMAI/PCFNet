"""Explicit, opt-in test contamination for leakage-sensitivity experiments."""

from __future__ import annotations

import logging
import random
from typing import Any

from torch.utils.data import ConcatDataset, DataLoader, Subset

from petct.config import ConfigError, validate_split_config
from petct.data import CombinedPETCTDataset, dataset_from_config, loader_for_dataset


class LeakageEpochLoader:
    """Resample labeled external cases per epoch without mutating any source."""

    def __init__(
        self,
        config: dict[str, Any],
        train_loader: DataLoader,
        settings: dict[str, Any],
        logger: logging.Logger,
    ) -> None:
        self.config = config
        self.train_dataset = train_loader.dataset
        self.count = settings["samples_per_cohort"]
        self.seed = int(config.get("seed", 42))
        self.logger = logger
        self.sources = {}
        for name in settings["cohorts"]:
            source = config["data"]["external"][name]
            validate_split_config(source, f"data.external.{name}", allow_groups=True)
            # Separate training views ensure paired augmentation never affects evaluation.
            if "members" in source:
                dataset = CombinedPETCTDataset(
                    [
                        dataset_from_config(config, member, training=True)
                        for member in source["members"].values()
                    ]
                )
            else:
                dataset = dataset_from_config(config, source, training=True)
            if self.count > len(dataset):
                raise ConfigError(
                    f"Leakage sample count ({self.count}) exceeds cohort {name!r} "
                    f"size ({len(dataset)}); sampling is without replacement"
                )
            self.sources[name] = dataset
        logger.warning(
            "LEAKAGE SIMULATION ENABLED: labeled test cases enter training every epoch; "
            "test cohorts remain intact. External test metrics are contaminated, "
            "not unbiased generalization performance. cohorts=%s samples_per_cohort=%d",
            ",".join(self.sources),
            self.count,
        )

    def __call__(self, epoch: int) -> DataLoader:
        """Draw afresh without replacement per cohort; allow repeats across epochs."""

        rng = random.Random(self.seed + epoch)
        additions = [
            Subset(dataset, rng.sample(range(len(dataset)), self.count))
            for dataset in self.sources.values()
        ]
        dataset = ConcatDataset([self.train_dataset, *additions])
        self.logger.info(
            "epoch=%d leakage_simulation=true original_train_cases=%d "
            "sampled_test_cases=%d total_train_cases=%d",
            epoch,
            len(self.train_dataset),
            self.count * len(self.sources),
            len(dataset),
        )
        return loader_for_dataset(self.config, dataset, training=True, seed=self.seed + epoch)
