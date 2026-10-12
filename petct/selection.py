"""Explicit, opt-in epoch selection using external cohort AUCs."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from petct.config import ConfigError


@dataclass(frozen=True)
class EpochPredictions:
    """One epoch's metrics and corresponding predictions for a single cohort."""

    epoch: int
    metrics: dict[str, Any]
    labels: np.ndarray
    probabilities: np.ndarray
    identifiers: list[str] | None


class ExternalEpochSelector:
    """Keep each cohort's highest-AUC epoch; preserve the first epoch on ties.

    This uses test labels for selection and is not an unbiased final evaluation.
    No checkpoint selection, scheduling, or early stopping depends on this class.
    """

    def __init__(self) -> None:
        self.best: dict[str, EpochPredictions] = {}

    def update(
        self,
        cohort: str,
        epoch: int,
        metrics: dict[str, Any],
        labels: np.ndarray,
        probabilities: np.ndarray,
        identifiers: list[str] | None = None,
    ) -> bool:
        auc = metrics.get("auc")
        if auc is None or not math.isfinite(float(auc)):
            raise ConfigError(
                f"External AUC is undefined for {cohort}; per_cohort_best_auc requires "
                "both classes and a finite AUC. Use validation_best for ordinary evaluation."
            )
        labels = np.asarray(labels)
        probabilities = np.asarray(probabilities)
        if labels.ndim != 1 or probabilities.shape != labels.shape:
            raise ConfigError("Epoch selection requires equally sized 1D labels and probabilities")
        if identifiers is not None and len(identifiers) != len(labels):
            raise ConfigError("Epoch selection identifiers must match the prediction count")
        previous = self.best.get(cohort)
        if previous is not None:
            if not np.array_equal(labels, previous.labels) or identifiers != previous.identifiers:
                raise ConfigError(
                    f"External cohort labels or case order changed across epochs: {cohort}"
                )
            if float(auc) <= float(previous.metrics["auc"]):
                return False
        self.best[cohort] = EpochPredictions(
            epoch=int(epoch),
            metrics=copy.deepcopy(metrics),
            labels=labels.copy(),
            probabilities=probabilities.copy(),
            identifiers=None if identifiers is None else list(identifiers),
        )
        return True

    def selected_metrics(self) -> dict[str, dict[str, Any]]:
        return {
            cohort: copy.deepcopy(prediction.metrics) for cohort, prediction in self.best.items()
        }
