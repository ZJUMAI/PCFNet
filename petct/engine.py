"""Shared training and evaluation engine."""

from __future__ import annotations

import copy
import logging
import random
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from petct.artifacts import save_checkpoint, save_history, snapshot_state_dict
from petct.config import checkpoint_saving_enabled
from petct.losses import FocalLoss
from petct.metrics import binary_metrics


def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {requested}")
    return torch.device(requested)


def build_criterion(config: dict[str, Any]) -> torch.nn.Module:
    loss_config = config.get("training", {}).get("loss", {})
    name = loss_config.get("name", "focal")
    if name == "focal":
        return FocalLoss(
            alpha=float(loss_config.get("alpha", 0.25)),
            gamma=float(loss_config.get("gamma", 2.0)),
        )
    if name == "cross_entropy":
        return torch.nn.CrossEntropyLoss()
    raise ValueError(f"Unsupported loss function: {name}")


def build_optimizer(model: torch.nn.Module, config: dict[str, Any]) -> torch.optim.Optimizer:
    training = config.get("training", {})
    optimizer_config = training.get("optimizer", {})
    name = str(optimizer_config.get("name", "adamw")).lower()
    learning_rate = float(
        optimizer_config.get("learning_rate", training.get("learning_rate", 1e-4))
    )
    weight_decay = float(optimizer_config.get("weight_decay", training.get("weight_decay", 1e-4)))
    if name == "adamw":
        return AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=float(optimizer_config.get("momentum", 0.9)),
        )
    raise ValueError(f"Unsupported optimizer: {name}")


def _run_loader(
    model: torch.nn.Module,
    loader,
    criterion: torch.nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    training = optimizer is not None
    model.train(training)
    losses: list[float] = []
    labels: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            ct = batch["ct"].to(device, non_blocking=True)
            pet = batch["pet"].to(device, non_blocking=True)
            target = batch["label"].to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(ct, pet)
            loss = criterion(logits, target)
            if training:
                loss.backward()
                optimizer.step()
            losses.append(float(loss.detach().cpu()))
            labels.append(target.detach().cpu().numpy())
            probabilities.append(torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy())
    all_labels = np.concatenate(labels)
    all_probabilities = np.concatenate(probabilities)
    metrics = binary_metrics(all_labels, all_probabilities)
    metrics["loss"] = float(np.mean(losses))
    return metrics, all_labels, all_probabilities


def train_epoch(model, loader, criterion, optimizer, device):
    return _run_loader(model, loader, criterion, device, optimizer)


def evaluate_loader(model, loader, criterion, device):
    return _run_loader(model, loader, criterion, device, None)


def fit(
    model: torch.nn.Module,
    train_loader,
    validation_loader,
    config: dict[str, Any],
    run_directory: Path,
    logger: logging.Logger,
    epoch_callback: Callable[[int, dict[str, Any]], None] | None = None,
    *,
    epoch_observer: Callable[[int, torch.nn.Module, torch.nn.Module, torch.device], None]
    | None = None,
) -> dict[str, Any]:
    """Select the best model by validation AUC; checkpoint export is opt-in."""

    training = config.get("training", {})
    save_best = checkpoint_saving_enabled(config)
    device = resolve_device(config.get("device", "auto"))
    model.to(device)
    criterion = build_criterion(config)
    optimizer = build_optimizer(model, config)
    scheduler_config = training.get("scheduler", {})
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(scheduler_config.get("factor", training.get("scheduler_factor", 0.2))),
        patience=int(scheduler_config.get("patience", training.get("scheduler_patience", 10))),
    )
    epochs = int(training.get("epochs", 100))
    early_stopping = int(training.get("early_stopping_patience", 20))
    best_score = -float("inf")
    best_validation: dict[str, Any] | None = None
    best_epoch: int | None = None
    best_state: dict[str, Any] | None = None
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []
    checkpoint_path = run_directory / "best.pt"

    for epoch in range(1, epochs + 1):
        train_metrics, _, _ = train_epoch(model, train_loader, criterion, optimizer, device)
        validation_metrics, _, _ = evaluate_loader(model, validation_loader, criterion, device)
        scheduler.step(float(validation_metrics["loss"]))
        auc = validation_metrics.get("auc")
        record = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"validation_{key}": value for key, value in validation_metrics.items()},
        }
        history.append(record)
        logger.info(
            "epoch=%d train_loss=%.5f validation_loss=%.5f validation_auc=%s",
            epoch,
            train_metrics["loss"],
            validation_metrics["loss"],
            "NA" if auc is None else f"{auc:.5f}",
        )
        improved = (auc is not None and float(auc) > best_score) or best_epoch is None
        if improved:
            if auc is not None:
                best_score = float(auc)
            best_validation = copy.deepcopy(validation_metrics)
            best_epoch = epoch
            epochs_without_improvement = 0
            best_state = snapshot_state_dict(model)
            if save_best:
                save_checkpoint(
                    checkpoint_path,
                    model,
                    config.get("model", {}),
                    epoch,
                    validation_metrics,
                )
        else:
            epochs_without_improvement += 1
        if epoch_callback:
            epoch_callback(epoch, validation_metrics)
        if epoch_observer:
            epoch_observer(epoch, model, criterion, device)
        if early_stopping > 0 and epochs_without_improvement >= early_stopping:
            logger.info("early_stopping epoch=%d", epoch)
            break

    save_history(run_directory / "history.csv", history)
    if best_state is None:
        raise RuntimeError("Training did not produce a best model")
    model.load_state_dict(best_state)
    return {
        "model": model,
        "device": device,
        "criterion": criterion,
        "best_epoch": int(best_epoch),
        "validation": best_validation,
        "history": history,
    }


def validation_auc(labels: np.ndarray, probabilities: np.ndarray) -> float:
    if np.unique(labels).size != 2:
        raise RuntimeError("Validation AUC requires both classes")
    return float(roc_auc_score(labels, probabilities))
