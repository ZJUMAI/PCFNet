"""Command-line interface for training, preprocessing, and evaluation."""

from __future__ import annotations

import argparse
from collections.abc import Callable


def _add_config_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, help="Path to a YAML configuration file")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a YAML value with a dotted key; may be repeated",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="petct",
        description="Reproducible multimodal PET/CT classification workflows",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command, help_text in (
        ("train", "Train one model and evaluate the best checkpoint"),
        ("hpo", "Run validation-only Optuna hyperparameter optimization"),
        ("crossval", "Run stratified cross-validation"),
        ("two-stage", "Train a CNN feature extractor followed by a tree model"),
        ("evaluate", "Compute metrics, thresholds, cross-fold summaries, or DeLong tests"),
        ("split", "Generate patient-level stratified manifests"),
    ):
        command_parser = subparsers.add_parser(command, help=help_text)
        _add_config_arguments(command_parser)

    preprocess_parser = subparsers.add_parser(
        "preprocess", help="Run the PET/CT preprocessing pipeline"
    )
    preprocess_parser.add_argument(
        "step",
        choices=("all", "resize", "window", "slice", "clahe"),
        help="Pipeline step to execute",
    )
    _add_config_arguments(preprocess_parser)
    return parser


def _handler(command: str) -> Callable:
    if command == "train":
        from petct.workflows import run_training

        return run_training
    if command == "crossval":
        from petct.workflows import run_cross_validation

        return run_cross_validation
    if command == "hpo":
        from petct.hpo import run_hpo

        return run_hpo
    if command == "two-stage":
        from petct.two_stage import run_two_stage

        return run_two_stage
    if command == "evaluate":
        from petct.evaluation import run_evaluation

        return run_evaluation
    if command == "split":
        from petct.splits import run_split

        return run_split
    raise ValueError(f"Unsupported command: {command}")


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    from petct.config import load_config, validate_config

    config = load_config(args.config, args.set)
    validate_config(config, args.command)
    if args.command == "preprocess":
        from petct.preprocessing import run_preprocessing

        run_preprocessing(config, args.step)
    else:
        _handler(args.command)(config)


def legacy_main(command: str, argv: list[str] | None = None) -> None:
    """Run a canonical command from a legacy thin wrapper."""

    main([command, *(argv or [])])


def legacy_preprocess_main(step: str, argv: list[str] | None = None) -> None:
    """Run a preprocessing step from a legacy thin wrapper."""

    main(["preprocess", step, *(argv or [])])
