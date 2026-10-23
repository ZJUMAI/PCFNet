"""Isolated, reproducible split-seed sweeps with one subprocess per GPU slot."""

from __future__ import annotations

import argparse
import copy
import json
import os
import queue
import random
import re
import shlex
import signal
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .config import (
    ConfigError,
    apply_overrides,
    load_config,
    public_config,
    resolve_config_path,
    validate_config,
)

REPOSITORY = Path(__file__).resolve().parents[1]
TRAIN_MANAGED = {
    "device",
    "cross_validation.folds",
    "data.train.manifest",
    "data.validation.manifest",
    "output.root",
    "output.run_name",
}
SPLIT_MANAGED = {
    "split.seed",
    "split.folds",
    "split.repeats",
    "split.output_directory",
    "split.filename_pattern",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _positive(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def _seed(value: str) -> int:
    number = int(value)
    if not 0 <= number < 2**32:
        raise argparse.ArgumentTypeError("must be between 0 and 4294967295")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPOSITORY / "configs/train.yaml")
    parser.add_argument("--split-config", type=Path, default=REPOSITORY / "configs/split.yaml")
    seeds = parser.add_mutually_exclusive_group()
    seeds.add_argument("--runs", type=_positive, default=5, help="Number of random split seeds")
    seeds.add_argument("--seeds", type=_seed, nargs="+", help="Replay explicit split seeds")
    parser.add_argument(
        "--sampler-seed", type=_seed, help="Make random seed generation reproducible"
    )
    parser.add_argument("--folds", type=_positive, default=5)
    parser.add_argument("--gpus", nargs="+", default=["3"], help="GPU slots, e.g. 3 5 or 3,5")
    parser.add_argument(
        "--parallel", type=_positive, help="Concurrent groups; default: all GPU slots"
    )
    parser.add_argument("--output-root", type=Path, help="Default: the training YAML's output.root")
    parser.add_argument("--batch-name", help="New batch directory name; default: UTC timestamp")
    parser.add_argument("--run-prefix", default="pcfnet_crossval")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Training overrides, e.g. training.leakage_simulation.enabled=true",
    )
    parser.add_argument(
        "--split-set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Split configuration overrides, e.g. split.metadata=/path/labels.xlsx",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print the plan without writing or running"
    )
    return parser


def sample_seeds(count: int, sampler_seed: int | None) -> list[int]:
    if not 0 < count <= 2**32:
        raise ConfigError("The number of seeds must be between 1 and 4294967296")
    generator = random.SystemRandom() if sampler_seed is None else random.Random(sampler_seed)
    return generator.sample(range(2**32), count)


def _check_overrides(overrides: list[str], managed: set[str]) -> None:
    for override in overrides:
        key = override.split("=", 1)[0]
        if any(
            key == item or key.startswith(item + ".") or item.startswith(key + ".")
            for item in managed
        ):
            raise ConfigError(f"{key} is managed by the sweep; use its dedicated options instead")


def _directory_name(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise ConfigError("Batch names and run prefixes must use letters, digits, _, - or .")
    return value


def _overrides(values: dict[str, Any]) -> list[str]:
    return [f"{key}={json.dumps(value, ensure_ascii=False)}" for key, value in values.items()]


def _command(command: str, path: Path, overrides: list[str]) -> list[str]:
    result = [sys.executable, "-m", "petct", command, "--config", str(path)]
    for override in overrides:
        result.extend(("--set", override))
    return result


def _snapshot_configs(training: dict[str, Any], split: dict[str, Any]) -> dict[str, dict]:
    """Freeze effective YAMLs and retain the original meaning of relative input paths."""
    training_snapshot = copy.deepcopy(public_config(training))

    def resolve_source(source: dict[str, Any]) -> None:
        if "members" in source:
            for member in source["members"].values():
                resolve_source(member)
        else:
            for key in ("manifest", "metadata", "ct_root", "pet_root"):
                if source.get(key):
                    source[key] = resolve_config_path(training, source[key]).as_posix()

    data = training_snapshot.get("data", {})
    for source in (data.get("train", {}), data.get("validation", {})):
        resolve_source(source)
    for source in data.get("external", {}).values():
        resolve_source(source)
    split_snapshot = copy.deepcopy(public_config(split))
    settings = split_snapshot.get("split", {})
    if settings.get("metadata"):
        settings["metadata"] = resolve_config_path(split, settings["metadata"]).as_posix()
    if settings.get("source_manifests") is not None:
        settings["source_manifests"] = [
            resolve_config_path(split, path).as_posix() for path in settings["source_manifests"]
        ]
    return {"training": training_snapshot, "split": split_snapshot}


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    """Validate configuration structure without opening data or initializing CUDA."""
    _check_overrides(args.set, TRAIN_MANAGED)
    _check_overrides(args.split_set, SPLIT_MANAGED)
    if args.folds < 2:
        raise ConfigError("At least two folds are required")
    gpu_slots = [part.strip() for value in args.gpus for part in value.split(",")]
    if not gpu_slots or any(
        not re.fullmatch(r"\d+|GPU-[\w-]+|MIG-[\w/-]+", gpu) for gpu in gpu_slots
    ):
        raise ConfigError("--gpus must contain GPU indices or CUDA GPU/MIG UUIDs")
    parallel = args.parallel or len(gpu_slots)
    if parallel > len(gpu_slots):
        raise ConfigError(
            "--parallel cannot exceed GPU slots; repeat a GPU ID explicitly to share it"
        )
    seeds = args.seeds if args.seeds is not None else sample_seeds(args.runs, args.sampler_seed)
    if len(seeds) != len(set(seeds)):
        raise ConfigError("Split seeds must be unique within a batch")
    config_path = args.config.expanduser().resolve()
    split_path = args.split_config.expanduser().resolve()
    training = load_config(config_path, args.set)
    split = load_config(split_path, args.split_set)
    mode = split.get("split", {}).get("mode", "kfold")
    if mode not in {"kfold", "repeated_holdout"}:
        raise ConfigError("The sweep supports kfold or repeated_holdout splits, not single holdout")
    root = (
        args.output_root.expanduser().resolve()
        if args.output_root
        else resolve_config_path(training, training.get("output", {}).get("root", "../runs"))
    )
    name = args.batch_name or datetime.now(timezone.utc).strftime("seed_sweep_%Y%m%dT%H%M%S_%fZ")
    directory = root / _directory_name(name)
    prefix = _directory_name(args.run_prefix)
    jobs = []
    for seed in seeds:
        run_name = f"{prefix}_seed_{seed}"
        run_directory = directory / run_name
        split_directory = run_directory / "splits"
        split_overrides = _overrides(
            {
                "split.seed": seed,
                f"split.{'folds' if mode == 'kfold' else 'repeats'}": args.folds,
                "split.output_directory": split_directory.as_posix(),
                "split.filename_pattern": "fold_{fold}_{split}.txt",
            }
        )
        training_overrides = _overrides(
            {
                "device": "cuda:0",
                "cross_validation.folds": args.folds,
                "data.train.manifest": (split_directory / "fold_{fold}_train.txt").as_posix(),
                "data.validation.manifest": (
                    split_directory / "fold_{fold}_validation.txt"
                ).as_posix(),
                "output.root": directory.as_posix(),
                "output.run_name": run_name,
            }
        )
        validate_config(apply_overrides(split, split_overrides), "split")
        validate_config(apply_overrides(training, training_overrides), "crossval")
        jobs.append(
            {
                "split_seed": seed,
                "status": "pending",
                "gpu": None,
                "run_directory": str(run_directory),
                "split_directory": str(split_directory),
                "split_command": _command(
                    "split", directory / "configs/split.yaml", split_overrides
                ),
                "training_command": _command(
                    "crossval", directory / "configs/training.yaml", training_overrides
                ),
            }
        )
    return {
        "created_at": _now(),
        "status": "planned",
        "directory": str(directory),
        "working_directory": str(REPOSITORY),
        "python_version": sys.version,
        "gpu_slots": gpu_slots,
        "parallel": parallel,
        "sampler_seed": args.sampler_seed,
        "training_seed": training.get("seed", 42),
        "folds_per_group": args.folds,
        "split_mode": mode,
        "jobs": jobs,
        "source_configs": {"training": str(config_path), "split": str(split_path)},
        "config_snapshots": _snapshot_configs(training, split),
        "training_overrides": args.set,
        "split_overrides": args.split_set,
        "evaluation": training.get("evaluation", {}),
        "leakage_simulation": training.get("training", {}).get("leakage_simulation", {}),
        "notice": "No test-based seed ranking is performed. Leakage simulation and external-best "
        "epoch selection produce exploratory, not unbiased, external test estimates.",
    }


def _terminate_process(process: subprocess.Popen) -> None:
    """Stop only a subprocess tree started by this batch."""
    if process.poll() is not None:
        return
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        else:
            os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            if os.name == "nt":
                process.kill()
            else:
                os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)
    except ProcessLookupError:
        pass


class BatchRunner:
    """Serialize status writes while independent workers own GPU slots."""

    def __init__(self, plan: dict[str, Any]) -> None:
        self.plan = copy.deepcopy(plan)
        self.directory = Path(plan["directory"])
        self.lock = threading.Lock()
        self.cancel = threading.Event()
        self.active: dict[int, subprocess.Popen] = {}
        self.slots: queue.Queue[str] = queue.Queue()
        for gpu in plan["gpu_slots"]:
            self.slots.put(gpu)

    def _save(self) -> None:
        """Called under the lock; replace a same-directory temporary JSON atomically."""
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", suffix=".tmp", dir=self.directory, delete=False
        ) as handle:
            temporary = Path(handle.name)
            json.dump(self.plan, handle, indent=2, ensure_ascii=False, allow_nan=False)
        try:
            os.replace(temporary, self.directory / "batch_summary.json")
        finally:
            temporary.unlink(missing_ok=True)

    def _update(self, job: dict[str, Any], **values: Any) -> None:
        with self.lock:
            job.update(values)
            self._save()
        if "status" in values:
            print(
                f"split_seed={job['split_seed']} gpu={job['gpu']} status={job['status']}",
                flush=True,
            )

    def _phase(self, job: dict[str, Any], phase: str, env: dict[str, str]) -> int | None:
        if self.cancel.is_set():
            return None
        command = job[f"{phase}_command"]
        logfile = Path(job["run_directory"]) / f"{phase}.log"
        with logfile.open("w", encoding="utf-8") as handle:
            handle.write(f"CUDA_VISIBLE_DEVICES={job['gpu']} {shlex.join(command)}\n\n")
            handle.flush()
            process = subprocess.Popen(
                command,
                cwd=self.plan["working_directory"],
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=os.name != "nt",
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
            )
            with self.lock:
                self.active[process.pid] = process
            try:
                if self.cancel.is_set():
                    _terminate_process(process)
                return process.wait()
            finally:
                with self.lock:
                    self.active.pop(process.pid, None)

    def _job(self, job: dict[str, Any]) -> None:
        gpu = self.slots.get()
        try:
            if self.cancel.is_set():
                self._update(job, status="cancelled")
                return
            Path(job["run_directory"]).mkdir(exist_ok=False)
            env = dict(
                os.environ, CUDA_VISIBLE_DEVICES=gpu, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8"
            )
            self._update(job, gpu=gpu, started_at=_now(), status="splitting")
            split_code = self._phase(job, "split", env)
            self._update(job, split_return_code=split_code)
            if self.cancel.is_set():
                self._update(job, status="cancelled")
            elif split_code != 0:
                self._update(job, status="split_failed")
            else:
                self._update(job, status="training")
                code = self._phase(job, "training", env)
                self._update(
                    job,
                    training_return_code=code,
                    status="cancelled"
                    if self.cancel.is_set()
                    else "completed"
                    if code == 0
                    else "training_failed",
                )
        except Exception as error:
            self._update(job, status="failed", error=f"{type(error).__name__}: {error}")
        finally:
            self._update(job, finished_at=_now())
            self.slots.put(gpu)

    def stop(self) -> None:
        self.cancel.set()
        with self.lock:
            processes = list(self.active.values())
        for process in processes:
            _terminate_process(process)

    def run(self) -> int:
        self.directory.mkdir(parents=True, exist_ok=False)
        config_directory = self.directory / "configs"
        config_directory.mkdir()
        for name, config in self.plan["config_snapshots"].items():
            (config_directory / f"{name}.yaml").write_text(
                yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
            )
        self.plan["status"] = "running"
        self._save()
        print(f"Batch status: {self.directory / 'batch_summary.json'}", flush=True)
        if len(set(self.plan["gpu_slots"])) != len(self.plan["gpu_slots"]):
            print(
                "WARNING: repeated GPU IDs explicitly share GPU memory between groups.", flush=True
            )
        executor = ThreadPoolExecutor(max_workers=self.plan["parallel"])
        interrupted = False
        try:
            futures = [executor.submit(self._job, job) for job in self.plan["jobs"]]
            for future in as_completed(futures):
                future.result()
        except KeyboardInterrupt:
            interrupted = True
            print("Stopping batch subprocesses and their workers...", flush=True)
            self.stop()
        finally:
            executor.shutdown(wait=True, cancel_futures=interrupted)
            with self.lock:
                for job in self.plan["jobs"]:
                    if job["status"] == "pending":
                        job.update(status="cancelled", finished_at=_now())
                successful = all(job["status"] == "completed" for job in self.plan["jobs"])
                self.plan.update(
                    status="interrupted"
                    if interrupted
                    else "completed"
                    if successful
                    else "failed",
                    finished_at=_now(),
                )
                self._save()
        return 130 if interrupted else 0 if successful else 1


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        plan = build_plan(args)
        if args.dry_run:
            print(json.dumps(plan, indent=2, ensure_ascii=False))
            return 0
        return BatchRunner(plan).run()
    except (ConfigError, FileExistsError) as error:
        parser.error(str(error))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
