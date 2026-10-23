"""GPU-independent checks for isolated split-seed scheduling and reproducibility."""

import json
import os
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

yaml = pytest.importorskip("yaml")

from petct import seed_sweep as sweep
from petct.config import ConfigError, load_config


@pytest.fixture
def profiles(tmp_path):
    directory = tmp_path / "configs"
    directory.mkdir()
    (tmp_path / "labels.csv").write_text(
        "id,label\n" + "".join(f"{index},{index % 2}\n" for index in range(20)), encoding="utf-8"
    )
    for name, indices in (("original_train", range(16)), ("original_validation", range(16, 20))):
        (tmp_path / f"{name}.txt").write_text("\n".join(map(str, indices)) + "\n", encoding="utf-8")
    source = dict(
        manifest="../original_train.txt",
        metadata="../labels.csv",
        ct_root="../ct",
        pet_root="../pet",
    )
    training = {
        "seed": 42,
        "data": {
            "id_column": "id",
            "label_column": "label",
            "train": source,
            "validation": dict(source, manifest="../original_validation.txt"),
            "external": {name: source for name in ("multi_hospital", "france", "SPH_test")},
        },
        "model": {"name": "fusion3d"},
        "training": {"epochs": 1, "optimizer": {"learning_rate": 0.001}},
        "output": {"root": "../runs"},
    }
    split = {
        "split": {
            "mode": "kfold",
            "folds": 5,
            "seed": 42,
            "metadata": "../labels.csv",
            "id_column": "id",
            "label_column": "label",
            "source_manifests": ["../original_train.txt", "../original_validation.txt"],
        }
    }
    for name, config in (("training", training), ("split", split)):
        (directory / f"{name}.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    return directory


def arguments(profiles, *extra):
    return sweep.build_parser().parse_args(
        [
            "--config",
            str(profiles / "training.yaml"),
            "--split-config",
            str(profiles / "split.yaml"),
            "--batch-name",
            "batch",
            *extra,
        ]
    )


def command_overrides(command):
    return [command[index + 1] for index, token in enumerate(command) if token == "--set"]


def effective_config(command):
    return load_config(command[command.index("--config") + 1], command_overrides(command))


def test_random_seed_sampling_is_reproducible_and_unique():
    first = sweep.sample_seeds(20, 42)
    assert first == sweep.sample_seeds(20, 42)
    assert first != sweep.sample_seeds(20, 43)
    assert len(set(first)) == 20
    assert all(0 <= seed < 2**32 for seed in first)
    assert len(sweep.sample_seeds(2, None)) == 2


def test_plan_isolates_paths_and_preserves_requested_experiment(profiles):
    options = arguments(
        profiles,
        "--seeds",
        "17",
        "29",
        "--gpus",
        "3,5",
        "--parallel",
        "2",
        "--set",
        "training.leakage_simulation.enabled=true",
        "--set",
        "training.leakage_simulation.cohort_samples.multi_hospital=2",
        "--set",
        "training.leakage_simulation.cohort_samples.france=1",
        "--set",
        "training.leakage_simulation.cohort_samples.SPH_test=3",
        "--set",
        "evaluation.external_selection=per_cohort_best_auc",
    )
    plan = sweep.build_plan(options)
    assert plan["training_seed"] == 42
    assert plan["gpu_slots"] == ["3", "5"]
    assert plan["parallel"] == 2
    assert plan["leakage_simulation"]["cohort_samples"] == {
        "multi_hospital": 2,
        "france": 1,
        "SPH_test": 3,
    }
    assert plan["evaluation"]["external_selection"] == "per_cohort_best_auc"
    assert len({job["split_directory"] for job in plan["jobs"]}) == 2
    for job in plan["jobs"]:
        assert job["split_directory"].startswith(job["run_directory"])
        assert f"split.seed={job['split_seed']}" in command_overrides(job["split_command"])
        assert any("fold_{fold}_train.txt" in value for value in job["training_command"])
        assert "seed=17" not in command_overrides(job["training_command"])
        assert job["training_command"][job["training_command"].index("--config") + 1].endswith(
            str(Path("configs/training.yaml"))
        )
    snapshot = plan["config_snapshots"]["training"]
    assert Path(snapshot["data"]["external"]["france"]["manifest"]) == (
        profiles.parent / "original_train.txt"
    )
    assert not Path(plan["directory"]).exists()


@pytest.mark.parametrize(
    "extra",
    [
        ["--seeds", "1", "1"],
        ["--folds", "1"],
        ["--parallel", "2"],
        ["--gpus", "-1"],
        ["--batch-name", "../escape"],
        ["--set", "output.run_name=shared"],
        ["--set", "data.train={}"],
        ["--set", "device=cpu"],
        ["--split-set", "split.seed=42"],
        ["--split-set", "split.mode=holdout"],
    ],
)
def test_unsafe_or_conflicting_options_fail_before_writing(profiles, extra):
    with pytest.raises(ConfigError):
        sweep.build_plan(arguments(profiles, *extra))
    assert not (profiles.parent / "runs").exists()


def test_repeated_holdout_and_explicit_gpu_sharing_are_supported(profiles):
    plan = sweep.build_plan(
        arguments(
            profiles,
            "--seeds",
            "77",
            "--gpus",
            "3",
            "3",
            "--split-set",
            "split.mode=repeated_holdout",
            "--split-set",
            "split.validation_fraction=0.3",
        )
    )
    assert plan["split_mode"] == "repeated_holdout"
    assert plan["parallel"] == 2
    assert "split.repeats=5" in command_overrides(plan["jobs"][0]["split_command"])


def test_dry_run_writes_nothing_and_does_not_spawn(profiles, monkeypatch, capsys):
    monkeypatch.setattr(sweep.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("spawned"))
    args = arguments(profiles, "--seeds", "1")
    argv = ["--config", str(args.config), "--split-config", str(args.split_config), "--dry-run"]
    assert sweep.main(argv) == 0
    payload = json.loads(capsys.readouterr().out)
    assert len(payload["jobs"]) == 5
    assert not Path(payload["directory"]).exists()


def test_parallel_workers_have_isolated_environments_and_split_before_training(
    profiles, monkeypatch
):
    plan = sweep.build_plan(
        arguments(profiles, "--seeds", "11", "12", "13", "14", "--gpus", "3", "5")
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "9")
    lock = threading.Lock()
    barrier = threading.Barrier(2)
    calls, active_gpus = [], set()
    max_active = 0

    class FakeProcess:
        def __init__(self, command, **kwargs):
            nonlocal max_active
            self.gpu = kwargs["env"]["CUDA_VISIBLE_DEVICES"]
            self.phase = command[3]
            self.returncode = None
            with lock:
                self.pid = 10000 + len(calls)
                calls.append((self.phase, self.gpu, command))
                assert self.gpu not in active_gpus
                active_gpus.add(self.gpu)
                max_active = max(max_active, len(active_gpus))
            kwargs["stdout"].write("simulated child output\n")

        def wait(self, timeout=None):
            if self.phase == "split" and self.pid < 10002:
                barrier.wait(timeout=10)
            threading.Event().wait(0.01)
            with lock:
                active_gpus.discard(self.gpu)
            self.returncode = 0
            return 0

    monkeypatch.setattr(sweep.subprocess, "Popen", FakeProcess)
    assert sweep.BatchRunner(plan).run() == 0
    assert max_active == 2
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "9"
    summary = json.loads((Path(plan["directory"]) / "batch_summary.json").read_text())
    assert summary["status"] == "completed"
    assert len(calls) == 8
    for job in summary["jobs"]:
        assert job["status"] == "completed"
        for phase, expected in (("split", "split"), ("training", "crossval")):
            matches = [call for call in calls if call[2] == job[f"{phase}_command"]]
            assert len(matches) == 1 and matches[0][:2] == (expected, job["gpu"])
            assert (Path(job["run_directory"]) / f"{phase}.log").is_file()
    with pytest.raises(FileExistsError):
        sweep.BatchRunner(plan).run()


def test_failed_split_does_not_train_and_other_groups_continue(profiles, monkeypatch):
    plan = sweep.build_plan(arguments(profiles, "--seeds", "1", "2"))
    commands = []

    class FakeProcess:
        def __init__(self, command, **kwargs):
            self.pid = 10000 + len(commands)
            commands.append(command)
            self.code = 7 if "split.seed=1" in command else 0

        def wait(self, timeout=None):
            return self.code

    monkeypatch.setattr(sweep.subprocess, "Popen", FakeProcess)
    assert sweep.BatchRunner(plan).run() == 1
    summary = json.loads((Path(plan["directory"]) / "batch_summary.json").read_text())
    assert summary["status"] == "failed"
    assert [job["status"] for job in summary["jobs"]] == ["split_failed", "completed"]
    assert len(commands) == 3
    assert not (Path(plan["jobs"][0]["run_directory"]) / "training.log").exists()


def test_stop_targets_only_registered_batch_processes(profiles, monkeypatch):
    runner = sweep.BatchRunner(sweep.build_plan(arguments(profiles, "--seeds", "1")))
    process = SimpleNamespace(pid=12345)
    runner.active[process.pid] = process
    stopped = []
    monkeypatch.setattr(sweep, "_terminate_process", stopped.append)
    runner.stop()
    assert runner.cancel.is_set()
    assert stopped == [process]


def test_training_failure_is_recorded_and_does_not_stop_other_seeds(profiles, monkeypatch):
    plan = sweep.build_plan(arguments(profiles, "--seeds", "1", "2"))

    class FakeProcess:
        pid = 12345

        def __init__(self, command, **kwargs):
            self.code = (
                9
                if command[3] == "crossval"
                and any('pcfnet_crossval_seed_1"' in value for value in command)
                else 0
            )

        def wait(self, timeout=None):
            return self.code

    monkeypatch.setattr(sweep.subprocess, "Popen", FakeProcess)
    assert sweep.BatchRunner(plan).run() == 1
    report = json.loads((Path(plan["directory"]) / "batch_summary.json").read_text())
    assert [job["status"] for job in report["jobs"]] == ["training_failed", "completed"]
    assert report["jobs"][0]["training_return_code"] == 9


def test_keyboard_interrupt_marks_batch_and_queued_work(profiles, monkeypatch):
    plan = sweep.build_plan(arguments(profiles, "--seeds", "1", "2", "3"))
    stopped = threading.Event()

    class FakeProcess:
        pid = 12345

        def __init__(self, *args, **kwargs):
            pass

        def wait(self, timeout=None):
            assert stopped.wait(timeout=5)
            return 0

    def interrupt(futures):
        raise KeyboardInterrupt

    monkeypatch.setattr(sweep.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(sweep, "as_completed", interrupt)
    monkeypatch.setattr(sweep, "_terminate_process", lambda process: stopped.set())
    assert sweep.BatchRunner(plan).run() == 130
    report = json.loads((Path(plan["directory"]) / "batch_summary.json").read_text())
    assert report["status"] == "interrupted"
    assert all(job["status"] in {"cancelled", "completed"} for job in report["jobs"])
    assert any(job["status"] == "cancelled" for job in report["jobs"])


def test_real_split_subprocesses_feed_matching_manifests_to_mock_trainers(profiles, monkeypatch):
    pytest.importorskip("sklearn")
    plan = sweep.build_plan(arguments(profiles, "--seeds", "17", "29", "--gpus", "3", "5"))
    real_popen = sweep.subprocess.Popen
    originals = {path: path.read_bytes() for path in profiles.parent.glob("original_*.txt")}
    # Queued groups must use frozen effective configs, not later edits to the user's YAML.
    original_config = yaml.safe_load((profiles / "training.yaml").read_text())
    original_config["seed"] = 99
    (profiles / "training.yaml").write_text(yaml.safe_dump(original_config), encoding="utf-8")
    seen = []

    class Trainer:
        def __init__(self, command, **kwargs):
            self.pid = 20000 + len(seen)
            config = effective_config(command)
            assert config["seed"] == 42
            assert config["device"] == "cuda:0"
            assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] in {"3", "5"}
            for fold in range(1, 6):
                paths = [
                    Path(config["data"][split]["manifest"].replace("{fold}", str(fold)))
                    for split in ("train", "validation")
                ]
                ids = [set(path.read_text().splitlines()) for path in paths]
                assert not ids[0].intersection(ids[1])
                assert ids[0] | ids[1] == set(map(str, range(20)))
                assert len(ids[0]) == 16 and len(ids[1]) == 4
            seen.append(config)

        def wait(self, timeout=None):
            return 0

    def dispatch(command, **kwargs):
        return (
            real_popen(command, **kwargs) if command[3] == "split" else Trainer(command, **kwargs)
        )

    monkeypatch.setattr(sweep.subprocess, "Popen", dispatch)
    assert sweep.BatchRunner(plan).run() == 0
    assert len(seen) == 2
    assert all(path.read_bytes() == value for path, value in originals.items())
    manifests = []
    for job in plan["jobs"]:
        directory = Path(job["split_directory"])
        assert len(list(directory.glob("fold_*.txt"))) == 10
        manifests.append((directory / "fold_1_validation.txt").read_bytes())
    assert manifests[0] != manifests[1]
