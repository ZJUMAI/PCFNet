"""Leakage simulation tests use synthetic data only."""

import copy
import logging
import random

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchio")

from torch.utils.data import DataLoader, Dataset, RandomSampler

import petct.leakage as leakage
from petct.config import ConfigError, leakage_simulation_settings


class IndexedDataset(Dataset):
    def __init__(self, start, size):
        self.start = start
        self.size = size
        self.identifiers = [str(start + index) for index in range(size)]

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return self.start + index


def _config():
    source = dict(manifest="cases.txt", metadata="labels.csv", ct_root="ct", pet_root="pet")
    return {
        "seed": 42,
        "data": {
            "external": {
                "pooled": {
                    "members": {
                        "a": dict(source, start=100, size=10),
                        "b": dict(source, start=110, size=10),
                    }
                },
                "single": dict(source, start=200, size=20),
            }
        },
        "training": {
            "batch_size": 4,
            "num_workers": 0,
            "pin_memory": False,
            "leakage_simulation": {"enabled": True, "samples_per_cohort": 3},
        },
    }


def _factory(config, monkeypatch):
    built = []

    def build(root, source, *, training):
        assert training is True
        dataset = IndexedDataset(source["start"], source["size"])
        built.append(dataset)
        return dataset

    monkeypatch.setattr(leakage, "dataset_from_config", build)
    train = DataLoader(IndexedDataset(0, 10), batch_size=4)
    factory = leakage.LeakageEpochLoader(
        config, train, leakage_simulation_settings(config), logging.getLogger("test.leakage")
    )
    return factory, train, built


def _draws(loader):
    return [list(subset.indices) for subset in loader.dataset.datasets[1:]]


def test_leakage_redraws_without_accumulation_and_keeps_original_data(monkeypatch, caplog):
    config = _config()
    original = copy.deepcopy(config)
    with caplog.at_level(logging.INFO):
        factory, train, built = _factory(config, monkeypatch)
        loaders = [factory(epoch) for epoch in range(1, 5)]
    assert len(built) == 3  # Two pooled members and one independent cohort.
    assert len(factory.sources["pooled"]) == 20
    assert len(factory.sources["single"]) == 20
    assert len(train.dataset) == 10
    assert config == original
    for loader in loaders:
        assert isinstance(loader.sampler, RandomSampler)
        assert loader.drop_last is False
        assert len(loader.dataset) == 16  # 10 + 3 pooled + 3 single; never 10 + 3 per member.
        values = torch.cat(list(loader)).tolist()
        assert len(values) == len(set(values)) == 16
        assert sorted(value for value in values if value < 100) == list(range(10))
        assert sum(100 <= value < 120 for value in values) == 3
        assert sum(200 <= value < 220 for value in values) == 3
        assert all(len(indices) == len(set(indices)) == 3 for indices in _draws(loader))
    assert len({tuple(map(tuple, _draws(loader))) for loader in loaders}) == 4
    assert "LEAKAGE SIMULATION ENABLED" in caplog.text
    assert "sampled_test_cases=6" in caplog.text
    assert "identifiers" not in caplog.text


def test_leakage_draws_and_shuffle_are_reproducible_and_rng_independent(monkeypatch):
    factory, _, _ = _factory(_config(), monkeypatch)
    first = factory(1)
    indices = _draws(first)
    values = torch.cat(list(first))
    random.seed(999)
    torch.manual_seed(999)
    repeated = factory(1)
    assert _draws(repeated) == indices
    assert torch.equal(torch.cat(list(repeated)), values)
    restarted, _, _ = _factory(_config(), monkeypatch)
    assert _draws(restarted(1)) == indices
    next_epoch = factory(2)
    rng = random.Random(44)
    assert _draws(next_epoch) == [rng.sample(range(20), 3), rng.sample(range(20), 3)]


def test_leakage_can_limit_source_cohorts(monkeypatch):
    config = _config()
    config["training"]["leakage_simulation"]["cohorts"] = ["single"]
    factory, _, built = _factory(config, monkeypatch)
    assert list(factory.sources) == ["single"]
    assert len(built) == 1
    assert len(factory(1).dataset) == 13


def test_leakage_rejects_oversized_draw_without_loading_images(monkeypatch):
    config = _config()
    config["training"]["leakage_simulation"]["samples_per_cohort"] = 21
    with pytest.raises(ConfigError, match="exceeds cohort.*sampling is without replacement"):
        _factory(config, monkeypatch)


def test_leakage_all_cases_can_recur_across_epochs(monkeypatch):
    config = _config()
    config["training"]["leakage_simulation"]["samples_per_cohort"] = 20
    factory, _, _ = _factory(config, monkeypatch)
    for epoch in (1, 2):
        loader = factory(epoch)
        assert len(loader.dataset) == 50
        assert all(set(indices) == set(range(20)) for indices in _draws(loader))


def test_leakage_epoch_subsets_reach_dataloader_workers(monkeypatch):
    config = _config()
    config["training"]["num_workers"] = 1
    factory, _, _ = _factory(config, monkeypatch)
    for epoch in (1, 2):
        loader = factory(epoch)
        expected = sorted(loader.dataset[index] for index in range(len(loader.dataset)))
        assert sorted(torch.cat(list(loader)).tolist()) == expected


def test_leakage_uses_independent_counts_and_uniform_fallback(monkeypatch, caplog):
    config = _config()
    config["training"]["leakage_simulation"]["cohort_samples"] = {"pooled": 5}
    with caplog.at_level(logging.INFO):
        factory, _, built = _factory(config, monkeypatch)
        loader = factory(1)
    assert len(built) == 3
    assert [len(subset) for subset in loader.dataset.datasets[1:]] == [5, 3]
    assert len(loader.dataset) == 18
    assert "sampled_test_cases=8" in caplog.text
    first = _draws(loader)
    assert _draws(factory(1)) == first
    assert _draws(factory(2)) != first


def test_zero_count_skips_building_that_cohorts_training_view(monkeypatch):
    config = _config()
    config["training"]["leakage_simulation"]["cohort_samples"] = {"pooled": 0, "single": 2}
    factory, train, built = _factory(config, monkeypatch)
    assert len(built) == 1
    assert list(factory.sources) == ["single"]
    assert len(train.dataset) == 10
    for epoch in (1, 2):
        loader = factory(epoch)
        values = torch.cat(list(loader)).tolist()
        assert len(values) == 12
        assert all(value < 100 or 200 <= value < 220 for value in values)
        assert [len(subset) for subset in loader.dataset.datasets[1:]] == [2]


def test_individual_count_is_checked_against_its_own_group(monkeypatch):
    config = _config()
    config["training"]["leakage_simulation"]["cohort_samples"] = {"single": 21}
    with pytest.raises(ConfigError, match="count \\(21\\) exceeds cohort 'single'"):
        _factory(config, monkeypatch)
