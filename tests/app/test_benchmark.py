import pytest

from evolution.app import benchmark
from evolution.species import ALL_SPECIES


def test_run_measures_every_species():
    result = benchmark.run(None, ticks=20, seed=1)
    assert result["world"]["ticks_per_second"] > 0
    for species in ALL_SPECIES:
        assert result[species.name]["population"] == pytest.approx(species.start_count, rel=0.5)


def test_average_and_report():
    results = [benchmark.run(None, ticks=5, seed=seed) for seed in (1, 2)]
    mean = benchmark.average(results)
    name = ALL_SPECIES[0].name
    assert mean[name]["population"] == pytest.approx(
        (results[0][name]["population"] + results[1][name]["population"]) / 2)
    assert "litters_per_life" in benchmark.format_report([("random", mean)])


def test_source_dir(tmp_path):
    assert benchmark.source_dir("random") is None
    assert benchmark.source_dir("best", tmp_path) == tmp_path
