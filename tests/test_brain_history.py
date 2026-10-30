import json
import os

import pytest

from evolution import brain_history as bh
from evolution import config
from evolution.neural_network import NeuralNetwork
from evolution.simulation import create_start_populations

WORLD_BOUNDS = (config.WORLD_WIDTH, config.WORLD_HEIGHT)


def _write_best(save_dir, score, species=bh.SPECIES):
    for sp in species:
        NeuralNetwork(config.BRAIN_TOPOLOGY).save(bh.best_file(sp), save_dir=str(save_dir),
                                                  generation=3, score=score)


def _snapshot(save_dir, score, name):
    path = save_dir / "history" / name
    path.mkdir(parents=True)
    _write_best(path, score)
    return path


def test_snapshot_copies_both_files_and_skips_duplicates(tmp_path):
    _write_best(tmp_path, 1.0)
    path = bh.snapshot_best(str(tmp_path))
    assert sorted(os.listdir(path)) == sorted(bh.best_file(sp) for sp in bh.SPECIES)
    assert os.path.exists(tmp_path / bh.best_file("herbivore"))
    assert bh.snapshot_best(str(tmp_path)) is None
    assert len(bh.snapshot_names(str(tmp_path))) == 1


def test_snapshot_move_removes_best_files_even_when_duplicate(tmp_path):
    _write_best(tmp_path, 1.0)
    bh.snapshot_best(str(tmp_path))
    assert bh.snapshot_best(str(tmp_path), move=True) is None
    assert not any(os.path.exists(tmp_path / bh.best_file(sp)) for sp in bh.SPECIES)
    assert len(bh.snapshot_names(str(tmp_path))) == 1


def test_snapshot_without_best_files(tmp_path):
    assert bh.snapshot_best(str(tmp_path), move=True) is None


def test_prune_keeps_newest_and_protected(tmp_path):
    for i in range(5):
        _snapshot(tmp_path, i, f"2026010{i}-000000")
    oldest = str(tmp_path / "history" / "20260100-000000")
    bh.prune(keep=2, save_dir=str(tmp_path), protect=oldest)
    assert bh.snapshot_names(str(tmp_path)) == [
        "20260104-000000", "20260103-000000", "20260100-000000"]


def test_resolve_by_index_and_name(tmp_path):
    _snapshot(tmp_path, 1.0, "20260101-000000")
    _snapshot(tmp_path, 2.0, "20260102-000000-pretrained")
    newest = str(tmp_path / "history" / "20260102-000000-pretrained")
    assert bh.resolve("1", str(tmp_path)) == newest
    assert bh.resolve("20260101-000000", str(tmp_path)).endswith("20260101-000000")
    for bad in ("0", "3", "nope"):
        with pytest.raises(ValueError):
            bh.resolve(bad, str(tmp_path))


def test_listing_shows_scores(tmp_path):
    _snapshot(tmp_path, 4.5, "20260101-000000")
    [(index, name, info)] = bh.list_snapshots(str(tmp_path))
    assert (index, name, info["carnivore"]) == (1, "20260101-000000", (4.5, 3))
    assert "score 4.5 gen 3" in bh.format_listing(str(tmp_path))


def test_new_run_moves_best_brains_to_history(tmp_path):
    _write_best(tmp_path, 7.0)
    source = bh.start_source(new=True, save_dir=str(tmp_path))
    assert source is None
    snapshot = bh.archive_after_start(source, new=True, save_dir=str(tmp_path))
    with open(os.path.join(snapshot, bh.best_file("herbivore"))) as f:
        assert json.load(f)["score"] == 7.0
    assert not os.path.exists(tmp_path / bh.best_file("herbivore"))


def test_continue_run_keeps_best_brains_and_copies_them(tmp_path):
    _write_best(tmp_path, 7.0)
    source = bh.start_source(save_dir=str(tmp_path))
    assert source == str(tmp_path)
    assert bh.archive_after_start(source, save_dir=str(tmp_path)) is not None
    assert os.path.exists(tmp_path / bh.best_file("herbivore"))


def test_loading_the_oldest_snapshot_survives_pruning(tmp_path):
    for i in range(3):
        _snapshot(tmp_path, i, f"2026010{i}-000000")
    _write_best(tmp_path, 9.0)
    source = bh.start_source(load="3", save_dir=str(tmp_path))
    herbs, carns, generation = create_start_populations(WORLD_BOUNDS, source)
    assert len(herbs) == config.NUM_HERBIVOROUS and generation == 3
    bh.archive_after_start(source, save_dir=str(tmp_path), keep=3)
    assert os.path.isdir(source)
    assert len(bh.snapshot_names(str(tmp_path))) == 4  # newest 3 plus the loaded one
    assert not os.path.exists(tmp_path / bh.best_file("carnivore"))


def test_start_populations_are_random_without_source():
    herbs, carns, generation = create_start_populations(WORLD_BOUNDS, None)
    assert generation == 1
    assert len(carns) == config.NUM_CARNIVORES


def test_chart_history_changes_do_not_create_a_new_snapshot(tmp_path):
    from evolution.brain_store import update_saved_history
    _write_best(tmp_path, 1.0)
    bh.snapshot_best(str(tmp_path))
    update_saved_history(bh.best_file("herbivore"), {"score": [1.0]}, save_dir=str(tmp_path))
    assert bh.snapshot_best(str(tmp_path)) is None
    _write_best(tmp_path, 2.0, species=("carnivore",))
    assert bh.snapshot_best(str(tmp_path)) is not None
