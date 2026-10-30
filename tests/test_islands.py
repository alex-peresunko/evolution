import os

from evolution import config
from evolution.brain_store import read_saved_score
from evolution.entities import Creature
from evolution.islands import (
    clear_island_files,
    import_if_better,
    main_file,
    publish_hall_of_fame,
    worker_file,
)
from evolution.neural_network import NeuralNetwork
from evolution.simulation import _retire_creature, autosave_hall_of_fame, build_sim_state

WORLD_BOUNDS = (config.WORLD_WIDTH, config.WORLD_HEIGHT)


def _sim(n_herbs=10):
    herbs = [Creature(WORLD_BOUNDS) for _ in range(n_herbs)]
    for i, c in enumerate(herbs):
        c.score = float(i)
    return build_sim_state(WORLD_BOUNDS, herbs, [], [], [])


def _write(tmp_path, filename, score):
    brain = NeuralNetwork(config.BRAIN_TOPOLOGY)
    brain.save(filename, save_dir=str(tmp_path), score=score, genes={"size": 7.0})
    return brain


def test_save_is_atomic(tmp_path):
    _write(tmp_path, "b.json", 1.0)
    assert os.listdir(tmp_path) == ["b.json"]


def test_import_better_brain_injects_immigrants(tmp_path):
    sim = _sim()
    sim["hall_of_fame"]["herbivore"]["score"] = 5.0
    weakest = set(map(id, sorted(sim["creatures"], key=lambda c: c.score)[:4]))
    brain = _write(tmp_path, worker_file(0, "herbivore"), 9.0)
    seen = {}
    assert import_if_better(sim, "herbivore", worker_file(0, "herbivore"), seen,
                            save_dir=str(tmp_path)) == (5.0, 9.0)
    hof = sim["hall_of_fame"]["herbivore"]
    assert hof["score"] == 9.0 and hof["genes"]["size"] == 7.0
    assert len(sim["creatures"]) == 10
    assert not weakest & set(map(id, sim["creatures"]))
    immigrants = sim["creatures"][-config.ISLAND_IMMIGRANTS:]
    assert (immigrants[0].brain.weights[0] == brain.weights[0]).all()
    assert all(c.respawn_immunity_ticks > 0 for c in immigrants)
    # Unchanged file: not imported again.
    assert import_if_better(sim, "herbivore", worker_file(0, "herbivore"), seen,
                            save_dir=str(tmp_path)) is None


def test_worse_or_missing_brain_is_ignored(tmp_path):
    sim = _sim()
    sim["hall_of_fame"]["herbivore"]["score"] = 5.0
    before = list(sim["creatures"])
    _write(tmp_path, worker_file(0, "herbivore"), 3.0)
    assert import_if_better(sim, "herbivore", worker_file(0, "herbivore"), {},
                            save_dir=str(tmp_path)) is None
    assert import_if_better(sim, "herbivore", worker_file(1, "herbivore"), {},
                            save_dir=str(tmp_path)) is None
    assert sim["creatures"] == before and sim["hall_of_fame"]["herbivore"]["score"] == 5.0


def test_publish_only_when_improved(tmp_path):
    sim = _sim()
    published = {}
    publish_hall_of_fame(sim, published, save_dir=str(tmp_path))
    assert os.listdir(tmp_path) == []  # empty Hall of Fame
    sim["hall_of_fame"]["herbivore"] = {"score": 4.0, "brain": NeuralNetwork(config.BRAIN_TOPOLOGY),
                                        "genes": {"size": 7.0}}
    publish_hall_of_fame(sim, published, save_dir=str(tmp_path))
    path = tmp_path / main_file("herbivore")
    assert read_saved_score(main_file("herbivore"), save_dir=str(tmp_path)) == 4.0
    mtime = path.stat().st_mtime_ns
    publish_hall_of_fame(sim, published, save_dir=str(tmp_path))
    assert path.stat().st_mtime_ns == mtime


def test_clear_island_files(tmp_path):
    for name in ("worker_0_herbivore.json", "worker_main_carnivore.json",
                 "worker_1_herbivore.json.123.tmp", "worker_0.log", "best_brain_herbivore.json"):
        (tmp_path / name).write_text("{}")
    clear_island_files(save_dir=str(tmp_path))
    assert sorted(os.listdir(tmp_path)) == ["best_brain_herbivore.json", "worker_0.log"]


def test_record_inside_autosave_interval_is_saved_later(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    sim = _sim()
    sim["tick"] = config.AUTOSAVE_INTERVAL
    first, second = sim["creatures"][-2:]
    _retire_creature(first, sim, worker_id=0)
    assert read_saved_score(worker_file(0, "herbivore")) == first.score
    sim["tick"] += 1
    _retire_creature(second, sim, worker_id=0)  # inside the interval: deferred, not dropped
    assert read_saved_score(worker_file(0, "herbivore")) == first.score
    sim["tick"] += config.AUTOSAVE_INTERVAL
    autosave_hall_of_fame(sim, worker_id=0)
    assert read_saved_score(worker_file(0, "herbivore")) == second.score


def test_forced_autosave_ignores_interval(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    sim = _sim()
    sim["tick"] = config.AUTOSAVE_INTERVAL
    _retire_creature(sim["creatures"][-2], sim, worker_id=0)
    _retire_creature(sim["creatures"][-1], sim, worker_id=0)
    autosave_hall_of_fame(sim, worker_id=0, force=True)
    assert read_saved_score(worker_file(0, "herbivore")) == sim["creatures"][-1].score
