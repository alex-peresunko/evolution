import os

from evolution import config
from evolution.brain.network import NeuralNetwork
from evolution.persistence.autosave import HallOfFameAutosaver
from evolution.persistence.brain_store import read_saved_score, save_brain
from evolution.persistence.islands import (
    IslandExchange,
    clear_island_files,
    main_file,
    worker_file,
)
from evolution.sim.records import Record
from evolution.sim.simulation import Simulation
from evolution.species import HERBIVORE
from evolution.world.creature import Creature
from evolution.world.world import World

WORLD_BOUNDS = (config.WORLD_WIDTH, config.WORLD_HEIGHT)


def _sim(n_herbs=10, listeners=()):
    herbs = [Creature(HERBIVORE, WORLD_BOUNDS) for _ in range(n_herbs)]
    for i, c in enumerate(herbs):
        c.score = float(i)
    return Simulation(World(WORLD_BOUNDS, {HERBIVORE: herbs}, []), listeners=listeners)


def _write(tmp_path, filename, score):
    brain = NeuralNetwork(config.BRAIN_TOPOLOGY)
    save_brain(brain, filename, save_dir=str(tmp_path), score=score, genes={"size": 7.0})
    return brain


def test_save_is_atomic(tmp_path):
    _write(tmp_path, "b.json", 1.0)
    assert os.listdir(tmp_path) == ["b.json"]


def test_import_better_brain_injects_immigrants(tmp_path):
    sim = _sim()
    sim.hall_of_fame[HERBIVORE].best.score = 5.0
    weakest = set(map(id, sorted(sim.world.herbivores, key=lambda c: c.score)[:4]))
    brain = _write(tmp_path, worker_file(0, HERBIVORE), 9.0)
    exchange = IslandExchange(sim, save_dir=str(tmp_path))
    assert exchange.import_if_better(HERBIVORE, worker_file(0, HERBIVORE)) == (5.0, 9.0)
    best = sim.hall_of_fame[HERBIVORE].best
    assert best.score == 9.0 and best.genes["size"] == 7.0
    assert len(sim.world.herbivores) == 10
    assert not weakest & set(map(id, sim.world.herbivores))
    immigrants = sim.world.herbivores[-config.ISLAND_IMMIGRANTS:]
    assert (immigrants[0].brain.weights[0] == brain.weights[0]).all()
    assert all(c.respawn_immunity_ticks > 0 for c in immigrants)
    # Unchanged file: not imported again.
    assert exchange.import_if_better(HERBIVORE, worker_file(0, HERBIVORE)) is None


def test_worse_or_missing_brain_is_ignored(tmp_path):
    sim = _sim()
    sim.hall_of_fame[HERBIVORE].best.score = 5.0
    before = list(sim.world.herbivores)
    _write(tmp_path, worker_file(0, HERBIVORE), 3.0)
    exchange = IslandExchange(sim, save_dir=str(tmp_path))
    assert exchange.import_if_better(HERBIVORE, worker_file(0, HERBIVORE)) is None
    assert exchange.import_if_better(HERBIVORE, worker_file(1, HERBIVORE)) is None
    assert sim.world.herbivores == before and sim.hall_of_fame[HERBIVORE].best.score == 5.0


def test_publish_only_when_improved(tmp_path):
    sim = _sim()
    exchange = IslandExchange(sim, save_dir=str(tmp_path))
    exchange.publish()
    assert os.listdir(tmp_path) == []  # empty Hall of Fame
    sim.hall_of_fame[HERBIVORE].best = Record(4.0, NeuralNetwork(config.BRAIN_TOPOLOGY),
                                              {"size": 7.0})
    exchange.publish()
    path = tmp_path / main_file(HERBIVORE)
    assert read_saved_score(main_file(HERBIVORE), save_dir=str(tmp_path)) == 4.0
    mtime = path.stat().st_mtime_ns
    exchange.publish()
    assert path.stat().st_mtime_ns == mtime


def test_clear_island_files(tmp_path):
    for name in ("worker_0_herbivore.json", "worker_main_carnivore.json",
                 "worker_1_herbivore.json.123.tmp", "worker_0.log", "best_brain_herbivore.json"):
        (tmp_path / name).write_text("{}")
    clear_island_files(save_dir=str(tmp_path))
    assert sorted(os.listdir(tmp_path)) == ["best_brain_herbivore.json", "worker_0.log"]


def test_record_inside_autosave_interval_is_saved_later(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    autosaver = HallOfFameAutosaver(worker_id=0)
    sim = _sim(listeners=[autosaver])
    sim.world.tick = config.AUTOSAVE_INTERVAL
    first, second = sim.world.herbivores[-2:]
    sim.retire(first)
    assert read_saved_score(worker_file(0, HERBIVORE)) == first.score
    sim.world.tick += 1
    sim.retire(second)  # inside the interval: deferred, not dropped
    assert read_saved_score(worker_file(0, HERBIVORE)) == first.score
    sim.world.tick += config.AUTOSAVE_INTERVAL
    autosaver.save_due(sim)
    assert read_saved_score(worker_file(0, HERBIVORE)) == second.score


def test_forced_autosave_ignores_interval(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    autosaver = HallOfFameAutosaver(worker_id=0)
    sim = _sim(listeners=[autosaver])
    sim.world.tick = config.AUTOSAVE_INTERVAL
    sim.retire(sim.world.herbivores[-2])
    sim.retire(sim.world.herbivores[-1])
    autosaver.save_due(sim, force=True)
    assert read_saved_score(worker_file(0, HERBIVORE)) == sim.world.herbivores[-1].score
