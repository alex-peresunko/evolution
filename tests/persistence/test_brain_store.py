import json

import numpy as np

from evolution import config
from evolution.app.bootstrap import restore_progress
from evolution.brain.network import NeuralNetwork
from evolution.persistence.autosave import HallOfFameAutosaver, save_on_exit
from evolution.persistence.brain_store import (
    load_brain,
    read_saved_history,
    read_saved_score,
    save_brain,
    update_saved_history,
)
from evolution.sim.records import Record
from evolution.sim.simulation import Simulation
from evolution.species import CARNIVORE, HERBIVORE
from evolution.world.creature import Creature
from evolution.world.world import World

WORLD_BOUNDS = (config.WORLD_WIDTH, config.WORLD_HEIGHT)


def _sim(herbivores=(), listeners=()):
    return Simulation(World(WORLD_BOUNDS, {HERBIVORE: list(herbivores)}, []),
                      listeners=listeners)


def test_new_record_overwrites_saved_brain_with_a_higher_faded_score(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "saved_brains").mkdir()
    save_brain(NeuralNetwork(config.BRAIN_TOPOLOGY), "best_brain_herbivore.json", score=100.0)

    c = Creature(HERBIVORE, WORLD_BOUNDS)
    c.score = 5.0
    autosaver = HallOfFameAutosaver()
    sim = _sim([c], [autosaver])
    sim.world.tick = config.AUTOSAVE_INTERVAL + 1
    assert sim.hall_of_fame[HERBIVORE].retire(c)
    autosaver.record_set(sim, HERBIVORE)

    assert sim.hall_of_fame[HERBIVORE].best.score == 5.0
    assert read_saved_score("best_brain_herbivore.json") == 5.0


def test_exit_save_keeps_the_file_without_a_new_record(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "saved_brains").mkdir()
    brain = NeuralNetwork(config.BRAIN_TOPOLOGY)
    save_brain(brain, "best_brain_herbivore.json", score=100.0)
    sim = _sim()
    restore_progress(sim, HERBIVORE, Record(None, brain, {}, 1))
    sim.hall_of_fame[HERBIVORE].best.score = 40.0  # faded
    save_on_exit(sim)
    assert read_saved_score("best_brain_herbivore.json") == 100.0


def test_old_topology_brain_loads_into_current_topology(tmp_path):
    old = NeuralNetwork([18, 16, 8, 2])
    save_brain(old, "old.json", save_dir=str(tmp_path))
    nn, _, _ = load_brain("old.json", config.BRAIN_TOPOLOGY, save_dir=str(tmp_path))
    assert nn.layer_sizes == config.BRAIN_TOPOLOGY
    np.testing.assert_array_equal(nn.weights[0][:18, :16], old.weights[0])
    np.testing.assert_array_equal(nn.weights[-1][:8, :2], old.weights[-1])
    np.testing.assert_array_equal(nn.biases[-1][:, :2], old.biases[-1])


def test_unscored_or_corrupt_file_reads_no_score(tmp_path):
    (tmp_path / "brain.json").write_text(json.dumps({"weights": []}))
    assert read_saved_score("brain.json", save_dir=str(tmp_path)) is None
    (tmp_path / "brain.json").write_text("{not json")
    assert read_saved_score("brain.json", save_dir=str(tmp_path)) is None
    assert read_saved_score("missing.json", save_dir=str(tmp_path)) is None


def test_history_round_trip_and_update(tmp_path):
    history = {"score": [1.0, 2.0], "longevity": [10, 20]}
    save_brain(NeuralNetwork([4, 3, 2]), "brain.json", generation=3, score=2.0,
               save_dir=str(tmp_path), history=history)
    assert read_saved_history("brain.json", save_dir=str(tmp_path)) == history
    before = json.loads((tmp_path / "brain.json").read_text())
    newer = {"score": [1.0, 2.0, 2.0], "longevity": [10, 20, 30]}
    assert update_saved_history("brain.json", newer, save_dir=str(tmp_path)) is True
    after = json.loads((tmp_path / "brain.json").read_text())
    assert after.pop("history") == newer
    before.pop("history")
    assert after == before
    assert update_saved_history("missing.json", newer, save_dir=str(tmp_path)) is False


def test_restore_progress_seeds_hof_and_pads_histories(tmp_path):
    brain = NeuralNetwork(config.BRAIN_TOPOLOGY)
    history = {"score": [1.0, 4.0, 7.0], "longevity": [100, 200, 300]}
    save_brain(brain, "best_brain_herbivore.json", generation=9, genes={"size": 5.0},
               score=7.0, save_dir=str(tmp_path), history=history)
    sim = _sim()
    sim.stats[CARNIVORE].score_history = [0.5]
    sim.stats[CARNIVORE].longevity_history = [50]
    restore_progress(sim, HERBIVORE, Record(None, brain, {"size": 5.0}, 9),
                     source=str(tmp_path))
    assert sim.hall_of_fame[HERBIVORE].best.score == 7.0
    assert sim.stats[HERBIVORE].score_history == [1.0, 4.0, 7.0]
    assert sim.stats[HERBIVORE].max_age == 300
    assert sim.stats[CARNIVORE].score_history == [0.5, 0.5, 0.5]
    assert sim.stats[CARNIVORE].longevity_history == [50, 50, 50]
