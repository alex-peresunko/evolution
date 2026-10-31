import json

from evolution import config
from evolution.brain_store import (read_saved_history, read_saved_score, save_brain_if_better,
                                   update_saved_history)
from evolution.entities import Creature
from evolution.neural_network import NeuralNetwork
from evolution.simulation import (_retire_creature, build_sim_state, restore_progress,
                                  save_hall_of_fame)

WORLD_BOUNDS = (config.WORLD_WIDTH, config.WORLD_HEIGHT)


def _save(tmp_path, score):
    return save_brain_if_better(NeuralNetwork([4, 3, 2]), "brain.json", generation=1,
                                genes=None, score=score, save_dir=str(tmp_path))


def test_writes_when_file_missing(tmp_path):
    assert _save(tmp_path, 1.0) is True
    assert read_saved_score("brain.json", save_dir=str(tmp_path)) == 1.0


def test_skips_when_saved_score_higher(tmp_path):
    _save(tmp_path, 5.0)
    before = (tmp_path / "brain.json").read_text()
    assert _save(tmp_path, 2.0) is False
    assert _save(tmp_path, 5.0) is False
    assert (tmp_path / "brain.json").read_text() == before


def test_overwrites_when_saved_score_lower(tmp_path):
    _save(tmp_path, 2.0)
    assert _save(tmp_path, 5.0) is True
    assert read_saved_score("brain.json", save_dir=str(tmp_path)) == 5.0


def test_writes_over_unscored_or_corrupt_file(tmp_path):
    (tmp_path / "brain.json").write_text(json.dumps({"weights": []}))
    assert _save(tmp_path, 0.5) is True
    (tmp_path / "brain.json").write_text("{not json")
    assert _save(tmp_path, 0.5) is True


def test_new_record_overwrites_saved_brain_with_a_higher_faded_score(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "saved_brains").mkdir()
    NeuralNetwork(config.BRAIN_TOPOLOGY).save("best_brain_herbivore.json", score=100.0)

    c = Creature(WORLD_BOUNDS)
    c.score = 5.0
    sim_state = build_sim_state(WORLD_BOUNDS, [c], [], [], [])
    sim_state["tick"] = config.AUTOSAVE_INTERVAL + 1
    _retire_creature(c, sim_state)

    assert sim_state["hall_of_fame"]["herbivore"]["score"] == 5.0
    assert read_saved_score("best_brain_herbivore.json") == 5.0


def test_exit_save_keeps_the_file_without_a_new_record(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "saved_brains").mkdir()
    brain = NeuralNetwork(config.BRAIN_TOPOLOGY)
    brain.save("best_brain_herbivore.json", score=100.0)
    sim_state = build_sim_state(WORLD_BOUNDS, [], [], [], [])
    restore_progress(sim_state, "herbivore", brain, {}, "best_brain_herbivore.json")
    sim_state["hall_of_fame"]["herbivore"]["score"] = 40.0  # faded
    save_hall_of_fame(sim_state)
    assert read_saved_score("best_brain_herbivore.json") == 100.0


def test_old_topology_brain_loads_into_current_topology(tmp_path):
    import numpy as np

    old = NeuralNetwork([18, 16, 8, 2])
    old.save("old.json", save_dir=str(tmp_path))
    nn, _, _ = NeuralNetwork.load("old.json", config.BRAIN_TOPOLOGY, save_dir=str(tmp_path))
    assert nn.layer_sizes == config.BRAIN_TOPOLOGY
    np.testing.assert_array_equal(nn.weights[0][:18, :16], old.weights[0])
    np.testing.assert_array_equal(nn.weights[-1][:8, :2], old.weights[-1])
    np.testing.assert_array_equal(nn.biases[-1][:, :2], old.biases[-1])


def test_history_round_trip_and_update(tmp_path):
    history = {"score": [1.0, 2.0], "longevity": [10, 20]}
    save_brain_if_better(NeuralNetwork([4, 3, 2]), "brain.json", generation=3, genes=None,
                         score=2.0, save_dir=str(tmp_path), history=history)
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
    save_brain_if_better(brain, "best_brain_herbivore.json", generation=9, genes={"size": 5.0},
                         score=7.0, save_dir=str(tmp_path), history=history)
    sim_state = build_sim_state(WORLD_BOUNDS, [], [], [], [])
    sim_state["carnivore_score_history"] = [0.5]
    sim_state["carnivore_longevity_history"] = [50]
    restore_progress(sim_state, "herbivore", brain, {"size": 5.0}, "best_brain_herbivore.json",
                     save_dir=str(tmp_path))
    assert sim_state["hall_of_fame"]["herbivore"]["score"] == 7.0
    assert sim_state["herbivore_score_history"] == [1.0, 4.0, 7.0]
    assert sim_state["max_age_herbivore"] == 300
    assert sim_state["carnivore_score_history"] == [0.5, 0.5, 0.5]
    assert sim_state["carnivore_longevity_history"] == [50, 50, 50]
