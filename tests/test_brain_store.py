import json

from evolution import config
from evolution.brain_store import read_saved_score, save_brain_if_better
from evolution.entities import Creature
from evolution.neural_network import NeuralNetwork
from evolution.simulation import _retire_creature, build_sim_state

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


def test_retire_creature_keeps_better_saved_brain(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "saved_brains").mkdir()
    NeuralNetwork(config.BRAIN_TOPOLOGY).save("best_brain_herbivore.json", score=100.0)

    c = Creature(WORLD_BOUNDS)
    c.score = 5.0
    sim_state = build_sim_state(WORLD_BOUNDS, [c], [], [], [])
    sim_state["tick"] = config.AUTOSAVE_INTERVAL + 1
    _retire_creature(c, sim_state)

    assert sim_state["hall_of_fame"]["herbivore"]["score"] == 5.0
    assert read_saved_score("best_brain_herbivore.json") == 100.0
    assert sim_state["last_autosave_tick"] == 0


def test_old_topology_brain_loads_into_current_topology(tmp_path):
    import numpy as np

    old = NeuralNetwork([18, 16, 8, 2])
    old.save("old.json", save_dir=str(tmp_path))
    nn, _, _ = NeuralNetwork.load("old.json", config.BRAIN_TOPOLOGY, save_dir=str(tmp_path))
    assert nn.layer_sizes == config.BRAIN_TOPOLOGY
    np.testing.assert_array_equal(nn.weights[0][:18, :16], old.weights[0])
    np.testing.assert_array_equal(nn.weights[-1][:8, :2], old.weights[-1])
    np.testing.assert_array_equal(nn.biases[-1][:, :2], old.biases[-1])
