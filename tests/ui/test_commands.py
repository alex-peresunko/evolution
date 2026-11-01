from types import SimpleNamespace

import pytest

pytest.importorskip("pygame")

from evolution import config  # noqa: E402
from evolution.app import bootstrap  # noqa: E402
from evolution.brain.network import NeuralNetwork  # noqa: E402
from evolution.species import HERBIVORE  # noqa: E402
from evolution.ui import commands  # noqa: E402


def _herbivore_data(generation):
    data = NeuralNetwork(config.BRAIN_TOPOLOGY).to_data()
    data["generation"] = generation
    return data


def test_reloading_one_species_keeps_the_deeper_generation():
    sim = bootstrap.start_simulation(None, world_bounds=(400, 300))
    sim.generation = 900
    app = SimpleNamespace(sim=sim)
    commands.reload_species(app, [HERBIVORE], {HERBIVORE: _herbivore_data(610)})
    assert sim.generation == 900
    commands.reload_species(app, [HERBIVORE], {HERBIVORE: _herbivore_data(950)})
    assert sim.generation == 950


def test_read_only_keys_cannot_write_saved_brains():
    writers = {commands.save_living_best, commands.toggle_background_mode}
    assert writers <= set(commands.KEY_COMMANDS.values())
    assert not writers & set(commands.READ_ONLY_KEY_COMMANDS.values())
    assert set(commands.READ_ONLY_KEY_COMMANDS.values()) == (
        set(commands.KEY_COMMANDS.values()) - writers)
