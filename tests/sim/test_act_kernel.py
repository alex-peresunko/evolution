"""The compiled act loop (sim.act_kernel) must leave the world exactly as the Python loop
(Simulation._act_in_python) does: same values, same types, same random stream."""
import contextlib
import io
import random

import numpy as np
import pytest

from evolution.app.bootstrap import start_simulation
from evolution.sim.simulation import Simulation
from evolution.world.creature import Creature
from evolution.world.food import Carcass, Food


def _state(sim):
    world = sim.world
    creatures = [
        (c.id, c.pos.x, c.pos.y, c.angle, c.turn, c.health, c.fat, c.stamina, c.score, c.size,
         c.current_speed, c.hurt, c.age, c.thirst, c.hunger, c.eating_ticks, c.meal_health,
         c.stunned_ticks, c.is_adult, c.winded, c.in_water, c.willing, c.mate_override,
         c.last_bite_tick, c.last_hit_tick, c.last_trample_tick, c.last_trampled_tick,
         None if c.meal_carcass is None else c.meal_carcass.id,
         sorted((o.id, t) for o, t in c.recent_biters.items()),
         sorted((o.id, t) for o, t in c.recent_tramplers.items()))
        for c in world.creatures()]
    return (repr(creatures), [(f.id, f.consumed) for f in world.food],
            [(k.id, k.meat) for k in world.carcasses], repr(world.tally), random.random())


def _run(compiled, seed, ticks, monkeypatch):
    monkeypatch.setattr(Simulation, "compiled_act", compiled)
    random.seed(seed)
    np.random.seed(seed)
    monkeypatch.setattr(Creature, "_id_counter", 1)
    monkeypatch.setattr(Food, "_id_counter", -1)
    monkeypatch.setattr(Carcass, "_id_counter", -10**9)
    with contextlib.redirect_stdout(io.StringIO()):
        sim = start_simulation(None)
        states = []
        for _ in range(ticks):
            sim.step()
            states.append(_state(sim))
    return states


@pytest.mark.parametrize("seed", [3, 11])
def test_compiled_act_loop_matches_the_python_loop(seed, monkeypatch):
    python = _run(False, seed, 120, monkeypatch)
    compiled = _run(True, seed, 120, monkeypatch)
    for tick, (a, b) in enumerate(zip(python, compiled), 1):
        assert a == b, f"tick {tick}"
