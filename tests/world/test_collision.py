import itertools
import random

import numpy as np
import pytest

from evolution import config
from evolution.app.bootstrap import build_world
from evolution.sim.populations import founders
from evolution.sim.simulation import Simulation
from evolution.species import ALL_SPECIES, CARNIVORE, HERBIVORE
from evolution.world import collision
from evolution.world.creature import Creature
from evolution.world.food import Carcass
from evolution.world.terrain import Obstacle
from evolution.world.world import World

WORLD_BOUNDS = (config.WORLD_WIDTH, config.WORLD_HEIGHT)


def _body(species, x, y, size=8):
    c = Creature(species, WORLD_BOUNDS)
    c.genetic_size = c.size = size
    c.pos = c.geometry.vec2(x, y)
    return c


def _gap(a, b):
    return a.pos.distance_to(b.pos) - (a.size + b.size) * config.BODY_RADIUS_FACTOR


def test_overlapping_creatures_end_touching():
    a, b = _body(HERBIVORE, 500, 400), _body(CARNIVORE, 510, 400)
    collision.separate([a, b], [], [])
    assert _gap(a, b) == pytest.approx(0, abs=1e-6)
    assert a.pos.y == b.pos.y == 400


def test_heavier_creature_moves_less():
    big, small = _body(HERBIVORE, 500, 400, size=16), _body(CARNIVORE, 510, 400, size=8)
    collision.separate([big, small], [], [])
    assert 500 - big.pos.x < small.pos.x - 510


def test_coincident_creatures_separate():
    a, b = _body(HERBIVORE, 500, 400), _body(HERBIVORE, 500, 400)
    collision.separate([a, b], [], [])
    assert _gap(a, b) == pytest.approx(0, abs=1e-6)


def test_carcass_is_immovable():
    dead = _body(HERBIVORE, 500, 400, size=10)
    carcass = Carcass(dead)
    c = _body(CARNIVORE, 505, 400, size=6)
    collision.separate([c], [carcass], [])
    assert (carcass.pos.x, carcass.pos.y) == (500, 400)
    assert c.pos.x == pytest.approx(500 + (10 + 6) * config.BODY_RADIUS_FACTOR)


def test_carnivore_against_a_carcass_scavenges_it():
    carcass = Carcass(_body(HERBIVORE, 500, 400, size=10))
    carn = _body(CARNIVORE, 516, 400, size=6)
    carn.is_adult, carn.respawn_immunity_ticks = True, 0
    world = World(WORLD_BOUNDS, {CARNIVORE: [carn]}, [])
    world.carcasses.append(carcass)
    world.reindex()
    carn.act([0.0, -1.0] + [0.0] * (config.BRAIN_OUTPUTS - 2), world)
    assert carcass.meat < carcass.max_meat


def test_dead_are_ignored():
    a, b = _body(HERBIVORE, 500, 400), _body(HERBIVORE, 505, 400)
    b.health = 0
    collision.separate([a, b], [], [])
    assert (a.pos.x, b.pos.x) == (500, 505)


def test_push_into_a_rock_is_reverted():
    rock = Obstacle(470, 380, 20, 40)
    a, b = _body(HERBIVORE, 500, 400), _body(HERBIVORE, 505, 400)
    collision.separate([a, b], [], [rock])
    assert (a.pos.x, a.pos.y) == (500, 400)
    assert b.pos.x > 505


def test_push_through_a_wall_is_clamped():
    a, b = _body(HERBIVORE, 8, 400), _body(HERBIVORE, 12, 400)
    collision.separate([a, b], [], [])
    assert a.pos.x >= a.size


def test_collisions_off(monkeypatch):
    monkeypatch.setattr(collision, "BODY_RADIUS_FACTOR", 0.0)
    a, b = _body(HERBIVORE, 500, 400), _body(HERBIVORE, 505, 400)
    collision.separate([a, b], [], [])
    assert (a.pos.x, b.pos.x) == (500, 505)


def test_simulation_keeps_bodies_apart():
    random.seed(3)
    np.random.seed(3)
    sim = Simulation(build_world({s: founders(s, WORLD_BOUNDS) for s in ALL_SPECIES}))
    for _ in range(200):
        sim.step()
    live = [c for c in sim.world.creatures() if c.health > 0 and c.age > 1]
    worst = min(_gap(a, b) for a, b in itertools.combinations(live, 2))
    assert worst > -1.0
