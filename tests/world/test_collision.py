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
    collision.separate([a, b], [], [], 0)
    assert _gap(a, b) == pytest.approx(0, abs=1e-6)
    assert a.pos.y == b.pos.y == 400


def test_heavier_creature_moves_less():
    big, small = _body(HERBIVORE, 500, 400, size=16), _body(CARNIVORE, 510, 400, size=8)
    collision.separate([big, small], [], [], 0)
    assert 500 - big.pos.x < small.pos.x - 510


def test_coincident_creatures_separate():
    a, b = _body(HERBIVORE, 500, 400), _body(HERBIVORE, 500, 400)
    collision.separate([a, b], [], [], 0)
    assert _gap(a, b) == pytest.approx(0, abs=1e-6)


def test_carcass_is_immovable():
    dead = _body(HERBIVORE, 500, 400, size=10)
    carcass = Carcass(dead)
    c = _body(CARNIVORE, 505, 400, size=6)
    collision.separate([c], [carcass], [], 0)
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
    collision.separate([a, b], [], [], 0)
    assert (a.pos.x, b.pos.x) == (500, 505)


def test_push_into_a_rock_is_reverted():
    rock = Obstacle(470, 380, 20, 40)
    a, b = _body(HERBIVORE, 500, 400), _body(HERBIVORE, 505, 400)
    collision.separate([a, b], [], [rock], 0)
    assert (a.pos.x, a.pos.y) == (500, 400)
    assert b.pos.x > 505


def test_push_through_a_wall_is_clamped():
    a, b = _body(HERBIVORE, 8, 400), _body(HERBIVORE, 12, 400)
    collision.separate([a, b], [], [], 0)
    assert a.pos.x >= a.size


def test_collisions_off(monkeypatch):
    monkeypatch.setattr(collision, "BODY_RADIUS_FACTOR", 0.0)
    a, b = _body(HERBIVORE, 500, 400), _body(HERBIVORE, 505, 400)
    collision.separate([a, b], [], [], 0)
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


def _moving(species, x, y, speed, angle=0.0, size=8):
    c = _body(species, x, y, size)
    c.current_speed, c.angle = speed, angle
    return c


def _momentum(*bodies):
    return sum(b.size ** 2 * b.knock.x for b in bodies), sum(b.size ** 2 * b.knock.y for b in bodies)


def test_head_on_impulse_conserves_momentum_and_restitution():
    a = _moving(HERBIVORE, 500, 400, 3.0, angle=0.0)
    b = _moving(HERBIVORE, 510, 400, 3.0, angle=np.pi)
    collision.separate([a, b], [], [], 0)
    friction = config.KNOCK_FRICTION
    assert _momentum(a, b) == pytest.approx((0, 0), abs=1e-9)
    va, vb = 3.0 + a.knock.x / friction, -3.0 + b.knock.x / friction
    assert vb - va == pytest.approx(config.RESTITUTION * 6.0)


def test_light_rammer_bounces_off_a_heavy_body():
    light = _moving(CARNIVORE, 500, 400, 3.0, size=6)
    heavy = _body(HERBIVORE, 512, 400, size=12)
    collision.separate([light, heavy], [], [], 0)
    assert light.knock.x < 0 < heavy.knock.x
    assert abs(light.knock.x) > abs(heavy.knock.x)
    assert light.knock.y == heavy.knock.y == 0


def test_bodies_moving_apart_get_no_impulse():
    a = _moving(HERBIVORE, 500, 400, 3.0, angle=np.pi)
    b = _moving(HERBIVORE, 510, 400, 3.0, angle=0.0)
    collision.separate([a, b], [], [], 0)
    assert (a.knock.x, b.knock.x) == (0, 0)


def test_knock_moves_the_creature_and_fades():
    c = _body(HERBIVORE, 500, 400)
    c.knock = c.geometry.vec2(2.0, 0.0)
    collision.separate([c], [], [], 0)
    assert c.pos.x == pytest.approx(502)
    assert c.knock.x == pytest.approx(2.0 * config.KNOCK_FRICTION)
    for _ in range(20):
        collision.separate([c], [], [], 0)
    assert c.knock.x == 0
    assert c.pos.x == pytest.approx(500 + 2.0 / (1 - config.KNOCK_FRICTION), abs=0.2)


def test_creature_bounces_off_a_carcass():
    carcass = Carcass(_body(HERBIVORE, 500, 400, size=10))
    c = _moving(CARNIVORE, 512, 400, 3.0, angle=np.pi, size=6)
    collision.separate([c], [carcass], [], 0)
    assert c.knock.x == pytest.approx((1 + config.RESTITUTION) * 3.0 * config.KNOCK_FRICTION)
    assert (carcass.pos.x, carcass.pos.y) == (500, 400)


def test_soft_impact_does_no_damage():
    a, b = _moving(HERBIVORE, 500, 400, 2.0), _body(HERBIVORE, 510, 400)
    collision.separate([a, b], [], [], 7)
    assert (a.health, b.health) == (a.max_health, b.max_health)
    assert a.last_impact_tick is None


def test_hard_impact_hurts_and_can_kill():
    carcass = Carcass(_body(HERBIVORE, 500, 400, size=10))
    c = _moving(CARNIVORE, 512, 400, 10.0, angle=np.pi, size=6)
    c.health = 10
    collision.separate([c], [carcass], [], 7)
    dv = (1 + config.RESTITUTION) * 10.0
    assert c.health == pytest.approx(10 - config.IMPACT_DAMAGE * (dv - config.IMPACT_DAMAGE_THRESHOLD))
    assert c.death_cause(7) == "impact"


def test_knock_into_a_rock_is_cancelled():
    rock = Obstacle(470, 380, 20, 40)
    c = _body(HERBIVORE, 500, 400)
    c.knock = c.geometry.vec2(-5.0, 0.0)
    collision.separate([c], [], [rock], 0)
    assert (c.pos.x, c.knock.x, c.health) == (500, 0, c.max_health)


def test_knock_into_a_wall_stops_on_that_axis():
    c = _body(HERBIVORE, 9, 400)
    c.knock = c.geometry.vec2(-5.0, 3.0)
    collision.separate([c], [], [], 0)
    assert c.pos.x == c.size
    assert c.knock.x == 0 and c.knock.y == pytest.approx(3.0 * config.KNOCK_FRICTION)
    assert c.health == c.max_health


@pytest.mark.parametrize("distinct", [True, False])
def test_overlapping_matches_the_full_pair_matrix(distinct):
    rng = np.random.default_rng(3)
    axy = rng.uniform(0, 400, (120, 2))
    ar = rng.uniform(2, 15, 120)
    bxy = axy if distinct else rng.uniform(0, 400, (80, 2))
    br = ar if distinct else rng.uniform(2, 15, 80)
    full = np.hypot(bxy[None, :, 0] - axy[:, None, 0],
                    bxy[None, :, 1] - axy[:, None, 1]) < ar[:, None] + br[None, :]
    if distinct:
        full &= np.triu(np.ones_like(full), 1).astype(bool)
    i, j = collision._overlapping(axy, ar, bxy, br, distinct=distinct)
    want_i, want_j = np.nonzero(full)
    assert len(want_i) > 0
    assert i.tolist() == want_i.tolist() and j.tolist() == want_j.tolist()
