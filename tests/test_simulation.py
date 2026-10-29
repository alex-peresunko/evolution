import math

import pytest

from evolution.entities import Creature, Food, Obstacle
from evolution.simulation import build_sim_state, create_initial_population, update_world
from evolution import config

WORLD_BOUNDS = (config.WORLD_WIDTH, config.WORLD_HEIGHT)


def _make_sim(n_herbs=2, n_carns=1, n_food=5):
    herbs = [Creature(WORLD_BOUNDS) for _ in range(n_herbs)]
    carns = [Creature(WORLD_BOUNDS, is_carnivore=True) for _ in range(n_carns)]
    food = [Food(WORLD_BOUNDS) for _ in range(n_food)]
    return build_sim_state(WORLD_BOUNDS, herbs, carns, food, [])


def test_update_world_runs_without_crash():
    sim_state = _make_sim()
    for _ in range(10):
        update_world(sim_state)
    assert sim_state["tick"] == 10


def test_update_world_removes_dead_creature():
    sim_state = _make_sim(n_herbs=1, n_carns=0)
    sim_state["creatures"][0].health = -1  # already dead
    update_world(sim_state)
    # Dead creature should be removed (possibly repopulated from HoF, but the dead one is gone)
    for c in sim_state["creatures"]:
        assert c.health >= 0


def test_create_initial_population_from_scratch():
    pop, gen = create_initial_population(5, False, WORLD_BOUNDS, "__nonexistent_brain__.json")
    assert len(pop) == 5
    assert gen == 1
    assert all(not c.is_carnivore for c in pop)


def test_build_sim_state_keys():
    sim_state = _make_sim()
    for key in ("creatures", "carnivores", "food_items", "obstacles", "hall_of_fame",
                "generation", "tick", "grid"):
        assert key in sim_state


def _adult(is_carnivore, x, y):
    c = Creature(WORLD_BOUNDS, is_carnivore=is_carnivore)
    c.is_adult = True
    c.pos = c._vec2(x, y)
    return c


@pytest.mark.parametrize("predator_x, expect_override", [(600, False), (1500, True)])
def test_herbivore_mate_seeking_skipped_when_predator_smelled(predator_x, expect_override):
    from evolution.simulation import _apply_mate_seeking_drive

    herb, mate = _adult(False, 500, 400), _adult(False, 550, 400)
    herb.angle = math.pi  # facing away from the mate
    predator = _adult(True, predator_x, 400)
    sim_state = build_sim_state(WORLD_BOUNDS, [herb, mate], [predator], [], [])
    for c in (herb, mate):
        sim_state["herb_grid"].insert(c)
    sim_state["carn_grid"].insert(predator)
    herb.see([], [], [herb, mate], [predator])

    outputs = _apply_mate_seeking_drive(herb, [0.0, 0.0], sim_state["herb_grid"], sim_state["carn_grid"])
    assert (outputs != [0.0, 0.0]) == expect_override


def test_dead_creature_leaves_carcass_that_rots():
    sim_state = _make_sim(n_herbs=2, n_carns=1)
    victim = sim_state["creatures"][0]
    victim.health = -1
    update_world(sim_state)
    assert len(sim_state["carcasses"]) == 1
    carcass = sim_state["carcasses"][0]
    assert (carcass.pos.x, carcass.pos.y) == pytest.approx((victim.pos.x, victim.pos.y))
    carcass.age = config.CARCASS_LIFETIME
    for c in sim_state["creatures"] + sim_state["carnivores"]:
        c.health = c.max_health
    update_world(sim_state)
    assert carcass not in sim_state["carcasses"]


def test_food_respawns_near_patches(monkeypatch):
    from evolution import simulation

    monkeypatch.setattr(simulation, "FOOD_PATCH_SHARE", 1.0)
    sim_state = _make_sim(n_herbs=0, n_carns=0, n_food=0)
    w, h = WORLD_BOUNDS
    for _ in range(200):
        f = simulation._spawn_food(sim_state)
        best = min(
            math.hypot(min(abs(f.pos.x - p["x"]), w - abs(f.pos.x - p["x"])),
                       min(abs(f.pos.y - p["y"]), h - abs(f.pos.y - p["y"])))
            for p in sim_state["food_patches"]
        )
        assert best < 5 * config.FOOD_PATCH_RADIUS


def test_day_light_cycle():
    from evolution.simulation import day_light

    day = config.DAY_LENGTH_TICKS
    assert day_light(0) == pytest.approx(1.0)
    assert day_light(day // 4) == pytest.approx(0.5)
    assert day_light(day // 2) == pytest.approx(0.0)
    assert day_light(day) == pytest.approx(1.0)


def test_update_world_stores_signal_and_memory():
    sim_state = _make_sim(n_herbs=3, n_carns=2)
    update_world(sim_state)
    for c in sim_state["creatures"] + sim_state["carnivores"]:
        assert 0.0 <= c.signal <= 1.0
        assert len(c.memory) == config.NUM_MEMORY
    assert any(m != 0.0 for c in sim_state["creatures"] for m in c.memory)
