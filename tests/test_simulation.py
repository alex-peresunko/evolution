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


def test_seasons_scale_the_food_cap():
    from evolution.simulation import food_cap, season, season_name
    year = config.SEASON_LENGTH_TICKS
    assert season(0) == pytest.approx(0) and season_name(0) == "Spring"
    assert season(year // 4) == pytest.approx(1) and season_name(year // 4) == "Summer"
    assert season_name(year // 2) == "Autumn"
    assert season(3 * year // 4) == pytest.approx(-1) and season_name(3 * year // 4) == "Winter"
    assert food_cap(1) > food_cap(0) == config.NUM_FOOD * 1.5 > food_cap(-1) > 0


def test_winter_stops_food_respawn_above_the_cap(monkeypatch):
    sim_state = _make_sim(n_herbs=0, n_carns=0, n_food=0)
    sim_state["tick"] = 3 * config.SEASON_LENGTH_TICKS // 4 - 1   # next tick is midwinter
    import evolution.simulation as simulation
    from evolution.simulation import food_cap
    winter_cap = food_cap(-1)
    sim_state["food_items"] = [Food(WORLD_BOUNDS) for _ in range(math.ceil(winter_cap))]
    monkeypatch.setattr(simulation, "_repopulate_from_hof", lambda *a, **k: [])
    update_world(sim_state)
    assert sim_state["season"] == pytest.approx(-1)
    assert len(sim_state["food_items"]) <= math.ceil(winter_cap)


def test_season_is_the_last_brain_input():
    sim_state = _make_sim()
    sim_state["tick"] = config.SEASON_LENGTH_TICKS // 8
    update_world(sim_state)
    for c in sim_state["creatures"] + sim_state["carnivores"]:
        assert c.brain_inputs[-1] == pytest.approx(sim_state["season"])


def test_elite_reinjection_replaces_the_weakest_with_hall_of_fame():
    from evolution.neural_network import NeuralNetwork
    from evolution.simulation import _reinject_elites

    sim_state = _make_sim(n_herbs=10, n_carns=2)
    for i, c in enumerate(sim_state["creatures"]):
        c.score = i
    weakest = {id(c) for c in sim_state["creatures"][:config.ELITE_REINJECT_COUNT]}
    elite = NeuralNetwork(config.BRAIN_TOPOLOGY)
    sim_state["hall_of_fame"]["herbivore"] = {
        "score": 99, "brain": elite, "genes": dict(config.DEFAULT_HERBIVORE_GENES)}
    carnivores = list(sim_state["carnivores"])
    _reinject_elites(sim_state)
    assert len(sim_state["creatures"]) == 10
    assert not weakest & {id(c) for c in sim_state["creatures"]}
    assert any((c.brain.weights[0] == elite.weights[0]).all() for c in sim_state["creatures"])
    assert sim_state["carnivores"] == carnivores  # no Hall of Fame, too few to touch


def test_mate_preference_weighs_score_against_attractiveness(monkeypatch):
    from evolution import simulation
    from evolution.simulation import _mate_preference

    monkeypatch.setattr(simulation, "MATE_SCORE_WEIGHT", 1.0)

    plain_skilled, pretty_unskilled = Creature(WORLD_BOUNDS), Creature(WORLD_BOUNDS)
    plain_skilled.score, plain_skilled.attractiveness = 10.0, 10
    pretty_unskilled.score, pretty_unskilled.attractiveness = 0.0, 15
    mates = [plain_skilled, pretty_unskilled]
    key = _mate_preference(mates, config.DEFAULT_HERBIVORE_GENES)
    assert max(mates, key=key) is plain_skilled


def test_food_patches_bounce_off_world_edges():
    from evolution.simulation import _update_food_patches

    sim_state = _make_sim()
    sim_state["food_patches"] = [{"x": WORLD_BOUNDS[0] - 0.1, "y": 300.0, "heading": 0.0}]
    _update_food_patches(sim_state)
    patch = sim_state["food_patches"][0]
    assert patch["x"] <= WORLD_BOUNDS[0]
    assert math.cos(patch["heading"]) < 0
