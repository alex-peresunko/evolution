import math

import pytest

from evolution import config
from evolution.brain.layout import INPUTS
from evolution.brain.network import NeuralNetwork
from evolution.sim import breeding
from evolution.sim import simulation as simulation_module
from evolution.sim.breeding import mate_preference, mate_seeking_drive
from evolution.sim.populations import founders
from evolution.sim.records import Record
from evolution.sim.simulation import Simulation
from evolution.species import CARNIVORE, HERBIVORE
from evolution.world.creature import Creature
from evolution.world.food import Food
from evolution.world.terrain import Lake
from evolution.world.world import World

WORLD_BOUNDS = (config.WORLD_WIDTH, config.WORLD_HEIGHT)


def _make_sim(n_herbs=2, n_carns=1, n_food=5):
    populations = {HERBIVORE: [Creature(HERBIVORE, WORLD_BOUNDS) for _ in range(n_herbs)],
                   CARNIVORE: [Creature(CARNIVORE, WORLD_BOUNDS) for _ in range(n_carns)]}
    food = [Food(WORLD_BOUNDS) for _ in range(n_food)]
    return Simulation(World(WORLD_BOUNDS, populations, food))


def _creature(species, score=0.0):
    c = Creature(species, WORLD_BOUNDS)
    c.score = score
    return c


def test_step_runs_without_crash():
    sim = _make_sim()
    for _ in range(10):
        sim.step()
    assert sim.tick == 10


def test_step_removes_dead_creature():
    sim = _make_sim(n_herbs=1, n_carns=0)
    sim.world.herbivores[0].health = -1
    sim.step()
    # Possibly repopulated from the Hall of Fame, but the dead one is gone.
    for c in sim.world.herbivores:
        assert c.health >= 0


def test_founders_from_scratch():
    pop = founders(HERBIVORE, WORLD_BOUNDS, 5)
    assert len(pop) == 5
    assert all(c.species is HERBIVORE and c.generation == 1 for c in pop)


def test_founders_from_a_progenitor_start_with_it_unmutated():
    brain = NeuralNetwork(config.BRAIN_TOPOLOGY)
    pop = founders(CARNIVORE, WORLD_BOUNDS, 4, Record(None, brain, None, generation=7))
    assert pop[0].brain is brain
    assert all(c.generation == 7 and c.species is CARNIVORE for c in pop)
    assert any((c.brain.weights[0] != brain.weights[0]).any() for c in pop[1:])


def _adult(species, x, y):
    c = Creature(species, WORLD_BOUNDS)
    c.is_adult = True
    c.pos = c.geometry.vec2(x, y)
    return c


@pytest.mark.parametrize("predator_x, expect_override", [(600, False), (1500, True)])
def test_herbivore_mate_seeking_skipped_when_predator_smelled(predator_x, expect_override):
    herb, mate = _adult(HERBIVORE, 500, 400), _adult(HERBIVORE, 550, 400)
    herb.angle = math.pi  # facing away from the mate
    predator = _adult(CARNIVORE, predator_x, 400)
    world = World(WORLD_BOUNDS, {HERBIVORE: [herb, mate], CARNIVORE: [predator]}, [])
    world.reindex()
    world.perceive()

    outputs = mate_seeking_drive(herb, [0.0, 0.0], world.index.of(HERBIVORE))
    assert (outputs != [0.0, 0.0]) == expect_override


def test_dead_creature_leaves_carcass_that_rots():
    sim = _make_sim(n_herbs=2, n_carns=1)
    victim = sim.world.herbivores[0]
    victim.health = -1
    sim.step()
    assert len(sim.world.carcasses) == 1
    carcass = sim.world.carcasses[0]
    assert (carcass.pos.x, carcass.pos.y) == pytest.approx((victim.pos.x, victim.pos.y))
    assert carcass.species is HERBIVORE
    carcass.age = config.CARCASS_LIFETIME
    for c in sim.world.creatures():
        c.health = c.max_health
    sim.step()
    assert carcass not in sim.world.carcasses


def test_step_stores_signal_and_memory():
    sim = _make_sim(n_herbs=3, n_carns=2)
    sim.step()
    for c in sim.world.creatures():
        assert 0.0 <= c.signal <= 1.0
        assert len(c.memory) == config.NUM_MEMORY
    assert any(m != 0.0 for c in sim.world.herbivores for m in c.memory)


def test_winter_stops_food_respawn_above_the_cap(monkeypatch):
    from evolution.world.environment import food_cap

    sim = _make_sim(n_herbs=0, n_carns=0, n_food=0)
    sim.world.tick = 3 * config.SEASON_LENGTH_TICKS // 4 - 1   # next tick is midwinter
    winter_cap = food_cap(-1)
    sim.world.food = [Food(WORLD_BOUNDS) for _ in range(math.ceil(winter_cap))]
    monkeypatch.setattr(simulation_module, "repopulate", lambda *a, **k: [])
    sim.step()
    assert sim.world.season == pytest.approx(-1)
    assert len(sim.world.food) <= math.ceil(winter_cap)


def test_season_is_a_brain_input():
    sim = _make_sim()
    sim.world.tick = config.SEASON_LENGTH_TICKS // 8
    sim.step()
    for c in sim.world.creatures():
        assert c.brain_inputs[INPUTS.season] == pytest.approx(sim.world.season)


def test_in_water_is_a_brain_input():
    sim = _make_sim(n_herbs=2, n_carns=0)
    wet, dry = sim.world.herbivores
    wet.pos, dry.pos = wet.geometry.vec2(500, 500), dry.geometry.vec2(1500, 500)
    sim.world.lakes = [Lake(500, 500, 100)]
    sim.step()
    assert wet.brain_inputs[INPUTS.in_water] == 1.0 and dry.brain_inputs[INPUTS.in_water] == 0.0


def test_elite_reinjection_replaces_the_weakest_with_hall_of_fame():
    sim = _make_sim(n_herbs=10, n_carns=2)
    for i, c in enumerate(sim.world.herbivores):
        c.score = i
    weakest = {id(c) for c in sim.world.herbivores[:config.ELITE_REINJECT_COUNT]}
    elite = NeuralNetwork(config.BRAIN_TOPOLOGY)
    sim.hall_of_fame[HERBIVORE].best = Record(99, elite, dict(config.DEFAULT_HERBIVORE_GENES))
    carnivores = list(sim.world.carnivores)
    sim._reinject_elites()
    herbs = sim.world.herbivores
    assert len(herbs) == 10
    assert not weakest & {id(c) for c in herbs}
    assert any((c.brain.weights[0] == elite.weights[0]).all() for c in herbs)
    assert sim.world.carnivores == carnivores  # no Hall of Fame, too few to touch


def test_mate_preference_weighs_score_against_attractiveness(monkeypatch):
    monkeypatch.setattr(breeding, "MATE_SCORE_WEIGHT", 1.0)
    plain_skilled, pretty_unskilled = _creature(HERBIVORE, 10.0), _creature(HERBIVORE)
    plain_skilled.attractiveness, pretty_unskilled.attractiveness = 10, 15
    mates = [plain_skilled, pretty_unskilled]
    assert max(mates, key=mate_preference(mates, HERBIVORE)) is plain_skilled


def test_elite_pool_keeps_the_best_retired_brains():
    sim = _make_sim(n_herbs=0, n_carns=0)
    for score in range(config.ELITE_POOL_SIZE + 5):
        sim.retire(_creature(HERBIVORE, score))
    pool = sim.hall_of_fame[HERBIVORE].elites
    assert sorted(r.score for r in pool) == list(range(5, config.ELITE_POOL_SIZE + 5))


def test_records_fade_so_a_newer_brain_can_take_the_hall_of_fame():
    sim = _make_sim(n_herbs=0, n_carns=0)
    sim.retire(_creature(HERBIVORE, 100.0))
    for _ in range(config.HOF_SCORE_HALF_LIFE // config.SCORE_SAMPLE_INTERVAL):
        sim.hall_of_fame.fade(0.5 ** (config.SCORE_SAMPLE_INTERVAL / config.HOF_SCORE_HALF_LIFE))
    records = sim.hall_of_fame[HERBIVORE]
    assert records.best.score == pytest.approx(50.0)
    assert records.elites[0].score == pytest.approx(50.0)
    sim.retire(_creature(HERBIVORE, 60.0))
    assert records.best.score == 60.0


def test_new_record_notifies_listeners():
    events = []

    class Spy:
        def record_set(self, sim, species):
            events.append(species)

    sim = Simulation(World(WORLD_BOUNDS, {}, []), listeners=[Spy()])
    sim.retire(_creature(HERBIVORE, 5.0))
    sim.retire(_creature(HERBIVORE, 1.0))
    assert events == [HERBIVORE]


def test_extinction_breeds_from_the_elite_pool_and_counts_separately():
    sim = _make_sim(n_herbs=0, n_carns=1)
    sim.hall_of_fame[HERBIVORE].elites = [
        Record(1.0, NeuralNetwork(config.BRAIN_TOPOLOGY), dict(config.DEFAULT_HERBIVORE_GENES), g)
        for g in (4, 7)]
    sim.step()
    assert sim.extinctions == 1 and sim.generation == 1
    bred = [c for c in sim.world.herbivores if c.generation > 1]
    assert len(bred) == round(config.NUM_HERBIVOROUS * (1 - config.REPOPULATE_RANDOM_SHARE))
    assert all(c.generation in (5, 8) for c in bred)


def test_generation_is_the_deepest_birth():
    sim = _make_sim(n_herbs=2, n_carns=1)
    a, b = sim.world.herbivores
    for c in (a, b):
        c.is_adult = True
        c.pos = c.geometry.vec2(500, 500)
    a.generation, b.generation = 3, 9
    sim.step()
    children = [c for c in sim.world.herbivores if c not in (a, b)]
    assert children and all(c.generation == 10 for c in children)
    assert sim.generation == 10
