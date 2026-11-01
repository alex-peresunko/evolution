import random

import numpy as np
import pytest

from evolution import config
from evolution.app.bootstrap import build_world
from evolution.app.metrics import GameMetrics, genetic_diversity
from evolution.genetics import genetic_distance
from evolution.persistence.autosave import HallOfFameAutosaver
from evolution.sim.populations import founders
from evolution.sim.simulation import Simulation
from evolution.species import ALL_SPECIES, CARNIVORE, HERBIVORE
from evolution.world.creature import DEATH_CAUSES, Creature

WORLD = (config.WORLD_WIDTH, config.WORLD_HEIGHT)


def _samples(metrics):
    return {(s.name, tuple(sorted(s.labels.items()))): s.value
            for family in metrics.collect() for s in family.samples}


@pytest.fixture
def sim():
    random.seed(1)
    np.random.seed(1)
    return Simulation(build_world({s: founders(s, WORLD) for s in ALL_SPECIES}))


def test_without_a_simulation_only_static_metrics(sim):
    names = {name for name, _ in _samples(GameMetrics())}
    assert names == {"evolution_running_seconds_total", "evolution_brain_layer_neurons",
                     "evolution_speciation_distance"}


def test_counts_follow_the_simulation(sim, tmp_path):
    metrics = GameMetrics()
    metrics.watch(sim, HallOfFameAutosaver(save_dir=tmp_path))
    for _ in range(400):
        sim.step()
    s = _samples(metrics)
    for species in ALL_SPECIES:
        sp = (("species", species.name),)
        stats = sim.stats[species]
        assert s[("evolution_population", sp)] == len(sim.world.population(species))
        assert s[("evolution_births_total", sp)] == stats.births
        deaths = sum(s[("evolution_deaths_total", (("cause", c), ("species", species.name)))]
                     for c in DEATH_CAUSES)
        assert deaths == sum(stats.deaths.values())
        assert s[("evolution_death_age_ticks_total", sp)] == stats.death_age_total
        assert s[("evolution_brain_saves_total", sp)] == 0
        assert (s[("evolution_score", (("species", species.name), ("stat", "best")))]
                >= s[("evolution_score", (("species", species.name), ("stat", "mean")))])
    assert s[("evolution_ticks_total", ())] == 400
    assert s[("evolution_food_removed_total", (("fate", "eaten"),))] > 0
    bitten = s[("evolution_deaths_total", (("cause", "bitten"), ("species", "herbivore")))]
    assert s[("evolution_bites_total", ())] >= bitten


def test_genetic_diversity_is_the_rms_pairwise_distance():
    random.seed(3)
    population = [Creature(HERBIVORE, (100, 100), genes={"size": random.uniform(6, 20),
                                                         "lifespan": random.uniform(2e3, 9e3)})
                  for _ in range(5)]
    pairs = [genetic_distance(a.genes, b.genes, HERBIVORE.gene_bounds) ** 2
             for a in population for b in population]
    assert genetic_diversity(population, HERBIVORE) == pytest.approx(np.sqrt(np.mean(pairs)))
    assert genetic_diversity(population[:1], HERBIVORE) == 0.0


def _dying(species, **state):
    c = Creature(species, (100, 100))
    for key, value in state.items():
        setattr(c, key, value)
    return c


def test_death_causes():
    lifespan = HERBIVORE.default_genes["lifespan"]
    assert _dying(HERBIVORE, last_hit_tick=7, hunger=10_000).death_cause(7) == "bitten"
    assert _dying(CARNIVORE, last_hit_tick=7).death_cause(7) == "retaliation"
    assert _dying(CARNIVORE, last_hit_tick=7, last_trampled_tick=7).death_cause(7) == "trampled"
    assert _dying(HERBIVORE, last_impact_tick=7, hunger=10_000).death_cause(7) == "impact"
    assert _dying(HERBIVORE, last_hit_tick=6, hunger=10_000).death_cause(7) == "starved"
    assert _dying(HERBIVORE, age=lifespan + 1).death_cause(7) == "old_age"
    assert _dying(HERBIVORE).death_cause(7) == "attrition"


def test_world_tally_sorts_carcasses_by_fate(sim):
    from evolution.world.food import Carcass
    eaten, rotting = Carcass(sim.world.herbivores[0]), Carcass(sim.world.herbivores[1])
    eaten.meat, rotting.age = 0, 10**9
    sim.world.carcasses[:] = [eaten, rotting]
    sim._update_carcasses()
    tally = sim.world.tally
    assert (tally.carcasses_eaten, tally.carcasses_rotted) == (1, 1)


def test_group_hunting_metrics_follow_the_tally(sim, tmp_path):
    metrics = GameMetrics()
    metrics.watch(sim, HallOfFameAutosaver(save_dir=tmp_path))
    for _ in range(50):
        sim.step()
    tally = sim.world.tally
    tally.bites_confused, tally.pack_bites, tally.tramples, tally.mob_tramples = 3, 2, 5, 4
    s = _samples(metrics)
    assert s[("evolution_bites_confused_total", ())] == 3
    assert s[("evolution_pack_bites_total", ())] == 2
    assert s[("evolution_tramples_total", ())] == 5
    assert s[("evolution_mob_tramples_total", ())] == 4
    for species in ALL_SPECIES:
        assert 0 <= s[("evolution_herd_density", (("species", species.name),))] <= 1
