import pytest

from evolution import config
from evolution.genetics import compatible
from evolution.sim.breeding import mate_seeking_drive, stochastic_round
from evolution.sim.simulation import Simulation
from evolution.species import CARNIVORE, HERBIVORE
from evolution.world.creature import Creature
from evolution.world.world import World

WORLD_BOUNDS = (config.WORLD_WIDTH, config.WORLD_HEIGHT)


def _act(c):
    world = World(WORLD_BOUNDS, {c.species: [c]}, [])
    world.reindex()
    c.act([0.0, 0.0], world)


def _adult(is_carnivore=False, **genes):
    species = CARNIVORE if is_carnivore else HERBIVORE
    c = Creature(species, WORLD_BOUNDS, genes={**species.default_genes, **genes})
    c.pos = c.geometry.vec2(WORLD_BOUNDS[0] / 2, WORLD_BOUNDS[1] / 2)  # clear of the walls
    c.is_adult = True
    c.age = config.ADULTHOOD_AGE
    return c


@pytest.mark.parametrize("is_carnivore", [False, True])
def test_default_genes_pay_the_base_upkeep(is_carnivore):
    assert sum(config.UPKEEP_WEIGHTS.values()) == pytest.approx(1.0)
    assert _adult(is_carnivore).upkeep() == pytest.approx(config.HEALTH_LOSS_PER_TICK)


def test_costly_traits_raise_upkeep():
    base = _adult().upkeep()
    d = config.DEFAULT_HERBIVORE_GENES
    assert _adult(sight_distance=d["sight_distance"] * 2).upkeep() > base
    assert _adult(attractiveness=d["attractiveness"] * 3).upkeep() > base
    assert _adult(lifespan=d["lifespan"] * 2).upkeep() > base
    assert _adult(size=d["size"] * 2).upkeep() > base
    assert _adult(sight_distance=d["sight_distance"] / 2).upkeep() < base


def test_body_upkeep_follows_kleiber_and_ignores_fat():
    small, big = _adult(size=8), _adult(size=16)
    body = config.UPKEEP_WEIGHTS["body"] * config.HEALTH_LOSS_PER_TICK
    assert big.upkeep() - small.upkeep() == pytest.approx(
        body * (16 ** 0.75 - 8 ** 0.75) / config.DEFAULT_HERBIVORE_GENES["size"] ** 0.75)
    juvenile = Creature(HERBIVORE, WORLD_BOUNDS)
    assert juvenile.upkeep(config.JUVENILE_SIZE_FACTOR) < juvenile.upkeep()


def test_act_charges_upkeep():
    c = _adult(sight_distance=config.DEFAULT_HERBIVORE_GENES["sight_distance"] * 2)
    health = c.health
    _act(c)
    assert c.health == pytest.approx(health - c.upkeep())


def _breeding_sim(genes1=None, genes2=None, is_carnivore=False):
    """Two healthy adults next to each other; one step lets them mate."""
    p1, p2 = _adult(is_carnivore, **(genes1 or {})), _adult(is_carnivore, **(genes2 or {}))
    vec2 = p1.geometry.vec2
    p1.pos, p2.pos = vec2(500, 400), vec2(505, 400)
    return Simulation(World(WORLD_BOUNDS, {p1.species: [p1, p2]}, [])), p1, p2


def _offspring(sim, parents):
    return [c for c in sim.world.creatures() if set(c.parent_ids) == set(parents)]


def test_child_inherits_the_first_parents_lineage():
    sim, p1, p2 = _breeding_sim()
    p2.descent = 4
    sim.step()
    children = _offspring(sim, (p1.id, p2.id))
    assert children
    for child in children:
        assert child.lineage in (p1.lineage, p2.lineage)
        assert child.descent == 5


def test_gene_history_sampled_per_species():
    sim, p1, p2 = _breeding_sim(genes1={"size": 8}, genes2={"size": 12})
    sim.stats.sample(sim.world, sim.hall_of_fame)
    history = sim.stats[HERBIVORE].gene_history
    assert set(history) == set(config.DEFAULT_HERBIVORE_GENES)
    assert list(history["size"]) == [pytest.approx(10)]
    assert history["size"].maxlen == config.CHART_HISTORY_SAVE_MAX
    assert not sim.stats[CARNIVORE].gene_history["size"]  # no carnivores: no sample
    assert (sim.stats[HERBIVORE].lineages, sim.stats[CARNIVORE].lineages) == (2, 0)


def test_senescence_starts_at_the_lifespan_and_grows():
    c = _adult(lifespan=4000)
    c.age = 3999
    assert c.senescence() == 0
    c.age = 6000
    mid = c.senescence()
    c.age = 8000
    assert 0 < mid < c.senescence() == pytest.approx(config.SENESCENCE_RATE)


def test_old_creature_loses_more_health():
    young, old = _adult(), _adult()
    old.age = int(old.genes["lifespan"] * 2)
    for c in (young, old):
        _act(c)
    assert old.max_health - old.health == pytest.approx(
        young.max_health - young.health + old.senescence(), rel=1e-3)


@pytest.mark.parametrize("is_carnivore", [False, True])
def test_litter_splits_the_endowment(is_carnivore):
    sim, p1, p2 = _breeding_sim({"litter_size": 3}, {"litter_size": 3}, is_carnivore)
    sim.step()
    young = _offspring(sim, (p1.id, p2.id))
    assert len(young) == 3
    for c in young:
        assert c.health == pytest.approx(c.max_health * config.LITTER_ENDOWMENT / 3)


def test_single_young_gets_full_health():
    sim, p1, p2 = _breeding_sim()
    sim.step()
    (child,) = _offspring(sim, (p1.id, p2.id))
    assert child.health == child.max_health


def test_stochastic_round_hits_the_mean():
    import random

    random.seed(0)
    draws = [stochastic_round(1.25) for _ in range(4000)]
    assert set(draws) == {1, 2}
    assert sum(draws) / len(draws) == pytest.approx(1.25, abs=0.03)


_DIVERGED = ({"color": 0.0, "size": 6, "lifespan": 1500}, {"color": 1.0, "size": 20, "lifespan": 15000})


def test_incompatible_pair_does_not_breed():
    sim, p1, p2 = _breeding_sim(*_DIVERGED)
    assert not compatible(p1.genes, p2.genes, HERBIVORE)
    sim.step()
    assert not _offspring(sim, (p1.id, p2.id))


def test_mate_seeking_ignores_incompatible_partners():
    sim, p1, p2 = _breeding_sim(*_DIVERGED)
    p2.pos = p2.geometry.vec2(700, 400)
    sim.world.reindex()
    kin = sim.world.index.of(HERBIVORE)
    p1.nearest_smell_strength = 0.0
    assert mate_seeking_drive(p1, [0.0, 0.0], kin) == [0.0, 0.0]
    p2.genes = dict(p1.genes)
    assert mate_seeking_drive(p1, [0.0, 0.0], kin) != [0.0, 0.0]
