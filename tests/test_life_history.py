import pytest

from evolution import config
from evolution.entities import Creature
from evolution.spatial import SpatialHashGrid

WORLD_BOUNDS = (config.WORLD_WIDTH, config.WORLD_HEIGHT)


def _grid():
    return SpatialHashGrid(*WORLD_BOUNDS, config.GRID_CELL_SIZE)


def _adult(is_carnivore=False, **genes):
    defaults = config.DEFAULT_CARNIVORE_GENES if is_carnivore else config.DEFAULT_HERBIVORE_GENES
    c = Creature(WORLD_BOUNDS, is_carnivore=is_carnivore, genes={**defaults, **genes})
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
    juvenile = Creature(WORLD_BOUNDS)
    assert juvenile.upkeep(config.JUVENILE_SIZE_FACTOR) < juvenile.upkeep()


def test_act_charges_upkeep():
    c = _adult(sight_distance=config.DEFAULT_HERBIVORE_GENES["sight_distance"] * 2)
    health = c.health
    c.act([0.0, 0.0], [], [], [], _grid())
    assert c.health == pytest.approx(health - c.upkeep())


def _breeding_sim(genes1=None, genes2=None, is_carnivore=False):
    """Two healthy adults next to each other; one update_world lets them mate."""
    from evolution.simulation import build_sim_state
    p1, p2 = _adult(is_carnivore, **(genes1 or {})), _adult(is_carnivore, **(genes2 or {}))
    p1.pos, p2.pos = p1._vec2(500, 400), p2._vec2(505, 400)
    herbs, carns = ([], [p1, p2]) if is_carnivore else ([p1, p2], [])
    return build_sim_state(WORLD_BOUNDS, herbs, carns, [], []), p1, p2


def _offspring(sim, parents):
    return [c for c in sim["creatures"] + sim["carnivores"] if set(c.parent_ids) == set(parents)]


def test_child_inherits_the_first_parents_lineage():
    from evolution.simulation import update_world
    sim, p1, p2 = _breeding_sim()
    p2.descent = 4
    update_world(sim)
    children = _offspring(sim, (p1.id, p2.id))
    assert children
    for child in children:
        assert child.lineage in (p1.lineage, p2.lineage)
        assert child.descent == 5


def test_gene_history_sampled_per_species():
    from evolution.simulation import sample_genes
    sim, p1, p2 = _breeding_sim(genes1={"size": 8}, genes2={"size": 12})
    sample_genes(sim)
    history = sim["gene_history"]["herbivore"]
    assert set(history) == set(config.DEFAULT_HERBIVORE_GENES)
    assert list(history["size"]) == [pytest.approx(10)]
    assert history["size"].maxlen == config.CHART_HISTORY_SAVE_MAX
    assert not sim["gene_history"]["carnivore"]["size"]  # no carnivores: no sample
    assert sim["lineages"] == {"herbivore": 2, "carnivore": 0}
