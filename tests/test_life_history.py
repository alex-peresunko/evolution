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
