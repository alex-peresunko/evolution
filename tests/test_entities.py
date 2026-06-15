import pytest

from evolution.entities import Creature, Food
from evolution.spatial import SpatialHashGrid
from evolution import config


WORLD_BOUNDS = (config.WORLD_WIDTH, config.WORLD_HEIGHT)


def test_creature_initial_state():
    c = Creature(WORLD_BOUNDS)
    assert c.health == config.HERBIVORE_HEALTH
    assert c.age == 0
    assert not c.is_adult
    assert len(c.brain.layer_sizes) == len(config.BRAIN_TOPOLOGY)
    assert c.brain.layer_sizes == config.BRAIN_TOPOLOGY


def test_carnivore_initial_state():
    c = Creature(WORLD_BOUNDS, is_carnivore=True)
    assert c.health == config.CARNIVORE_HEALTH
    assert c.is_carnivore


def test_creature_see_returns_correct_length():
    c = Creature(WORLD_BOUNDS)
    inputs = c.see([], [])
    assert len(inputs) == config.BRAIN_TOPOLOGY[0]


def test_food_freshness_decay():
    f = Food(WORLD_BOUNDS)
    v0 = f.get_health_value()
    for _ in range(100):
        f.update()
    v1 = f.get_health_value()
    assert v1 < v0


def test_food_rotten_at_max_lifetime():
    f = Food(WORLD_BOUNDS)
    for _ in range(config.FOOD_MAX_LIFETIME):
        f.update()
    assert f.is_rotten()
    assert f.get_health_value() == 0


def _empty_grid():
    return SpatialHashGrid(config.WORLD_WIDTH, config.WORLD_HEIGHT, config.GRID_CELL_SIZE)


def test_creature_hunger_increments():
    c = Creature(WORLD_BOUNDS)
    c.act([0.0, 0.0], [], [], [], _empty_grid())
    assert c.hunger == 1


def test_creature_health_decreases_per_tick():
    c = Creature(WORLD_BOUNDS)
    initial_health = c.health
    c.act([0.0, 0.0], [], [], [], _empty_grid())
    assert c.health < initial_health
