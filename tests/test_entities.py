import math

import pytest

from evolution.entities import Carcass, Creature, Food
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


def _adult_at(is_carnivore, x, y, angle=0.0, size=None):
    c = Creature(WORLD_BOUNDS, is_carnivore=is_carnivore)
    c.is_adult = True
    c.genetic_size = c.size = size or c.genetic_size
    c.pos = c._vec2(x, y)
    c.angle = angle
    return c


def _carnivore_step(carn, herbs, tick, carcasses=()):
    """One act() of a carnivore standing still, with prey and carcasses in grids."""
    herb_grid = SpatialHashGrid(*WORLD_BOUNDS, config.GRID_CELL_SIZE)
    carcass_grid = SpatialHashGrid(*WORLD_BOUNDS, config.GRID_CELL_SIZE)
    for h in herbs:
        herb_grid.insert(h)
    for cc in carcasses:
        carcass_grid.insert(cc)
    carn.act([0.0, -1.0], [], herbs, [], herb_grid, herb_grid=herb_grid,
             carcass_grid=carcass_grid, tick=tick)


def test_bite_damages_without_killing_healthy_prey():
    carn = _adult_at(True, 500, 400, size=8)
    prey = _adult_at(False, 505, 400, size=8)
    prey.stamina = 0
    carn.health = 3000
    _carnivore_step(carn, [prey], tick=1)
    assert 0 < prey.health < config.HERBIVORE_HEALTH
    assert prey.stamina == config.ADRENALINE_STAMINA
    assert carn.health > 3000 - 10
    assert carn.score == pytest.approx(config.BITE_SCORE)
    assert prey.last_hit_tick == carn.last_bite_tick == 1


def test_bite_cooldown_and_kill_bonus():
    carn = _adult_at(True, 500, 400, size=8)
    prey = _adult_at(False, 505, 400, size=8)
    _carnivore_step(carn, [prey], tick=1)
    after_first = prey.health
    _carnivore_step(carn, [prey], tick=2)
    assert prey.health == after_first
    prey.health = 100
    _carnivore_step(carn, [prey], tick=1 + config.BITE_COOLDOWN_TICKS)
    assert prey.health <= 0
    assert carn.score == pytest.approx(2 * config.BITE_SCORE + 1)


def test_no_bite_from_behind():
    carn = _adult_at(True, 500, 400, angle=math.pi, size=8)
    prey = _adult_at(False, 505, 400, size=8)
    _carnivore_step(carn, [prey], tick=1)
    assert prey.health == config.HERBIVORE_HEALTH


def test_large_prey_fights_back():
    small = _adult_at(True, 500, 400, size=6)
    big_prey = _adult_at(False, 510, 400, size=18)
    equal_prey = _adult_at(False, 510, 300, size=6)
    _carnivore_step(small, [big_prey], tick=1)
    big_loss = config.CARNIVORE_HEALTH - small.health
    other = _adult_at(True, 500, 300, size=6)
    _carnivore_step(other, [equal_prey], tick=1)
    assert big_loss > config.RETALIATION_DAMAGE
    assert other.health > small.health


def test_carnivore_scavenges_carcass():
    carn = _adult_at(True, 500, 400, size=8)
    carn.health = 2000
    dead = _adult_at(False, 505, 400, size=10)
    carcass = Carcass(dead)
    meat = carcass.meat
    _carnivore_step(carn, [], tick=1, carcasses=[carcass])
    assert carcass.meat == meat - config.CARCASS_EAT_RATE
    assert carn.health > 2000 - 10
    assert carn.hunger == 0
    assert carn.score > 0
    assert carn.eating_ticks == config.CARNIVORE_EAT_TICKS
    # It stays at the carcass, eating every tick, even though the brain says "run".
    pos = (carn.pos.x, carn.pos.y)
    for tick in range(2, 6):
        carn.act([0.0, 1.0], [], [], [], _empty_grid(), tick=tick)
        assert (carn.pos.x, carn.pos.y) == pos
    assert carcass.meat == meat - 5 * config.CARCASS_EAT_RATE


def test_carnivore_meal_ends_when_carcass_is_gone():
    carn = _adult_at(True, 500, 400, size=8)
    carn.health = 2000
    carcass = Carcass(_adult_at(False, 505, 400, size=10))
    _carnivore_step(carn, [], tick=1, carcasses=[carcass])
    carcass.meat = 0
    carn.act([0.0, 1.0], [], [], [], _empty_grid(), tick=2)
    assert carn.eating_ticks == 0 and carn.meal_carcass is None


def _herbivore_step(herb, food_items, speed=1.0):
    food_grid = SpatialHashGrid(*WORLD_BOUNDS, config.GRID_CELL_SIZE)
    for f in food_items:
        food_grid.insert(f)
    herb.act([0.0, speed], food_items, [herb], [], food_grid, food_grid=food_grid)


def test_herbivore_stops_to_eat_and_digests_gradually():
    herb = _adult_at(False, 500, 400)
    herb.health = 1000
    food = Food(WORLD_BOUNDS, pos=(500, 400))
    _herbivore_step(herb, [food], speed=0.0)
    assert food.consumed and herb.score == 1
    assert herb.eating_ticks == config.HERBIVORE_EAT_TICKS
    meal = herb.meal_health
    start = herb.health
    pos = (herb.pos.x, herb.pos.y)
    for _ in range(config.HERBIVORE_EAT_TICKS):
        _herbivore_step(herb, [], speed=1.0)
        assert (herb.pos.x, herb.pos.y) == pos and herb.current_speed == 0
        assert start < herb.health < start + meal
    assert herb.eating_ticks == 0
    decay = config.HERBIVORE_EAT_TICKS * config.HEALTH_LOSS_PER_TICK
    assert herb.health == pytest.approx(start + meal - decay)
    _herbivore_step(herb, [], speed=1.0)
    assert herb.current_speed > 0


def test_bite_interrupts_meal():
    herb = _adult_at(False, 505, 400, size=8)
    herb.eating_ticks, herb.meal_health = 30, 400.0
    carn = _adult_at(True, 500, 400, size=8)
    _carnivore_step(carn, [herb], tick=1)
    assert herb.eating_ticks == 0 and herb.meal_health == 0


def test_instant_eating_when_eat_ticks_zero(monkeypatch):
    import evolution.entities as entities
    monkeypatch.setattr(entities, "HERBIVORE_EAT_TICKS", 0)
    herb = _adult_at(False, 500, 400)
    herb.health = 1000
    _herbivore_step(herb, [Food(WORLD_BOUNDS, pos=(500, 400))], speed=0.0)
    assert herb.eating_ticks == 0 and herb.health > 1000 + 100


def test_surplus_food_is_stored_as_fat():
    herb = _adult_at(False, 500, 400)
    _herbivore_step(herb, [Food(WORLD_BOUNDS, pos=(500, 400))], speed=0.0)
    for _ in range(config.HERBIVORE_EAT_TICKS):
        _herbivore_step(herb, [], speed=0.0)
    assert herb.fat > 0
    assert herb.health == pytest.approx(herb.max_health - config.HEALTH_LOSS_PER_TICK, abs=1)


def test_fat_creature_is_bigger_and_slower():
    lean, fat = _adult_at(False, 500, 400), _adult_at(False, 500, 400)
    fat.fat = config.FAT_MAX
    lean.act([0.0, 1.0], [], [], [], _empty_grid())
    fat.act([0.0, 1.0], [], [], [], _empty_grid())
    assert fat.current_speed < lean.current_speed
    assert fat.size > fat.genetic_size == lean.size


def test_starvation_is_paid_from_fat():
    herb = _adult_at(False, 500, 400)
    herb.fat, herb.hunger = 1000.0, 2 * config.HUNGER_MAX
    health = herb.health
    herb.act([0.0, 0.0], [], [], [], _empty_grid())
    assert herb.health == pytest.approx(health - config.HEALTH_LOSS_PER_TICK)
    assert herb.fat == pytest.approx(1000 - config.FAT_BURN_PER_TICK - config.HUNGER_HEALTH_PENALTY)


def test_full_carnivore_keeps_eating_and_gains_fat():
    carn = _adult_at(True, 500, 400, size=8)
    carcass = Carcass(_adult_at(False, 505, 400, size=10))
    _carnivore_step(carn, [], tick=1, carcasses=[carcass])
    assert carn.eating_ticks == config.CARNIVORE_EAT_TICKS
    for tick in range(2, 10):
        carn.act([0.0, 1.0], [], [], [], _empty_grid(), tick=tick)
    assert carn.eating_ticks > 0 and carn.fat > 0
