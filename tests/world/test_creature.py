import math

import pytest

from evolution import config
from evolution.brain.layout import INPUTS, OUTPUTS
from evolution.species import CARNIVORE, HERBIVORE
from evolution.world.creature import Creature
from evolution.world.food import Carcass, Food
from evolution.world.perception import perceive_all
from evolution.world.terrain import Lake, Tree
from evolution.world.world import World

WORLD_BOUNDS = (config.WORLD_WIDTH, config.WORLD_HEIGHT)


def _act(c, outputs, *, food=(), herbivores=(), carcasses=(), obstacles=(), lakes=(), tick=0):
    """One act() of c in a world holding exactly the given things."""
    world = World(WORLD_BOUNDS, {HERBIVORE: list(herbivores)}, list(food), list(obstacles),
                  lakes)
    world.carcasses = list(carcasses)
    world.tick = tick
    world.reindex()
    c.act(list(outputs) + [0.0] * (config.BRAIN_OUTPUTS - len(outputs)), world)


def test_creature_initial_state():
    c = Creature(HERBIVORE, WORLD_BOUNDS)
    assert c.health == config.HERBIVORE_HEALTH
    assert c.age == 0
    assert not c.is_adult
    assert len(c.brain.layer_sizes) == len(config.BRAIN_TOPOLOGY)
    assert c.brain.layer_sizes == config.BRAIN_TOPOLOGY


def test_carnivore_initial_state():
    c = Creature(CARNIVORE, WORLD_BOUNDS)
    assert c.health == config.CARNIVORE_HEALTH
    assert c.is_carnivore


def test_perception_row_has_brain_input_width():
    c = Creature(HERBIVORE, WORLD_BOUNDS)
    inputs = perceive_all([c], [], [], [], [])[0]
    assert len(inputs) == config.BRAIN_TOPOLOGY[0]


def test_food_freshness_decay():
    f = Food(WORLD_BOUNDS)
    v0 = f.health_value()
    for _ in range(100):
        f.update()
    v1 = f.health_value()
    assert v1 < v0


def test_food_rotten_at_max_lifetime():
    f = Food(WORLD_BOUNDS)
    for _ in range(config.FOOD_MAX_LIFETIME):
        f.update()
    assert f.is_rotten()
    assert f.health_value() == 0


def test_creature_hunger_increments():
    c = Creature(HERBIVORE, WORLD_BOUNDS)
    _act(c, [0.0, 0.0])
    assert c.hunger == 1


def test_creature_health_decreases_per_tick():
    c = Creature(HERBIVORE, WORLD_BOUNDS)
    initial_health = c.health
    _act(c, [0.0, 0.0])
    assert c.health < initial_health


def _adult_at(is_carnivore, x, y, angle=0.0, size=None):
    c = Creature(CARNIVORE if is_carnivore else HERBIVORE, WORLD_BOUNDS)
    c.is_adult = True
    c.genetic_size = c.size = size or c.genetic_size
    c.pos = c.geometry.vec2(x, y)
    c.angle = angle
    return c


def _carnivore_step(carn, herbs, tick, carcasses=()):
    """One act() of a carnivore standing still, with prey and carcasses around."""
    _act(carn, [0.0, -1.0], herbivores=herbs, carcasses=carcasses, tick=tick)


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


def test_bite_hurts_the_prey_and_the_hurt_fades():
    carn = _adult_at(True, 500, 400, size=8)
    prey = _adult_at(False, 505, 400, size=8)
    _carnivore_step(carn, [prey], tick=1)
    assert prey.hurt == 1.0 and carn.hurt == 0.0
    _act(prey, [0.0, 0.0])
    assert prey.hurt == pytest.approx(1 - 1 / config.HURT_FADE_TICKS)
    for _ in range(config.HURT_FADE_TICKS):
        _act(prey, [0.0, 0.0])
    assert prey.hurt == 0.0


def test_retaliation_hurts_the_carnivore():
    carn = _adult_at(True, 500, 400, size=6)
    prey = _adult_at(False, 510, 400, size=12)
    _carnivore_step(carn, [prey], tick=1)
    assert carn.hurt == 1.0


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
        _act(carn, [0.0, 1.0], tick=tick)
        assert (carn.pos.x, carn.pos.y) == pos
    assert carcass.meat == meat - 5 * config.CARCASS_EAT_RATE


def test_carnivore_meal_ends_when_carcass_is_gone():
    carn = _adult_at(True, 500, 400, size=8)
    carn.health = 2000
    carcass = Carcass(_adult_at(False, 505, 400, size=10))
    _carnivore_step(carn, [], tick=1, carcasses=[carcass])
    carcass.meat = 0
    _act(carn, [0.0, 1.0], tick=2)
    assert carn.eating_ticks == 0 and carn.meal_carcass is None


def _herbivore_step(herb, food_items, speed=1.0):
    _act(herb, [0.0, speed], food=food_items, herbivores=[herb])


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


def test_stop_eating_output_breaks_off_the_meal():
    herb = _adult_at(False, 500, 400)
    herb.health = 1000
    _herbivore_step(herb, [Food(WORLD_BOUNDS, pos=(500, 400))], speed=0.0)
    stop = [0.0] * config.BRAIN_OUTPUTS
    stop[OUTPUTS.speed] = 1.0
    stop[OUTPUTS.stop_eating] = config.MEAL_BREAK_THRESHOLD - 0.01
    _act(herb, stop, herbivores=[herb])
    assert herb.eating_ticks == config.HERBIVORE_EAT_TICKS - 1 and herb.current_speed == 0
    stop[OUTPUTS.stop_eating] = config.MEAL_BREAK_THRESHOLD + 0.01
    world = World(WORLD_BOUNDS, {HERBIVORE: [herb]}, [], [], ())
    world.reindex()
    herb.act(stop, world)
    assert herb.eating_ticks == 0 and herb.meal_health == 0
    assert herb.current_speed > 0
    assert world.tally.meals_abandoned[HERBIVORE.name] == 1


def test_eating_input_is_the_share_of_the_meal_left():
    herb = _adult_at(False, 500, 400)
    herb.eating_ticks = config.HERBIVORE_EAT_TICKS // 3
    idle = _adult_at(False, 300, 300)
    inputs = perceive_all([herb, idle], [], [herb, idle], [], [])
    assert inputs[0, INPUTS.eating] == pytest.approx(herb.eating_ticks / config.HERBIVORE_EAT_TICKS)
    assert inputs[1, INPUTS.eating] == 0.0


def test_bite_interrupts_meal():
    herb = _adult_at(False, 505, 400, size=8)
    herb.eating_ticks, herb.meal_health = 30, 400.0
    carn = _adult_at(True, 500, 400, size=8)
    _carnivore_step(carn, [herb], tick=1)
    assert herb.eating_ticks == 0 and herb.meal_health == 0


def test_instant_eating_when_eat_ticks_zero(monkeypatch):
    monkeypatch.setattr(HERBIVORE, "eat_ticks", 0)
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
    _act(lean, [0.0, 1.0])
    _act(fat, [0.0, 1.0])
    assert fat.current_speed < lean.current_speed
    assert fat.size > fat.genetic_size == lean.size


def test_starvation_is_paid_from_fat():
    herb = _adult_at(False, 500, 400)
    herb.fat, herb.hunger = 1000.0, 2 * config.HUNGER_MAX
    health = herb.health
    _act(herb, [0.0, 0.0])
    assert herb.health == pytest.approx(health - config.HEALTH_LOSS_PER_TICK)
    assert herb.fat == pytest.approx(1000 - config.FAT_BURN_PER_TICK - config.HUNGER_HEALTH_PENALTY)


def test_full_carnivore_keeps_eating_and_gains_fat():
    carn = _adult_at(True, 500, 400, size=8)
    carcass = Carcass(_adult_at(False, 505, 400, size=10))
    _carnivore_step(carn, [], tick=1, carcasses=[carcass])
    assert carn.eating_ticks == config.CARNIVORE_EAT_TICKS
    for tick in range(2, 10):
        _act(carn, [0.0, 1.0], tick=tick)
    assert carn.eating_ticks > 0 and carn.fat > 0


def test_world_edge_is_a_wall_not_a_wrap():
    bounds = (config.WORLD_WIDTH, config.WORLD_HEIGHT)
    c = Creature(HERBIVORE, bounds)
    c.pos = c.geometry.vec2(bounds[0] - c.size - 1, 400)
    c.angle = 0.0  # heading straight at the right edge
    health = c.health
    for _ in range(5):
        _act(c, [0.0, 1.0])
        assert c.size <= c.pos.x <= bounds[0] - c.size
    assert c.health < health - config.HEALTH_LOST_ON_HIT
    assert math.isclose(c.pos.y, 400)


@pytest.mark.parametrize("is_carnivore", [False, True])
def test_swimming_is_slow_and_drains_stamina(is_carnivore):
    lake = Lake(500, 400, 100)
    dry, wet = _adult_at(is_carnivore, 200, 400), _adult_at(is_carnivore, 500, 400)
    dry.stamina = wet.stamina = 500
    _act(dry, [0.0, 0.5], lakes=[lake])
    _act(wet, [0.0, 0.5], lakes=[lake])
    assert wet.in_water and not dry.in_water
    assert wet.current_speed == pytest.approx(dry.current_speed * config.WATER_SPEED_FACTOR)
    cost = (config.CARNIVORE_SWIM_STAMINA_COST if is_carnivore
            else config.HERBIVORE_SWIM_STAMINA_COST)
    assert wet.stamina == pytest.approx(500 - cost)
    assert dry.stamina > 500


def test_carnivores_tire_faster_in_water_than_herbivores():
    assert config.CARNIVORE_SWIM_STAMINA_COST > config.HERBIVORE_SWIM_STAMINA_COST


def test_tree_trunk_is_a_centred_square():
    tree = Tree(500, 400, 50)
    side = 50 * config.TREE_TRUNK_FRACTION
    assert math.isclose(tree.rect.w, side) and math.isclose(tree.rect.h, side)
    assert math.isclose(tree.rect.x + side / 2, 500) and math.isclose(tree.rect.y + side / 2, 400)


def test_only_the_tree_trunk_blocks():
    tree = Tree(500, 400, 50)
    trunk = _adult_at(False, 500, 400, size=8)
    crown = _adult_at(False, 500, 360, size=8)
    for c in (trunk, crown):
        health = c.health
        _act(c, [0.0, 0.0], obstacles=[tree])
        c.lost = health - c.health
    assert trunk.lost >= config.HEALTH_LOST_ON_HIT
    assert crown.lost < config.HEALTH_LOST_ON_HIT


def test_stamina_drains_with_effort_even_when_weak():
    weak = _adult_at(False, 500, 400)
    weak.health = weak.max_health * 0.4
    weak.stamina = 500
    _act(weak, [0.0, 1.0])
    assert weak.stamina == pytest.approx(500 - config.STAMINA_DEPLETION_RATE)


def test_cruising_below_the_effort_threshold_regenerates_stamina():
    c = _adult_at(False, 500, 400)
    c.stamina = 500
    _act(c, [0.0, config.STAMINA_EFFORT_THRESHOLD - 0.05])
    assert c.stamina == pytest.approx(500 + config.STAMINA_REGEN_RATE)


def test_winded_creature_recovers_while_trying_to_run():
    c = _adult_at(False, 500, 400)
    c.stamina = 0
    for _ in range(3):
        _act(c, [0.0, 1.0])
    assert c.winded and c.stamina == pytest.approx(3 * config.STAMINA_REGEN_RATE)
    c.stamina = config.STAMINA_RECOVERY_SHARE * c.max_stamina
    _act(c, [0.0, 1.0])
    assert not c.winded and c.stamina < config.STAMINA_RECOVERY_SHARE * c.max_stamina
