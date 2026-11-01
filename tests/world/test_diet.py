import math

import pytest

from evolution import config
from evolution.species import CARNIVORE, HERBIVORE
from evolution.world import diet
from evolution.world.creature import Creature
from evolution.world.world import World

WORLD_BOUNDS = (config.WORLD_WIDTH, config.WORLD_HEIGHT)


def _adult(species, x, y, size=8, angle=0.0):
    c = Creature(species, WORLD_BOUNDS)
    c.is_adult = True
    c.genetic_size = c.size = size
    c.pos = c.geometry.vec2(x, y)
    c.angle = angle
    return c


def _bite(carn, herbivores, carnivores=(), tick=1):
    """One act() of carn standing still; returns the world."""
    world = World(WORLD_BOUNDS, {HERBIVORE: list(herbivores),
                                 CARNIVORE: [carn, *carnivores]}, [])
    world.tick = tick
    world.reindex()
    carn.act([0.0, -1.0] + [0.0] * (config.BRAIN_OUTPUTS - 2), world)
    return world


def _herd(n, x=505, y=400):
    return [_adult(HERBIVORE, x + 20 * (i % 4), y + 30 + 15 * (i // 4)) for i in range(n)]


def test_isolated_prey_is_bitten(monkeypatch):
    monkeypatch.setattr(diet, "CONFUSION_MAX_MISS", 1.0)
    carn, prey = _adult(CARNIVORE, 500, 400), _adult(HERBIVORE, 505, 400)
    world = _bite(carn, [prey])
    assert prey.health < config.HERBIVORE_HEALTH
    assert world.tally.bites == 1 and world.tally.bites_confused == 0


def test_herd_confuses_the_bite_and_spends_the_cooldown(monkeypatch):
    monkeypatch.setattr(diet, "CONFUSION_MAX_MISS", 1.0)
    carn, prey = _adult(CARNIVORE, 500, 400), _adult(HERBIVORE, 505, 400)
    world = _bite(carn, [prey, *_herd(config.CONFUSION_HERD_NORM)])
    assert prey.health == config.HERBIVORE_HEALTH
    assert carn.last_bite_tick == 1
    assert world.tally.bites == 0 and world.tally.bites_confused == 1


@pytest.mark.parametrize("gap, bonus", [(10, True), (config.PACK_WINDOW_TICKS + 1, False)])
def test_second_biter_within_the_window_bites_harder(gap, bonus):
    first, second = _adult(CARNIVORE, 500, 400), _adult(CARNIVORE, 505, 392, angle=math.pi / 2)
    prey = _adult(HERBIVORE, 505, 400)
    _bite(first, [prey], tick=1)
    before = prey.health
    world = _bite(second, [prey], tick=1 + gap)
    expected = config.BITE_DAMAGE * (config.PACK_BITE_BONUS if bonus else 1)
    assert before - prey.health == pytest.approx(expected)
    assert world.tally.pack_bites == int(bonus)


def test_retaliation_is_split_between_attackers():
    alone = _adult(CARNIVORE, 500, 400, size=6)
    _bite(alone, [_adult(HERBIVORE, 512, 400, size=12)])
    solo_loss = config.CARNIVORE_HEALTH - alone.health

    carn, helper = _adult(CARNIVORE, 500, 300, size=6), _adult(CARNIVORE, 524, 300, size=6)
    _bite(carn, [_adult(HERBIVORE, 512, 300, size=12)], carnivores=[helper])
    assert config.CARNIVORE_HEALTH - carn.health < solo_loss
    assert solo_loss > 0


def test_kill_gives_earlier_biters_an_assist():
    first, second = _adult(CARNIVORE, 500, 400), _adult(CARNIVORE, 505, 392, angle=math.pi / 2)
    prey = _adult(HERBIVORE, 505, 400)
    _bite(first, [prey], tick=1)
    score = first.score
    prey.health = 1
    _bite(second, [prey], tick=5)
    assert prey.health <= 0
    assert first.score == pytest.approx(score + config.PACK_ASSIST_SCORE)
    assert second.score == pytest.approx(config.BITE_SCORE + 1)


def _charge(herb, carnivores, herbivores=(), tick=1, speed=config.TRAMPLE_REF_SPEED):
    """herb's feed at `speed` among the given creatures; returns the world."""
    world = World(WORLD_BOUNDS, {HERBIVORE: [herb, *herbivores], CARNIVORE: list(carnivores)}, [])
    world.tick = tick
    world.reindex()
    herb.current_speed = speed
    HERBIVORE.diet.feed(herb, world)
    return world


LONE_TRAMPLE = config.TRAMPLE_DAMAGE * 2   # size 16 vs 8 at the reference speed


def test_charge_tramples_and_stuns_the_predator():
    herb, carn = _adult(HERBIVORE, 500, 400, size=16), _adult(CARNIVORE, 520, 400)
    carn.eating_ticks = 10
    world = _charge(herb, [carn])
    assert config.CARNIVORE_HEALTH - carn.health == pytest.approx(LONE_TRAMPLE)
    assert carn.stunned_ticks == config.TRAMPLE_STUN_TICKS and carn.eating_ticks == 0
    assert carn.hurt == 1.0
    assert herb.score == config.TRAMPLE_SCORE and herb.last_trample_tick == 1
    assert world.tally.tramples == 1 and world.tally.mob_tramples == 0


@pytest.mark.parametrize("setup", ["slow", "behind", "cooldown", "juvenile"])
def test_no_trample(setup):
    herb, carn = _adult(HERBIVORE, 500, 400, size=16), _adult(CARNIVORE, 520, 400)
    speed = config.TRAMPLE_REF_SPEED
    if setup == "slow":
        speed = config.TRAMPLE_MIN_SPEED - 0.1
    elif setup == "behind":
        herb.angle = math.pi
    elif setup == "cooldown":
        herb.last_trample_tick = 1 - config.TRAMPLE_COOLDOWN_TICKS + 1
    else:
        herb.is_adult = False
    world = _charge(herb, [carn], speed=speed)
    assert carn.health == config.CARNIVORE_HEALTH and world.tally.tramples == 0


def test_faster_charge_hits_harder_up_to_the_cap():
    carn = _adult(CARNIVORE, 520, 400)
    _charge(_adult(HERBIVORE, 500, 400, size=16), [carn], speed=100)
    assert config.CARNIVORE_HEALTH - carn.health == pytest.approx(
        LONE_TRAMPLE * config.TRAMPLE_MAX_SPEED_RATIO)


def test_herd_gives_courage():
    herb, carn = _adult(HERBIVORE, 500, 400, size=16), _adult(CARNIVORE, 520, 400)
    _charge(herb, [carn], herbivores=_herd(config.CONFUSION_HERD_NORM, x=470, y=380))
    assert config.CARNIVORE_HEALTH - carn.health == pytest.approx(
        LONE_TRAMPLE * (1 + config.HERD_COURAGE_MAX))


@pytest.mark.parametrize("gap, bonus", [(10, True), (config.MOB_WINDOW_TICKS + 1, False)])
def test_mobbing_tramples_harder(gap, bonus):
    carn = _adult(CARNIVORE, 520, 400)
    first = _adult(HERBIVORE, 500, 400, size=16)
    second = _adult(HERBIVORE, 520, 420, size=16, angle=-math.pi / 2)
    _charge(first, [carn], tick=1)
    before = carn.health
    world = _charge(second, [carn], tick=1 + gap)
    assert before - carn.health == pytest.approx(
        LONE_TRAMPLE * (config.MOB_BONUS if bonus else 1))
    assert world.tally.mob_tramples == int(bonus)


def test_trample_kill_credits_the_mob():
    carn = _adult(CARNIVORE, 520, 400)
    first = _adult(HERBIVORE, 500, 400, size=16)
    second = _adult(HERBIVORE, 520, 420, size=16, angle=-math.pi / 2)
    _charge(first, [carn], tick=1)
    carn.health = 1
    _charge(second, [carn], tick=5)
    assert carn.death_cause(5) == "trampled"
    assert second.score == config.TRAMPLE_SCORE + 1
    assert first.score == config.TRAMPLE_SCORE + config.MOB_ASSIST_SCORE


def test_stunned_carnivore_cannot_bite(monkeypatch):
    monkeypatch.setattr(diet, "CONFUSION_MAX_MISS", 0.0)
    carn, prey = _adult(CARNIVORE, 500, 400), _adult(HERBIVORE, 505, 400)
    carn.stunned_ticks = 2
    _bite(carn, [prey], tick=1)
    assert prey.health == config.HERBIVORE_HEALTH and carn.stunned_ticks == 1
    _bite(carn, [prey], tick=2)
    assert prey.health < config.HERBIVORE_HEALTH
