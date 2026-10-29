import math
import random

import numpy as np
import pytest

from evolution import config
from evolution.entities import Carcass, Creature, Food, Obstacle
from evolution.perception import perceive_all

WORLD_BOUNDS = (config.WORLD_WIDTH, config.WORLD_HEIGHT)


def _reference_see(self, food_items, obstacles, herbivores, carnivores, carcasses=(), light=1.0):
    """The original per-creature, pure-Python Creature.see(), plus scalar versions of the
    later inputs (kin, signal, carcass, light, memory); perceive_all must match it."""
    eye = self.sight_distance * (config.NIGHT_SIGHT_FACTOR + (1 - config.NIGHT_SIGHT_FACTOR) * light)
    inputs = []
    whiskers = []
    for whisker_angle in self.whisker_angles:
        end_point = self.pos + self._vec2(
            math.cos(self.angle + whisker_angle), math.sin(self.angle + whisker_angle),
        ) * self.sight_distance
        closest_dist = self.sight_distance
        for obs in obstacles:
            clipped_line = obs.rect.clipline(self.pos, end_point)
            if clipped_line:
                closest_dist = min(closest_dist, self.pos.distance_to(clipped_line[0]))
        whiskers.append(closest_dist)
        inputs.append(1.0 - closest_dist / self.sight_distance)

    def in_sight(target):
        if target.id == self.id:
            return False
        vec = target.pos - self.pos
        if 0 < vec.length() <= eye:
            angle_to_target = (math.atan2(vec.y, vec.x) - self.angle + math.pi) % (2 * math.pi) - math.pi
            return abs(angle_to_target) <= self.sight_angle / 2
        return False

    def target_info(target):
        dist = self.pos.distance_to(target.pos)
        angle = (math.atan2(target.pos.y - self.pos.y, target.pos.x - self.pos.x)
                 - self.angle + math.pi) % (2 * math.pi) - math.pi
        return angle / (self.sight_angle / 2), dist / eye

    nearest_prey = None
    if self.is_carnivore:
        visible_prey = [h for h in herbivores if in_sight(h)]
        if visible_prey:
            nearest_prey = min(visible_prey, key=lambda h: self.pos.distance_to(h.pos))
            info = target_info(nearest_prey)
        else:
            info = (0, 1)
        nearest_target_dist = info[1] if visible_prey else None
        inputs.extend(info)
    else:
        visible_food = [f for f in food_items if in_sight(f)]
        food_angle, food_dist = (
            target_info(min(visible_food, key=lambda f: self.pos.distance_to(f.pos)))
            if visible_food else (0, 1)
        )
        nearest_target_dist = food_dist if visible_food else None
        visible_preds = [c for c in carnivores if in_sight(c)]
        predator_angle, predator_dist = (
            target_info(min(visible_preds, key=lambda c: self.pos.distance_to(c.pos)))
            if visible_preds else (0, 1)
        )
        avoid_angle = food_angle - predator_angle * (1 - predator_dist) ** 2
        inputs.extend([np.clip(avoid_angle, -1, 1), food_dist])

    smell_angle, smell_strength, nearest_smell_strength = 0.0, 0.0, 0.0
    others = [o for o in (herbivores if self.is_carnivore else carnivores) if o.id != self.id]
    if others:
        closest = min(others, key=lambda p: self.pos.distance_to(p.pos))
        dist = self.pos.distance_to(closest.pos)
        if dist < config.SMELL_DISTANCE:
            angle = (math.atan2(closest.pos.y - self.pos.y, closest.pos.x - self.pos.x)
                     - self.angle + math.pi) % (2 * math.pi) - math.pi
            smell_angle = angle / math.pi
            smell_strength = 1.0 - dist / config.SMELL_DISTANCE
            # Fixed vs. the original see(), which only recorded this for carnivores.
            nearest_smell_strength = smell_strength
    inputs.extend([smell_angle, smell_strength])

    pvx, pvy = 0.0, 0.0
    if nearest_prey is not None:
        spd = nearest_prey.current_speed / nearest_prey.max_speed if nearest_prey.max_speed > 0 else 0
        pvx, pvy = math.cos(nearest_prey.angle) * spd, math.sin(nearest_prey.angle) * spd
    inputs.extend([pvx, pvy])

    inputs += [
        self.health / self.max_health, math.sin(self.angle), math.cos(self.angle),
        self.size / config.GENE_MIN_MAX["size"][1],
        self.current_speed / self.genetic_max_speed,
        self.sight_distance / config.MAX_NORMALIZED_SIGHT_DISTANCE,
        self.sight_angle / (2 * math.pi),
        self.stamina / self.max_stamina,
        min(1.0, self.hunger / config.HUNGER_MAX),
    ]

    def rel_angle(target):
        vec = target.pos - self.pos
        return (math.atan2(vec.y, vec.x) - self.angle + math.pi) % (2 * math.pi) - math.pi

    kin = [k for k in (carnivores if self.is_carnivore else herbivores) if k.id != self.id]
    near = [k for k in kin if self.pos.distance_to(k.pos) < config.KIN_SENSE_RADIUS]
    if near:
        closest = min(near, key=lambda k: self.pos.distance_to(k.pos))
        inputs += [rel_angle(closest) / math.pi,
                   self.pos.distance_to(closest.pos) / config.KIN_SENSE_RADIUS]
    else:
        inputs += [0.0, 1.0]
    inputs.append(min(1.0, len(near) / config.KIN_DENSITY_NORM))
    heard = [(k.signal * (1 - self.pos.distance_to(k.pos) / config.SIGNAL_RADIUS), k)
             for k in kin if self.pos.distance_to(k.pos) < config.SIGNAL_RADIUS]
    loudest = max(heard, key=lambda h: h[0], default=(0.0, None))
    inputs += [rel_angle(loudest[1]) / math.pi if loudest[0] > 0 else 0.0, loudest[0]]

    visible_carcasses = [cc for cc in carcasses if in_sight(cc)] if self.is_carnivore else []
    if visible_carcasses:
        inputs.extend(target_info(min(visible_carcasses, key=lambda cc: self.pos.distance_to(cc.pos))))
    else:
        inputs += [0.0, 1.0]
    inputs.append(light)
    inputs.extend(self.memory)
    return inputs, whiskers, nearest_target_dist, nearest_smell_strength


def _random_creature(rnd, is_carnivore, obstacles):
    lo_hi = config.CARNIVORE_GENE_MIN_MAX if is_carnivore else config.GENE_MIN_MAX
    genes = {k: rnd.uniform(*lo_hi[k]) for k in lo_hi}
    c = Creature(WORLD_BOUNDS, is_carnivore=is_carnivore, genes=genes)
    while any(o.rect.left <= c.pos.x <= o.rect.right and o.rect.top <= c.pos.y <= o.rect.bottom
              for o in obstacles):
        c.pos = c._vec2(rnd.uniform(0, WORLD_BOUNDS[0]), rnd.uniform(0, WORLD_BOUNDS[1]))
    c.angle = rnd.uniform(0, 2 * math.pi)
    c.size = c.genetic_size * rnd.uniform(0.2, 1.0)
    c.health = c.max_health * rnd.uniform(0.05, 1.0)
    c.current_speed = rnd.uniform(0, c.max_speed)
    c.stamina = c.max_stamina * rnd.uniform(0, 1)
    c.hunger = rnd.randint(0, 2 * config.HUNGER_MAX)
    c.signal = rnd.choice([0.0, rnd.uniform(0, 1)])
    c.memory = [rnd.uniform(-1, 1) for _ in range(config.NUM_MEMORY)]
    return c


@pytest.mark.parametrize("light", [1.0, 0.3])
@pytest.mark.parametrize("seed, n_herb, n_carn, n_food", [
    (0, 40, 40, 60), (1, 120, 30, 75), (2, 5, 150, 10), (3, 1, 1, 0), (4, 0, 20, 30), (5, 30, 0, 30),
])
def test_perceive_all_matches_reference(seed, n_herb, n_carn, n_food, light):
    rnd = random.Random(seed)
    random.seed(seed)
    obstacles = [Obstacle(rnd.randint(100, 1700), rnd.randint(100, 700),
                          rnd.randint(50, 150), rnd.randint(50, 150)) for _ in range(5)]
    herbs = [_random_creature(rnd, False, obstacles) for _ in range(n_herb)]
    carns = [_random_creature(rnd, True, obstacles) for _ in range(n_carn)]
    food = [Food(WORLD_BOUNDS) for _ in range(n_food)]
    carcasses = [Carcass(_random_creature(rnd, rnd.random() < 0.5, obstacles)) for _ in range(seed * 4)]
    everyone = herbs + carns

    got = perceive_all(everyone, food, herbs, carns, obstacles, carcasses, light)
    assert got.shape == (len(everyone), config.BRAIN_TOPOLOGY[0])
    for row, c in zip(got, everyone):
        inputs, whiskers, target_dist, smell = _reference_see(c, food, obstacles, herbs, carns,
                                                              carcasses, light)
        np.testing.assert_allclose(row, inputs, atol=1e-9)
        np.testing.assert_allclose(c.whiskers, whiskers, atol=1e-9)
        if target_dist is None:
            assert c._nearest_target_dist is None
        else:
            assert c._nearest_target_dist == pytest.approx(target_dist)
        assert c._nearest_smell_strength == pytest.approx(smell)


def test_whisker_starting_inside_obstacle_is_touching():
    obstacle = Obstacle(100, 100, 100, 100)
    c = Creature(WORLD_BOUNDS)
    c.pos = c._vec2(150, 150)
    inputs = perceive_all([c], [], [c], [], [obstacle])
    assert list(inputs[0, :config.NUM_WHISKERS]) == [1.0] * config.NUM_WHISKERS
    assert c.whiskers == [0.0] * config.NUM_WHISKERS
