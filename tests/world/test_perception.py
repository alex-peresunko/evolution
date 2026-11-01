import math
import random

import numpy as np
import pytest

from evolution import config
from evolution.brain.layout import INPUTS
from evolution.species import CARNIVORE, HERBIVORE
from evolution.world.creature import Creature
from evolution.world.food import Carcass, Food
from evolution.world.perception import perceive_all
from evolution.world.terrain import Lake, Obstacle, Tree

WORLD_BOUNDS = (config.WORLD_WIDTH, config.WORLD_HEIGHT)


def _reference_edge_distance(pos, angle, world_bounds):
    dists = []
    for p, u, size in ((pos.x, math.cos(angle), world_bounds[0]),
                       (pos.y, math.sin(angle), world_bounds[1])):
        if u > 0:
            dists.append((size - p) / u)
        elif u < 0:
            dists.append(-p / u)
    return max(0.0, min(dists))


def _reference_crown_hides(p, q, trees):
    for t in trees:
        sx, sy = q.x - p.x, q.y - p.y
        len2 = sx * sx + sy * sy
        k = min(1.0, max(0.0, ((t.x - p.x) * sx + (t.y - p.y) * sy) / len2)) if len2 else 0.0
        r2 = t.crown_radius ** 2
        crosses = (p.x + k * sx - t.x) ** 2 + (p.y + k * sy - t.y) ** 2 < r2
        shared = ((p.x - t.x) ** 2 + (p.y - t.y) ** 2 < r2
                  and (q.x - t.x) ** 2 + (q.y - t.y) ** 2 < r2)
        if crosses and not shared:
            return True
    return False


def _reference_crown_occlusion(p, ux, uy, trees):
    best = math.inf
    for t in trees:
        dx, dy = t.x - p.x, t.y - p.y
        along, d2, r2 = dx * ux + dy * uy, dx * dx + dy * dy, t.crown_radius ** 2
        if d2 - along * along >= r2 and d2 >= r2:
            continue
        half = math.sqrt(max(r2 - (d2 - along * along), 0.0))
        if d2 < r2:
            best = min(best, along + half)
        elif along - half > 0:
            best = min(best, along - half)
    return best


def _reference_see(self, food_items, obstacles, herbivores, carnivores, carcasses=(), light=1.0,
                   season=0.0, lakes=()):
    """The original per-creature, pure-Python Creature.see(), plus scalar versions of the
    later inputs (kin, signal, carcass, light, memory); perceive_all must match it."""
    eye = self.sight_distance * (config.NIGHT_SIGHT_FACTOR + (1 - config.NIGHT_SIGHT_FACTOR) * light)
    inputs = []
    whiskers = []
    for whisker_angle in self.whisker_angles:
        end_point = self.pos + self.geometry.vec2(
            math.cos(self.angle + whisker_angle), math.sin(self.angle + whisker_angle),
        ) * self.sight_distance
        closest_dist = min(self.sight_distance, _reference_edge_distance(
            self.pos, self.angle + whisker_angle, self.world_bounds))
        for obs in obstacles:
            clipped_line = obs.rect.clipline(self.pos, end_point)
            if clipped_line:
                closest_dist = min(closest_dist, self.pos.distance_to(clipped_line[0]))
        whiskers.append(closest_dist)
        inputs.append(1.0 - closest_dist / self.sight_distance)

    trees = [o for o in obstacles if isinstance(o, Tree)]

    def in_sight(target):
        if target.id == self.id or _reference_crown_hides(self.pos, target.pos, trees):
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

    targets = list(herbivores) + list(carcasses) if self.is_carnivore else food_items
    target_whiskers = []
    for whisker_angle, obstacle_dist in zip(self.whisker_angles, whiskers):
        ux, uy = math.cos(self.angle + whisker_angle), math.sin(self.angle + whisker_angle)
        obstacle_dist = min(obstacle_dist, _reference_crown_occlusion(self.pos, ux, uy, trees))
        best = None
        for t in targets:
            dx, dy = t.pos.x - self.pos.x, t.pos.y - self.pos.y
            along, perp = dx * ux + dy * uy, abs(dy * ux - dx * uy)
            if (0 < along <= eye and along < obstacle_dist and perp <= config.WHISKER_TARGET_WIDTH
                    and (best is None or along < best)):
                best = along
        target_whiskers.append(best)
        inputs.append(0.0 if best is None else 1.0 - best / eye)
    inputs.append(season)

    cc_smell_angle, cc_smell_strength = 0.0, 0.0
    if self.is_carnivore and carcasses:
        closest = min(carcasses, key=lambda cc: self.pos.distance_to(cc.pos))
        dist = self.pos.distance_to(closest.pos)
        if dist < config.SMELL_DISTANCE:
            cc_smell_angle = rel_angle(closest) / math.pi
            cc_smell_strength = 1.0 - dist / config.SMELL_DISTANCE
    inputs += [cc_smell_angle, cc_smell_strength]
    inputs.append(1.0 if any(lake.contains(self.pos.x, self.pos.y) for lake in lakes) else 0.0)
    inputs.append(self.eating_ticks / self.species.eat_ticks if self.eating_ticks else 0.0)
    inputs.append(self.hurt)
    return inputs, whiskers, target_whiskers, nearest_target_dist, nearest_smell_strength


def _random_creature(rnd, is_carnivore, obstacles):
    lo_hi = config.CARNIVORE_GENE_MIN_MAX if is_carnivore else config.GENE_MIN_MAX
    genes = {k: rnd.uniform(*lo_hi[k]) for k in lo_hi}
    c = Creature(CARNIVORE if is_carnivore else HERBIVORE, WORLD_BOUNDS, genes=genes)
    while any(o.rect.left <= c.pos.x <= o.rect.right and o.rect.top <= c.pos.y <= o.rect.bottom
              for o in obstacles):
        c.pos = c.geometry.vec2(rnd.uniform(0, WORLD_BOUNDS[0]), rnd.uniform(0, WORLD_BOUNDS[1]))
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
    obstacles += [Tree(rnd.uniform(60, WORLD_BOUNDS[0] - 60), rnd.uniform(60, WORLD_BOUNDS[1] - 60),
                       rnd.uniform(*config.TREE_CROWN_RADIUS_RANGE)) for _ in range(12)]
    herbs = [_random_creature(rnd, False, obstacles) for _ in range(n_herb)]
    carns = [_random_creature(rnd, True, obstacles) for _ in range(n_carn)]
    food = [Food(WORLD_BOUNDS) for _ in range(n_food)]
    carcasses = [Carcass(_random_creature(rnd, rnd.random() < 0.5, obstacles)) for _ in range(seed * 4)]
    everyone = herbs + carns

    lakes = [Lake(rnd.uniform(0, WORLD_BOUNDS[0]), rnd.uniform(0, WORLD_BOUNDS[1]),
                  rnd.uniform(100, 300)) for _ in range(seed)]
    season = light - 0.5
    got = perceive_all(everyone, food, herbs, carns, obstacles, carcasses, light, season, lakes)
    assert got.shape == (len(everyone), config.BRAIN_TOPOLOGY[0])
    for row, c in zip(got, everyone):
        inputs, whiskers, target_whiskers, target_dist, smell = _reference_see(c, food, obstacles, herbs, carns,
                                                              carcasses, light, season, lakes)
        np.testing.assert_allclose(row, inputs, atol=1e-9)
        np.testing.assert_allclose(c.whiskers, whiskers, atol=1e-9)
        assert [d is None for d in c.target_whiskers] == [d is None for d in target_whiskers]
        np.testing.assert_allclose([d or 0 for d in c.target_whiskers],
                                   [d or 0 for d in target_whiskers], atol=1e-9)
        if target_dist is None:
            assert c.nearest_target_dist is None
        else:
            assert c.nearest_target_dist == pytest.approx(target_dist)
        assert c.nearest_smell_strength == pytest.approx(smell)


def test_whisker_starting_inside_obstacle_is_touching():
    obstacle = Obstacle(100, 100, 100, 100)
    c = Creature(HERBIVORE, WORLD_BOUNDS)
    c.pos = c.geometry.vec2(150, 150)
    inputs = perceive_all([c], [], [c], [], [obstacle])
    assert list(inputs[0, :config.NUM_WHISKERS]) == [1.0] * config.NUM_WHISKERS
    assert c.whiskers == [0.0] * config.NUM_WHISKERS


def test_target_whisker_sees_food_ahead_and_is_occluded():
    c = Creature(HERBIVORE, WORLD_BOUNDS)
    c.pos, c.angle = c.geometry.vec2(500, 500), 0.0
    food = Food(WORLD_BOUNDS)
    food.pos = type(food.pos)(500 + c.sight_distance / 2, 503)
    tw = INPUTS.target_whiskers
    centre = config.NUM_WHISKERS // 2

    inputs = perceive_all([c], [food], [c], [], [])
    assert inputs[0, tw][centre] == pytest.approx(0.5)
    assert c.target_whiskers[centre] == pytest.approx(c.sight_distance / 2)

    wall = Obstacle(500 + c.sight_distance / 4, 480, 10, 40)
    inputs = perceive_all([c], [food], [c], [], [wall])
    assert inputs[0, tw][centre] == 0.0
    assert c.target_whiskers[centre] is None


def test_whiskers_see_the_world_edge():
    c = Creature(HERBIVORE, WORLD_BOUNDS)
    c.pos = c.geometry.vec2(WORLD_BOUNDS[0] - 40, 500)
    c.angle = 0.0
    perceive_all([c], [], [c], [], [])
    centre = config.NUM_WHISKERS // 2
    assert c.whiskers[centre] == pytest.approx(40)


def _facing(is_carnivore, x, y, angle=0.0):
    c = Creature(CARNIVORE if is_carnivore else HERBIVORE, WORLD_BOUNDS)
    c.pos, c.angle = c.geometry.vec2(x, y), angle
    return c


def _sees_prey(carn, herb, obstacles):
    perceive_all([carn], [], [herb], [carn], obstacles)
    return carn.nearest_target_dist is not None


def test_crown_hides_prey_behind_and_under_it():
    tree = Tree(700, 400, 50)
    carn = _facing(True, 500, 400)
    assert _sees_prey(carn, _facing(False, 900, 400), [])
    assert not _sees_prey(carn, _facing(False, 900, 400), [tree])
    assert not _sees_prey(carn, _facing(False, 710, 400), [tree])
    assert _sees_prey(carn, _facing(False, 900, 600), [tree])


def test_creatures_under_the_same_crown_see_each_other_but_not_out():
    tree = Tree(700, 400, 60)
    carn = _facing(True, 660, 400)
    assert _sees_prey(carn, _facing(False, 720, 400), [tree])
    assert not _sees_prey(carn, _facing(False, 900, 400), [tree])


def test_herbivore_cannot_see_food_or_predator_behind_a_crown():
    tree = Tree(570, 400, 30)
    herb = _facing(False, 500, 400)
    food = Food(WORLD_BOUNDS)
    food.pos = type(food.pos)(500 + herb.sight_distance * 0.9, 400)
    pred = _facing(True, 500 + herb.sight_distance * 0.8, 400)
    perceive_all([herb], [food], [herb], [pred], [tree])
    assert herb.nearest_target_dist is None
    perceive_all([herb], [food], [herb], [pred], [])
    assert herb.nearest_target_dist is not None


def test_crown_occludes_target_whiskers_but_not_obstacle_whiskers():
    tree = Tree(700, 400, 50)
    carn = _facing(True, 500, 400)
    carn.whisker_angles = [0.0] * config.NUM_WHISKERS
    prey = _facing(False, 900, 400)
    perceive_all([carn], [], [prey], [carn], [tree])
    assert all(d is None for d in carn.target_whiskers)
    trunk_left = tree.rect.left - 500
    assert all(math.isclose(d, min(trunk_left, carn.sight_distance)) for d in carn.whiskers)
