import math

import pytest

from evolution import config
from evolution.species import CARNIVORE, HERBIVORE
from evolution.world import world as world_module
from evolution.world.creature import Creature
from evolution.world.environment import FoodPatch, day_light, food_cap, season, season_name
from evolution.world.terrain import Lake, Obstacle, Tree
from evolution.world.world import World
from evolution.world.worldgen import generate_lakes, generate_obstacles, random_shore

WORLD_BOUNDS = (config.WORLD_WIDTH, config.WORLD_HEIGHT)


def _empty_world():
    return World(WORLD_BOUNDS, {}, [])


def test_creatures_lists_herbivores_first():
    herb, carn = Creature(HERBIVORE, WORLD_BOUNDS), Creature(CARNIVORE, WORLD_BOUNDS)
    world = World(WORLD_BOUNDS, {CARNIVORE: [carn], HERBIVORE: [herb]}, [])
    assert world.creatures() == [herb, carn]
    assert world.herbivores == [herb] and world.carnivores == [carn]


def test_food_respawns_near_patches(monkeypatch):
    monkeypatch.setattr(world_module, "FOOD_PATCH_SHARE", 1.0)
    world = _empty_world()
    w, h = WORLD_BOUNDS
    for _ in range(200):
        f = world.spawn_food()
        best = min(
            math.hypot(min(abs(f.pos.x - p.x), w - abs(f.pos.x - p.x)),
                       min(abs(f.pos.y - p.y), h - abs(f.pos.y - p.y)))
            for p in world.food_patches
        )
        assert best < 5 * config.FOOD_PATCH_RADIUS


def test_food_does_not_spawn_in_lakes():
    world = _empty_world()
    world.food_patches = []
    world.lakes = [Lake(WORLD_BOUNDS[0] / 2, WORLD_BOUNDS[1] / 2, WORLD_BOUNDS[1] / 3)]
    for _ in range(200):
        food = world.spawn_food()
        assert not world.lakes[0].contains(food.pos.x, food.pos.y)


def test_food_does_not_spawn_on_obstacles():
    world = _empty_world()
    world.food_patches = []
    rock = Obstacle(0, 0, WORLD_BOUNDS[0] / 4, WORLD_BOUNDS[1])
    world.obstacles = [rock]
    for _ in range(200):
        food = world.spawn_food()
        assert food.pos.x >= rock.rect.right + config.FOOD_RADIUS


def test_patch_food_at_a_wall_does_not_line_up_on_it(monkeypatch):
    monkeypatch.setattr(world_module, "FOOD_PATCH_SHARE", 1.0)
    world = _empty_world()
    world.food_patches = [FoodPatch(0.0, WORLD_BOUNDS[1] / 2, 0.0)]
    xs = [world.spawn_food().pos.x for _ in range(200)]
    assert sum(x == config.FOOD_RADIUS for x in xs) <= 1


def test_food_patches_bounce_off_world_edges():
    world = _empty_world()
    world.food_patches = [FoodPatch(WORLD_BOUNDS[0] - 0.1, 300.0, 0.0)]
    world.drift_food_patches()
    patch = world.food_patches[0]
    assert patch.x <= WORLD_BOUNDS[0]
    assert math.cos(patch.heading) < 0


def test_clock_sets_light_and_season():
    world = _empty_world()
    world.tick = config.SEASON_LENGTH_TICKS // 4 - 1
    world.advance_clock()
    assert world.light == day_light(world.tick)
    assert world.season == pytest.approx(1)


def test_day_light_cycle():
    day = config.DAY_LENGTH_TICKS
    assert day_light(0) == pytest.approx(1.0)
    assert day_light(day // 4) == pytest.approx(0.5)
    assert day_light(day // 2) == pytest.approx(0.0)
    assert day_light(day) == pytest.approx(1.0)


def test_seasons_scale_the_food_cap():
    year = config.SEASON_LENGTH_TICKS
    assert season(0) == pytest.approx(0) and season_name(0) == "Spring"
    assert season(year // 4) == pytest.approx(1) and season_name(year // 4) == "Summer"
    assert season_name(year // 2) == "Autumn"
    assert season(3 * year // 4) == pytest.approx(-1) and season_name(3 * year // 4) == "Winter"
    assert food_cap(1) > food_cap(0) == config.NUM_FOOD * 1.5 > food_cap(-1) > 0


def test_lakes_stay_inside_walls_and_off_obstacles():
    obstacles = [Obstacle(1000, 600, 150, 150)]
    blobs = generate_lakes(WORLD_BOUNDS, obstacles, count=8)
    assert len(blobs) >= 8
    for blob in blobs:
        assert blob.radius <= config.LAKE_RADIUS_RANGE[1]
        assert blob.max_radius <= blob.x <= WORLD_BOUNDS[0] - blob.max_radius
        assert blob.max_radius <= blob.y <= WORLD_BOUNDS[1] - blob.max_radius
        assert not blob.contains(1075, 675)


def test_trees_stay_inside_walls_off_rocks_and_apart():
    obstacles = generate_obstacles(WORLD_BOUNDS, rocks=4, trees=20)
    rocks = [o for o in obstacles if not isinstance(o, Tree)]
    trees = [o for o in obstacles if isinstance(o, Tree)]
    assert len(rocks) == 4 and 0 < len(trees) <= 20
    lo, hi = config.TREE_CROWN_RADIUS_RANGE
    for t in trees:
        r = t.crown_radius
        assert lo <= r <= hi
        assert r <= t.x <= WORLD_BOUNDS[0] - r and r <= t.y <= WORLD_BOUNDS[1] - r
        for rock in rocks:
            nx = min(max(t.x, rock.rect.left), rock.rect.right)
            ny = min(max(t.y, rock.rect.top), rock.rect.bottom)
            assert math.hypot(t.x - nx, t.y - ny) >= r
    for i, a in enumerate(trees):
        for b in trees[i + 1:]:
            assert math.hypot(a.x - b.x, a.y - b.y) >= 0.8 * (a.crown_radius + b.crown_radius)


def test_lake_shore_is_irregular():
    lake = Lake(500, 500, 100, random_shore())
    radii = [lake.shore(t * math.pi / 18) for t in range(36)]
    assert max(radii) - min(radii) > 10
    for t, r in zip(range(36), radii):
        a = t * math.pi / 18
        assert lake.contains(500 + math.cos(a) * (r - 1), 500 + math.sin(a) * (r - 1))
        assert not lake.contains(500 + math.cos(a) * (r + 1), 500 + math.sin(a) * (r + 1))


def test_terrain_grid_keeps_every_obstacle_and_lake_a_body_can_touch():
    import random as rnd

    world = World(WORLD_BOUNDS, {}, [])
    world.obstacles = generate_obstacles(WORLD_BOUNDS, world.geometry)
    world.lakes = generate_lakes(WORLD_BOUNDS, world.obstacles)
    grid, rng = world.terrain, rnd.Random(5)
    for _ in range(3000):
        x = rng.uniform(-50, WORLD_BOUNDS[0] + 50)
        y = rng.uniform(-50, WORLD_BOUNDS[1] + 50)
        size = rng.uniform(1, grid.reach)
        body = world.geometry.rect(x - size, y - size, 2 * size, 2 * size)
        near_obstacles, near_lakes = grid.near(x, y)
        assert ([o for o in world.obstacles if o.rect.colliderect(body)]
                == [o for o in near_obstacles if o.rect.colliderect(body)])
        assert ([lake for lake in world.lakes if lake.contains(x, y)]
                == [lake for lake in near_lakes if lake.contains(x, y)])


def test_replacing_obstacles_rebuilds_the_terrain_grid():
    world = World(WORLD_BOUNDS, {}, [])
    rock = Obstacle(100, 100, 50, 50)
    world.obstacles = [rock]
    assert rock in world.terrain.near(120, 120)[0]
    world.obstacles = []
    assert world.terrain.near(120, 120)[0] == ()
