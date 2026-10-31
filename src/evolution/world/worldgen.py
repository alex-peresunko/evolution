"""Random terrain: rocks, trees and lakes."""
import math
import random

from ..config import (
    LAKE_LOBES,
    LAKE_RADIUS_RANGE,
    LAKE_SHORE_HARMONICS,
    LAKE_SHORE_ROUGHNESS,
    NUM_LAKES,
    NUM_OBSTACLES,
    NUM_TREES,
    TREE_CROWN_RADIUS_RANGE,
)
from .geometry import HEADLESS
from .terrain import Lake, Obstacle, Tree, circle_hits_rect


def generate_obstacles(world_bounds, geometry=HEADLESS, rocks=NUM_OBSTACLES, trees=NUM_TREES,
                       attempts=50):
    """`rocks` rocks, then up to `trees` trees whose crowns stay inside the walls, off the
    rocks, and no closer to another tree than 0.8 x the sum of their crown radii."""
    w, h = world_bounds
    obstacles = [
        Obstacle(random.randint(100, w - 100), random.randint(100, h - 100),
                 random.randint(50, 150), random.randint(50, 150), geometry=geometry)
        for _ in range(rocks)
    ]
    placed = []
    for _ in range(trees * attempts):
        if len(placed) == trees:
            break
        r = random.uniform(*TREE_CROWN_RADIUS_RANGE)
        x, y = random.uniform(r, w - r), random.uniform(r, h - r)
        if any(circle_hits_rect(x, y, r, o.rect) for o in obstacles):
            continue
        if any(math.hypot(x - t.x, y - t.y) < 0.8 * (r + t.crown_radius) for t in placed):
            continue
        placed.append(Tree(x, y, r, geometry=geometry))
    return obstacles + placed


def generate_lakes(world_bounds, obstacles=(), count=NUM_LAKES, attempts=50):
    """Up to `count` random lakes inside the walls that don't touch an obstacle. Each lake is
    1..LAKE_LOBES overlapping `Lake` blobs, all returned in one flat list."""
    w, h = world_bounds
    blobs, placed = [], 0
    for _ in range(count * attempts):
        if placed == count:
            break
        lake = _random_lake(random.uniform(0, w), random.uniform(0, h))
        if all(b.max_radius <= b.x <= w - b.max_radius and b.max_radius <= b.y <= h - b.max_radius
               and not any(circle_hits_rect(b.x, b.y, b.max_radius, obs.rect)
                           for obs in obstacles)
               for b in lake):
            blobs += lake
            placed += 1
    return blobs


def _random_lake(x, y):
    r = random.uniform(*LAKE_RADIUS_RANGE)
    blobs = [Lake(x, y, r, random_shore())]
    for _ in range(random.randint(*LAKE_LOBES) - 1):
        parent = random.choice(blobs)
        a = random.uniform(0, 2 * math.pi)
        off = parent.radius * random.uniform(0.6, 1.0)
        blobs.append(Lake(parent.x + math.cos(a) * off, parent.y + math.sin(a) * off,
                          r * random.uniform(0.4, 0.8), random_shore()))
    return blobs


def random_shore(roughness=LAKE_SHORE_ROUGHNESS, harmonics=LAKE_SHORE_HARMONICS):
    """Lake harmonics with amplitudes falling off with frequency and summing to `roughness`."""
    ks = range(2, harmonics + 2)
    amps = [random.uniform(0.3, 1.0) / k ** 1.4 for k in ks]
    scale = roughness / sum(amps)
    return [(k, a * scale, random.uniform(0, 2 * math.pi)) for k, a in zip(ks, amps)]
