import math

import numpy as np

from ..config import TREE_TRUNK_FRACTION
from .geometry import HEADLESS


class Obstacle:
    def __init__(self, x, y, width, height, *, geometry=HEADLESS):
        self.rect = geometry.rect(x, y, width, height)


class Tree(Obstacle):
    """Only the trunk (`rect`, centred on x, y) is solid; the crown is drawn over creatures."""

    def __init__(self, x, y, crown_radius, *, geometry=HEADLESS):
        side = crown_radius * TREE_TRUNK_FRACTION
        super().__init__(x - side / 2, y - side / 2, side, side, geometry=geometry)
        self.x, self.y, self.crown_radius = x, y, crown_radius


class Lake:
    """Shore radius at angle t is radius * (1 + sum of a * cos(k * t + phase) over
    `harmonics` (k, a, phase)); no harmonics is a circle."""

    def __init__(self, x, y, radius, harmonics=()):
        self.x, self.y, self.radius = x, y, radius
        self.harmonics = tuple(harmonics)
        self.max_radius = radius * (1 + sum(abs(a) for _, a, _ in self.harmonics))

    def shore(self, theta, cos=math.cos):
        """Shore radius at `theta`; pass cos=np.cos for an array of angles."""
        return self.radius * (1 + sum(a * cos(k * theta + p) for k, a, p in self.harmonics))

    def contains(self, x, y):
        dx, dy = x - self.x, y - self.y
        d2 = dx * dx + dy * dy
        return d2 < self.max_radius ** 2 and d2 < self.shore(math.atan2(dy, dx)) ** 2

    def shore_gap(self, xs, ys):
        """Distance past the shore along the ray from the centre (< 0 = in water), for arrays."""
        dx, dy = xs - self.x, ys - self.y
        return np.hypot(dx, dy) - self.shore(np.arctan2(dy, dx), np.cos)


def circle_hits_rect(x, y, r, rect):
    nx = min(max(x, rect.left), rect.right)
    ny = min(max(y, rect.top), rect.bottom)
    return (x - nx) ** 2 + (y - ny) ** 2 < r ** 2


class TerrainGrid:
    """Per grid cell, the obstacles and lake blobs a creature in that cell can touch, in their
    original order. Obstacle candidates hold only for bodies of size <= `reach`. Cells on the
    border extend to infinity, so positions outside the world are covered too."""

    def __init__(self, world_bounds, obstacles, lakes, cell_size, reach):
        self.cell_size = cell_size
        self.reach = reach
        self.cols = max(1, math.ceil(world_bounds[0] / cell_size))
        self.rows = max(1, math.ceil(world_bounds[1] / cell_size))
        self.cells = {}
        for cx in range(self.cols):
            x0 = -math.inf if cx == 0 else cx * cell_size
            x1 = math.inf if cx == self.cols - 1 else (cx + 1) * cell_size
            for cy in range(self.rows):
                y0 = -math.inf if cy == 0 else cy * cell_size
                y1 = math.inf if cy == self.rows - 1 else (cy + 1) * cell_size
                near_obstacles = tuple(
                    o for o in obstacles
                    if o.rect.left - reach <= x1 and o.rect.right + reach >= x0
                    and o.rect.top - reach <= y1 and o.rect.bottom + reach >= y0)
                near_lakes = tuple(
                    lake for lake in lakes
                    if _box_distance(lake.x, lake.y, x0, x1, y0, y1) <= lake.max_radius + 1)
                self.cells[cx, cy] = (near_obstacles, near_lakes)

    def near(self, x, y):
        """(obstacles, lake blobs) that may touch a body at x, y."""
        return self.cells[min(self.cols - 1, max(0, int(x // self.cell_size))),
                          min(self.rows - 1, max(0, int(y // self.cell_size)))]


def _box_distance(x, y, x0, x1, y0, y1):
    dx = max(x0 - x, 0.0, x - x1)
    dy = max(y0 - y, 0.0, y - y1)
    return math.hypot(dx, dy)
