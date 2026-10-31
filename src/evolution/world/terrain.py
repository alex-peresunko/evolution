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
