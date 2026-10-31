import math
from collections.abc import Callable
from dataclasses import dataclass


class Vector2:
    """Headless 2D vector, API-compatible with pygame.math.Vector2."""
    def __init__(self, x=0.0, y=0.0):
        self.x = float(x)
        self.y = float(y)

    def __add__(self, other):
        return Vector2(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Vector2(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar):
        return Vector2(self.x * scalar, self.y * scalar)

    def __truediv__(self, scalar):
        if scalar == 0:
            return Vector2(0, 0)
        return Vector2(self.x / scalar, self.y / scalar)

    def __iadd__(self, other):
        self.x += other.x
        self.y += other.y
        return self

    def __isub__(self, other):
        self.x -= other.x
        self.y -= other.y
        return self

    def distance_to(self, other):
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    def length(self):
        return math.sqrt(self.x ** 2 + self.y ** 2)

    def __repr__(self):
        return f"Vector2({self.x}, {self.y})"


class SimpleRect:
    """Headless rectangle, API-compatible with pygame.Rect for simulation use."""
    def __init__(self, x, y, width, height):
        self.x, self.y, self.w, self.h = x, y, width, height
        self.left, self.right = x, x + width
        self.top, self.bottom = y, y + height

    def colliderect(self, other_rect):
        return (
            self.x < other_rect.x + other_rect.w
            and self.x + self.w > other_rect.x
            and self.y < other_rect.y + other_rect.h
            and self.y + self.h > other_rect.y
        )

    def clipline(self, p1, p2):
        """Cohen-Sutherland clip; returns [intersection_point] or None."""
        p1_x, p1_y = p1.x, p1.y
        p2_x, p2_y = p2.x, p2.y
        dx, dy = p2_x - p1_x, p2_y - p1_y
        t0, t1 = 0.0, 1.0
        checks = [
            (-dx, p1_x - self.left),
            (dx, self.right - p1_x),
            (-dy, p1_y - self.top),
            (dy, self.bottom - p1_y),
        ]
        for p, q in checks:
            if p == 0:
                if q < 0:
                    return None
            else:
                r = q / p
                if p < 0:
                    if r > t1:
                        return None
                    t0 = max(t0, r)
                else:
                    if r < t0:
                        return None
                    t1 = min(t1, r)
        if t0 < t1 and t0 > 0:
            return [Vector2(p1_x + t0 * dx, p1_y + t0 * dy)]
        return None


@dataclass(frozen=True)
class Geometry:
    """The vector and rectangle types every entity of one world is built with."""

    vec2: Callable
    rect: Callable


HEADLESS = Geometry(Vector2, SimpleRect)
