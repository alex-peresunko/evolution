"""Time of day, seasons and the drifting food patches."""
import math
import random
from dataclasses import dataclass

from ..config import (
    DAY_LENGTH_TICKS,
    FOOD_PATCH_SPEED,
    FOOD_PATCH_TURN,
    NUM_FOOD,
    SEASON_FOOD_AMPLITUDE,
    SEASON_LENGTH_TICKS,
)


def day_light(tick):
    """0 at midnight, 1 at noon; a run starts at noon."""
    return 0.5 + 0.5 * math.cos(2 * math.pi * tick / DAY_LENGTH_TICKS)


def season(tick):
    """-1 at midwinter, 1 at midsummer; a run starts in spring (0, rising)."""
    return math.sin(2 * math.pi * tick / SEASON_LENGTH_TICKS)


def season_name(tick):
    phase = (tick / SEASON_LENGTH_TICKS) % 1
    return ("Spring", "Summer", "Autumn", "Winter")[int((phase + 0.125) * 4) % 4]


def food_cap(season_value):
    """Most food items on the map at once: plentiful summers, lean winters."""
    return NUM_FOOD * 1.5 * (1 + SEASON_FOOD_AMPLITUDE * season_value)


@dataclass
class FoodPatch:
    """A wandering spot around which most respawned food lands."""

    x: float
    y: float
    heading: float

    @classmethod
    def scatter(cls, world_bounds, count):
        w, h = world_bounds
        return [cls(random.uniform(0, w), random.uniform(0, h), random.uniform(0, 2 * math.pi))
                for _ in range(count)]

    def drift(self, world_bounds):
        """One tick of a random walk that bounces off the world edges."""
        w, h = world_bounds
        self.heading += random.uniform(-FOOD_PATCH_TURN, FOOD_PATCH_TURN)
        x = self.x + math.cos(self.heading) * FOOD_PATCH_SPEED
        y = self.y + math.sin(self.heading) * FOOD_PATCH_SPEED
        if not 0 <= x <= w:
            self.heading = math.pi - self.heading
        if not 0 <= y <= h:
            self.heading = -self.heading
        self.x, self.y = min(max(x, 0), w), min(max(y, 0), h)
