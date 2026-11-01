import random

from ..config import (
    FOOD_PATCH_COUNT,
    FOOD_PATCH_RADIUS,
    FOOD_PATCH_SHARE,
    FOOD_RADIUS,
    GRID_CELL_SIZE,
)
from ..species import ALL_SPECIES, CARNIVORE, HERBIVORE
from .environment import FoodPatch, day_light, season
from .food import Food
from .geometry import HEADLESS
from .perception import perceive_all
from .spatial import SpatialIndex
from .tally import WorldTally
from .terrain import circle_hits_rect


class World:
    """Everything that exists in the simulated world: terrain, creatures, food and carcasses,
    plus the clock (tick, light, season)."""

    def __init__(self, world_bounds, populations, food, obstacles=(), lakes=(), *,
                 geometry=HEADLESS):
        self.bounds = world_bounds
        self.geometry = geometry
        self.populations = {s: list(populations.get(s, ())) for s in ALL_SPECIES}
        self.food = food
        self.obstacles = obstacles
        self.lakes = list(lakes)
        self.carcasses = []
        self.food_patches = FoodPatch.scatter(world_bounds, FOOD_PATCH_COUNT)
        self.tick = 0
        self.light = 1.0
        self.season = 0.0
        self.index = SpatialIndex(world_bounds, ALL_SPECIES, GRID_CELL_SIZE)
        self.tally = WorldTally()

    @property
    def herbivores(self):
        return self.populations[HERBIVORE]

    @property
    def carnivores(self):
        return self.populations[CARNIVORE]

    def population(self, species):
        return self.populations[species]

    def creatures(self):
        """Every creature, species in ALL_SPECIES order."""
        return [c for s in ALL_SPECIES for c in self.populations[s]]

    def advance_clock(self):
        self.tick += 1
        self.light = day_light(self.tick)
        self.season = season(self.tick)

    def reindex(self):
        self.index.rebuild(self.food, self.populations, self.carcasses)

    def perceive(self):
        """Brain input rows for creatures() (see perception.perceive_all)."""
        return perceive_all(self.creatures(), self.food, self.herbivores, self.carnivores,
                            self.obstacles, self.carcasses, self.light, self.season, self.lakes)

    # --- food ------------------------------------------------------------------------------

    def drift_food_patches(self):
        for patch in self.food_patches:
            patch.drift(self.bounds)

    def spawn_food(self, attempts=10, near_patches=True):
        """New food near a random patch (FOOD_PATCH_SHARE of the time, unless near_patches is
        False) or anywhere, on dry land off obstacles unless `attempts` placements all fail."""
        for _ in range(attempts):
            food = self._place_food(near_patches)
            if self._open_ground(food.pos.x, food.pos.y):
                break
        return food

    def _open_ground(self, x, y):
        return not (any(lake.contains(x, y) for lake in self.lakes)
                    or any(circle_hits_rect(x, y, FOOD_RADIUS, o.rect) for o in self.obstacles))

    def _place_food(self, near_patches):
        w, h = self.bounds
        pos = None
        if near_patches and self.food_patches and random.random() < FOOD_PATCH_SHARE:
            patch = random.choice(self.food_patches)
            pos = (min(max(patch.x + random.gauss(0, FOOD_PATCH_RADIUS), FOOD_RADIUS),
                       w - FOOD_RADIUS),
                   min(max(patch.y + random.gauss(0, FOOD_PATCH_RADIUS), FOOD_RADIUS),
                       h - FOOD_RADIUS))
        return Food(self.bounds, geometry=self.geometry, pos=pos)
