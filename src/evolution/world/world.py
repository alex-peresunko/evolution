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

    def spawn_food(self, attempts=10):
        """New food near a random patch (FOOD_PATCH_SHARE of the time) or anywhere, on dry land
        unless `attempts` placements all land in a lake."""
        for _ in range(attempts):
            food = self._place_food()
            if not any(lake.contains(food.pos.x, food.pos.y) for lake in self.lakes):
                break
        return food

    def _place_food(self):
        w, h = self.bounds
        pos = None
        if self.food_patches and random.random() < FOOD_PATCH_SHARE:
            patch = random.choice(self.food_patches)
            pos = (min(max(patch.x + random.gauss(0, FOOD_PATCH_RADIUS), FOOD_RADIUS),
                       w - FOOD_RADIUS),
                   min(max(patch.y + random.gauss(0, FOOD_PATCH_RADIUS), FOOD_RADIUS),
                       h - FOOD_RADIUS))
        return Food(self.bounds, geometry=self.geometry, pos=pos)
