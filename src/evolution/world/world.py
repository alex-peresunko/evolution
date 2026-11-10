import random

from ..config import (
    FAT_SIZE_FACTOR,
    FOOD_PATCH_COUNT,
    FOOD_PATCH_RADIUS,
    FOOD_PATCH_SHARE,
    FOOD_RADIUS,
    FOOD_REGROW_CROWD,
    FOOD_REGROW_CROWD_RADIUS,
    FOOD_REGROW_RADIUS,
    FOOD_REGROW_SHARE,
    GRID_CELL_SIZE,
)
from ..species import ALL_SPECIES, CARNIVORE, HERBIVORE
from .collision import separate
from .environment import FoodPatch, day_light, season
from .food import Food
from .geometry import HEADLESS
from .perception import perceive_all
from .spatial import SpatialIndex
from .tally import WorldTally
from .terrain import TerrainGrid, circle_hits_rect


def _fold(v, lo, hi):
    if v < lo:
        v = 2 * lo - v
    elif v > hi:
        v = 2 * hi - v
    return min(max(v, lo), hi)


# Largest body: the biggest size gene, full grown, with full fat stores; +2 covers pygame.Rect's
# integer truncation.
_MAX_BODY_SIZE = max(s.gene_bounds["size"][1] for s in ALL_SPECIES) * (1 + FAT_SIZE_FACTOR) + 2


class World:
    """Everything that exists in the simulated world: terrain, creatures, food and carcasses,
    plus the clock (tick, light, season)."""

    def __init__(self, world_bounds, populations, food, obstacles=(), lakes=(), *,
                 geometry=HEADLESS):
        self.bounds = world_bounds
        self.geometry = geometry
        self.populations = {s: list(populations.get(s, ())) for s in ALL_SPECIES}
        self.food = food
        self._obstacles = obstacles
        self._lakes = list(lakes)
        self._terrain = None
        self.carcasses = []
        self.food_patches = FoodPatch.scatter(world_bounds, FOOD_PATCH_COUNT)
        self.tick = 0
        self.light = 1.0
        self.season = 0.0
        self.index = SpatialIndex(world_bounds, ALL_SPECIES, GRID_CELL_SIZE)
        self.tally = WorldTally()

    @property
    def obstacles(self):
        return self._obstacles

    @obstacles.setter
    def obstacles(self, obstacles):
        self._obstacles, self._terrain = obstacles, None

    @property
    def lakes(self):
        return self._lakes

    @lakes.setter
    def lakes(self, lakes):
        self._lakes, self._terrain = lakes, None

    @property
    def terrain(self):
        """TerrainGrid of the obstacles and lakes; rebuilt when either list is replaced."""
        if self._terrain is None:
            self._terrain = TerrainGrid(self.bounds, self._obstacles, self._lakes,
                                        GRID_CELL_SIZE, _MAX_BODY_SIZE)
        return self._terrain

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
                            self.obstacles, self.carcasses, self.light, self.season, self.lakes,
                            self.tick)

    def separate(self):
        separate(self.creatures(), self.carcasses, self.obstacles, self.tick, self.terrain)

    # --- food ------------------------------------------------------------------------------

    def drift_food_patches(self):
        for patch in self.food_patches:
            patch.drift(self.bounds)

    def spawn_food(self, attempts=10, anywhere=False):
        """New food on dry land off obstacles, unless `attempts` placements all fail. It sprouts
        beside existing food (FOOD_REGROW_SHARE), else near a random patch (FOOD_PATCH_SHARE of
        the rest), else anywhere; `anywhere` skips the first two. Crowding is judged by the food
        grid as of the last reindex()."""
        for _ in range(attempts):
            food = Food(self.bounds, geometry=self.geometry,
                        pos=None if anywhere else self._food_site())
            if self._open_ground(food.pos.x, food.pos.y):
                break
        return food

    def _open_ground(self, x, y):
        return not (any(lake.contains(x, y) for lake in self.lakes)
                    or any(circle_hits_rect(x, y, FOOD_RADIUS, o.rect) for o in self.obstacles))

    def _food_site(self):
        """(x, y) for new food, or None for anywhere."""
        if self.food and random.random() < FOOD_REGROW_SHARE:
            parent = random.choice(self.food).pos
            site = self._scatter(parent.x, parent.y, FOOD_REGROW_RADIUS)
            if not self._crowded(*site):
                return site
        if self.food_patches and random.random() < FOOD_PATCH_SHARE:
            patch = random.choice(self.food_patches)
            return self._scatter(patch.x, patch.y, FOOD_PATCH_RADIUS)
        return None

    def _scatter(self, x, y, sigma):
        w, h = self.bounds
        return (_fold(x + random.gauss(0, sigma), FOOD_RADIUS, w - FOOD_RADIUS),
                _fold(y + random.gauss(0, sigma), FOOD_RADIUS, h - FOOD_RADIUS))

    def _crowded(self, x, y):
        near = self.index.food.query(self.geometry.vec2(x, y), FOOD_REGROW_CROWD_RADIUS)
        limit = FOOD_REGROW_CROWD_RADIUS ** 2
        return sum((f.pos.x - x) ** 2 + (f.pos.y - y) ** 2 <= limit
                   for f in near) >= FOOD_REGROW_CROWD
