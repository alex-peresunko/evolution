import random

import numpy as np

from ..config import (
    FAT_SIZE_FACTOR,
    FOOD_MAX_LIFETIME,
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
from . import creature_table as T
from .collision import separate
from .creature_table import CreatureTable
from .environment import FoodPatch, day_light, season
from .food import Food, FoodClock
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
        # The creatures' state, arranged in creatures() order by perceive (world.creature_table).
        self.table = CreatureTable()
        # Food ages by the clock: the items of _food_list (the food list as last synced) are
        # attached to it, with their birth ticks and ids in arrays in list order.
        self._food_clock = FoodClock()
        self._food_list = None
        self._food_last = None
        self._food_born = np.zeros(0, dtype=np.int64)
        self._food_ids = np.zeros(0, dtype=np.int64)

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
        creatures = self.creatures()
        self.table.arrange(creatures)
        xy = self.table.fl[:, (T.X, T.Y)]
        positions, start = {}, 0
        for s in ALL_SPECIES:
            end = start + len(self.populations[s])
            positions[s], start = xy[start:end], end
        self.index.rebuild(self.food, self.populations, self.carcasses, positions)

    def perceive(self):
        """Brain input rows for creatures() (see perception.perceive_all)."""
        creatures = self.creatures()
        self.table.arrange(creatures)
        return perceive_all(creatures, self.food, self.herbivores, self.carnivores,
                            self.obstacles, self.carcasses, self.light, self.season, self.lakes,
                            self.tick, food_xy=self.index.food_xy(self.food))

    def separate(self):
        separate(self.creatures(), self.carcasses, self.obstacles, self.tick, self.terrain)

    # --- food ------------------------------------------------------------------------------

    def _sync_food_clock(self):
        """Attach the items appended to self.food since the last sync, or every item after
        any other change to the list."""
        food, clock = self.food, self._food_clock
        n = len(self._food_born)
        if (food is not self._food_list or len(food) < n
                or (n and food[n - 1] is not self._food_last)):
            clock.eaten = []
            new, born, ids = food, None, None
        else:
            new, born, ids = food[n:], self._food_born, self._food_ids
        if new or born is None:
            for item in new:
                item.attach(clock)
            new_born = np.array([f._born for f in new], dtype=np.int64)
            new_ids = np.array([f.id for f in new], dtype=np.int64)
            self._food_born = new_born if born is None else np.concatenate((born, new_born))
            self._food_ids = new_ids if ids is None else np.concatenate((ids, new_ids))
        self._food_list = food
        self._food_last = food[-1] if food else None

    def food_lifetimes(self):
        """(F,) float lifetimes of the items of self.food."""
        self._sync_food_clock()
        return (self._food_clock.now - self._food_born).astype(np.float64)

    def age_food(self):
        """Age every food item by a tick and drop the eaten and the rotten ones."""
        self._sync_food_clock()
        food, clock = self.food, self._food_clock
        clock.now += 1
        gone = self._food_born <= clock.now - FOOD_MAX_LIFETIME
        eaten, clock.eaten = clock.eaten, []
        if eaten:
            gone |= np.isin(self._food_ids, np.array([f.id for f in eaten], dtype=np.int64))
        gone = np.flatnonzero(gone)
        if len(gone):
            self._food_born = np.delete(self._food_born, gone)
            self._food_ids = np.delete(self._food_ids, gone)
            gone = gone.tolist()
            items = [food[i] for i in gone]
            self.tally.food_rotted += sum(not item.consumed for item in items)
            for i in reversed(gone):
                del food[i]
            for item in items:
                item.detach()
            self._food_last = food[-1] if food else None
            self.index.food_removed(food, gone, items)

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
