"""What creatures eat: plant food for herbivores, carcasses for carnivores."""
import operator
import random

import numpy as np

from ..config import (
    CARCASS_FRESH_TICKS,
    CARCASS_LIFETIME,
    CARCASS_MEAT_PER_SIZE,
    FOOD_MAX_LIFETIME,
    FOOD_MIN_HEALTH_FACTOR,
    HERBIVORE_HEALTH_PER_FOOD,
)
from .geometry import HEADLESS


class FoodClock:
    """A world's food age clock. Food attached to it ages by one tick with every `now += 1`,
    and food eaten while attached is listed in `eaten` until the world collects it."""

    def __init__(self):
        self.now = 0
        self.eaten = []


class Food:
    _id_counter = -1

    def __init__(self, world_bounds, *, geometry=HEADLESS, pos=None):
        self.id = Food._id_counter
        Food._id_counter -= 1
        if pos is None:
            pos = (random.uniform(0, world_bounds[0]), random.uniform(0, world_bounds[1]))
        self.pos = geometry.vec2(*pos)
        # Detached (no clock), _born holds the lifetime; attached, the clock tick it was born at.
        self._clock = None
        self._born = 0
        self._consumed = False

    @property
    def lifetime(self):
        clock = self._clock
        return self._born if clock is None else clock.now - self._born

    @lifetime.setter
    def lifetime(self, value):
        clock = self._clock
        self._born = value if clock is None else clock.now - value

    @property
    def consumed(self):
        return self._consumed

    @consumed.setter
    def consumed(self, value):
        self._consumed = value
        if value and self._clock is not None:
            self._clock.eaten.append(self)

    def attach(self, clock):
        """Age with `clock` from now on, keeping the current lifetime."""
        lifetime = self.lifetime
        self._clock = clock
        self._born = clock.now - lifetime
        if self._consumed:
            clock.eaten.append(self)

    def detach(self):
        """Stop ageing, keeping the current lifetime."""
        self._born, self._clock = self.lifetime, None

    def update(self):
        self.lifetime += 1

    def is_rotten(self):
        return self.lifetime >= FOOD_MAX_LIFETIME

    def health_value(self):
        """Health it gives, falling from HERBIVORE_HEALTH_PER_FOOD as it ages; 0 once rotten."""
        if self.is_rotten():
            return 0
        freshness = 1.0 - (self.lifetime / FOOD_MAX_LIFETIME)
        return int(HERBIVORE_HEALTH_PER_FOOD
                   * (FOOD_MIN_HEALTH_FACTOR + (1 - FOOD_MIN_HEALTH_FACTOR) * freshness))


class Carcass:
    """Meat left where a creature died; carnivores scavenge it until it is eaten or rots."""

    _id_counter = -10**9

    def __init__(self, creature):
        self.id = Carcass._id_counter
        Carcass._id_counter -= 1
        self.pos = creature.geometry.vec2(creature.pos.x, creature.pos.y)
        self.radius = max(3.0, creature.size)
        self.species = creature.species
        self.angle = creature.angle
        self.color = creature.genes["color"]
        self.max_meat = CARCASS_MEAT_PER_SIZE * creature.size
        self.meat = self.max_meat
        self.age = 0

    def update(self):
        self.age += 1
        if self.age > CARCASS_FRESH_TICKS:
            self.meat = max(0.0, self.meat
                            - self.max_meat / (CARCASS_LIFETIME - CARCASS_FRESH_TICKS))

    def is_gone(self):
        return self.meat <= 0 or self.age >= CARCASS_LIFETIME


class _CarcassColumns:
    def __init__(self):
        self._members = ()
        self._columns = np.zeros((0, 4))

    def of(self, carcasses):
        members = self._members
        if len(carcasses) != len(members) or not all(map(operator.is_, carcasses, members)):
            self._members = list(carcasses)
            self._columns = np.array([(k.pos.x, k.pos.y, k.radius, k.id) for k in carcasses],
                                     dtype=np.float64).reshape(-1, 4)
        return self._columns


_carcass_columns = _CarcassColumns()


def carcass_columns(carcasses):
    """(C, 4) read-only array of the x, y, radius and id of `carcasses`, which never change
    during a carcass's life; kept until the list's members change."""
    return _carcass_columns.of(carcasses)
