"""What creatures eat: plant food for herbivores, carcasses for carnivores."""
import random

from ..config import (
    CARCASS_LIFETIME,
    CARCASS_MEAT_PER_SIZE,
    FOOD_MAX_LIFETIME,
    FOOD_MIN_HEALTH_FACTOR,
    HERBIVORE_HEALTH_PER_FOOD,
)
from .geometry import HEADLESS


class Food:
    _id_counter = -1

    def __init__(self, world_bounds, *, geometry=HEADLESS, pos=None):
        self.id = Food._id_counter
        Food._id_counter -= 1
        if pos is None:
            pos = (random.uniform(0, world_bounds[0]), random.uniform(0, world_bounds[1]))
        self.pos = geometry.vec2(*pos)
        self.lifetime = 0
        self.consumed = False

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

    def is_gone(self):
        return self.meat <= 0 or self.age >= CARCASS_LIFETIME
