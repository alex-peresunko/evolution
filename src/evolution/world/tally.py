"""Running counts of world events, read by the metrics."""
from collections import Counter
from dataclasses import dataclass, field


@dataclass
class WorldTally:
    bites: int = 0
    food_eaten: int = 0
    food_rotted: int = 0
    carcasses_eaten: int = 0
    carcasses_rotted: int = 0
    meals_abandoned: Counter = field(default_factory=Counter)  # by species name
