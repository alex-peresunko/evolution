"""Running counts of world events, read by the metrics."""
from dataclasses import dataclass


@dataclass
class WorldTally:
    bites: int = 0
    food_eaten: int = 0
    food_rotted: int = 0
    carcasses_eaten: int = 0
    carcasses_rotted: int = 0
