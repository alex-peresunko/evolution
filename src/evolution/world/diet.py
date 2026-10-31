"""How each species feeds during its tick: the strategy a Species plugs into Creature.act."""
import math

from ..config import (
    ADRENALINE_STAMINA,
    BITE_COOLDOWN_TICKS,
    BITE_DAMAGE,
    BITE_HEALTH_TRANSFER,
    BITE_SCORE,
    CARCASS_EAT_RATE,
    CARNIVORE_BITE_ANGLE,
    FOOD_RADIUS,
    GENE_MIN_MAX,
    RETALIATION_DAMAGE,
    SCAVENGE_SCORE_FACTOR,
)

_MAX_SIZE = GENE_MIN_MAX["size"][1]


def _reach(c):
    """Query radius that finds everything the creature could touch."""
    return c.size + _MAX_SIZE


class GrazingDiet:
    """Eats the first food item it touches, then digests it over the species' eat_ticks."""

    tracks_by_smell = False

    def feed(self, c, world):
        if c.eating_ticks > 0:
            c.digest()
            return
        for food in world.index.food.query(c.pos, _reach(c)):
            if food.consumed:
                continue
            if c.pos.distance_to(food.pos) < (c.size + FOOD_RADIUS):
                food.consumed = True
                world.tally.food_eaten += 1
                gain = c.size_adjusted_gain(food.health_value())
                if c.species.eat_ticks > 0:
                    c.meal_health, c.eating_ticks = gain, c.species.eat_ticks
                else:
                    c.absorb(gain)
                c.hunger = 0
                c.score += 1
                return


class PredatorDiet:
    """Bites live `prey`; when no bite lands, scavenges a carcass, eating for eat_ticks."""

    tracks_by_smell = True

    def __init__(self, prey):
        self.prey = prey

    def feed(self, c, world):
        if c.eating_ticks > 0:
            self._continue_meal(c)
        elif c.respawn_immunity_ticks == 0:
            tick = world.tick
            bit = ((c.last_bite_tick is None or tick - c.last_bite_tick >= BITE_COOLDOWN_TICKS)
                   and self._bite(c, world.index.of(self.prey).query(c.pos, _reach(c)), tick))
            if bit:
                world.tally.bites += 1
            else:
                self._scavenge(c, world.index.carcasses.query(c.pos, _reach(c)))

    @staticmethod
    def _bite(c, nearby_prey, tick):
        """Bite the first live prey in contact and in front; returns whether a bite landed."""
        for prey in nearby_prey:
            if prey.health <= 0 or c.pos.distance_to(prey.pos) >= c.size + prey.size:
                continue
            vec = prey.pos - c.pos
            facing = (math.atan2(vec.y, vec.x) - c.angle + math.pi) % (2 * math.pi) - math.pi
            if abs(facing) > CARNIVORE_BITE_ANGLE / 2:
                continue
            ratio = c.size / prey.size if prey.size > 0 else 2.0
            damage = BITE_DAMAGE * min(2.0, max(0.5, ratio))
            prey.health -= damage
            prey.stamina = min(prey.max_stamina, prey.stamina + ADRENALINE_STAMINA)
            prey.last_hit_tick = tick
            prey.stop_eating()   # the rest of the interrupted meal is lost
            c.gain_meat(damage * BITE_HEALTH_TRANSFER)
            c.score += BITE_SCORE + (1 if prey.health <= 0 else 0)
            # Large prey fights back.
            c.health -= RETALIATION_DAMAGE * min(3.0, max(0.0, 1 / ratio - 1))
            c.last_bite_tick = tick
            c.last_hit_tick = tick
            return True
        return False

    def _scavenge(self, c, nearby_carcasses):
        for carcass in nearby_carcasses:
            if carcass.meat <= 0 or c.pos.distance_to(carcass.pos) >= c.size + carcass.radius:
                continue
            self._eat_from(c, carcass)
            if c.species.eat_ticks > 0 and not carcass.is_gone() and not c.is_stuffed():
                c.meal_carcass, c.eating_ticks = carcass, c.species.eat_ticks
            return

    def _continue_meal(self, c):
        """Ends early once the carcass is gone or the carnivore is stuffed."""
        carcass = c.meal_carcass
        if carcass is None or carcass.is_gone() or c.is_stuffed():
            c.stop_eating()
            return
        self._eat_from(c, carcass)
        c.eating_ticks -= 1
        if c.eating_ticks == 0:
            c.meal_carcass = None

    @staticmethod
    def _eat_from(c, carcass):
        eaten = min(CARCASS_EAT_RATE, carcass.meat)
        carcass.meat -= eaten
        gain = c.gain_meat(eaten)
        c.score += gain / c.species.health_per_food * SCAVENGE_SCORE_FACTOR
