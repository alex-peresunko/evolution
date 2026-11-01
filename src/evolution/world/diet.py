"""How each species feeds during its tick: the strategy a Species plugs into Creature.act."""
import math
import random

from ..config import (
    ADRENALINE_STAMINA,
    BITE_COOLDOWN_TICKS,
    BITE_DAMAGE,
    BITE_HEALTH_TRANSFER,
    BITE_SCORE,
    CARCASS_EAT_RATE,
    CARNIVORE_BITE_ANGLE,
    CONFUSION_HERD_NORM,
    CONFUSION_MAX_MISS,
    CONFUSION_RADIUS,
    CONTACT_REACH,
    FOOD_RADIUS,
    GENE_MIN_MAX,
    PACK_ASSIST_SCORE,
    PACK_BITE_BONUS,
    PACK_WINDOW_TICKS,
    RETALIATION_DAMAGE,
    SCAVENGE_SCORE_FACTOR,
)

_MAX_SIZE = GENE_MIN_MAX["size"][1]


def _reach(c):
    """Query radius that finds everything the creature could touch."""
    return c.size + _MAX_SIZE + CONTACT_REACH


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


BITTEN, CONFUSED = "bitten", "confused"


def _confused(prey, world):
    """Whether the herd around prey makes the bite miss."""
    if CONFUSION_MAX_MISS <= 0:
        return False
    herd = sum(1 for other in world.index.of(prey.species).query(prey.pos, CONFUSION_RADIUS)
               if other is not prey and other.health > 0
               and prey.pos.distance_to(other.pos) < CONFUSION_RADIUS)
    if herd == 0:
        return False
    return random.random() < CONFUSION_MAX_MISS * min(1.0, herd / CONFUSION_HERD_NORM)


def _attackers(c, prey, world):
    """Carnivores of c's species in contact with prey, c included."""
    others = sum(1 for other in world.index.of(c.species).query(prey.pos, _reach(prey))
                 if other is not c and other.health > 0
                 and prey.pos.distance_to(other.pos) < other.size + prey.size + CONTACT_REACH)
    return 1 + others


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
            outcome = None
            if c.last_bite_tick is None or tick - c.last_bite_tick >= BITE_COOLDOWN_TICKS:
                outcome = self._bite(c, world, tick)
            if outcome == BITTEN:
                world.tally.bites += 1
            elif outcome == CONFUSED:
                world.tally.bites_confused += 1
            else:
                self._scavenge(c, world.index.carcasses.query(c.pos, _reach(c)))

    def _bite(self, c, world, tick):
        """Bite the first live prey in contact and in front. Returns BITTEN, CONFUSED (the herd
        around the prey made the bite miss; the cooldown is still spent) or None."""
        for prey in world.index.of(self.prey).query(c.pos, _reach(c)):
            if (prey.health <= 0
                    or c.pos.distance_to(prey.pos) >= c.size + prey.size + CONTACT_REACH):
                continue
            vec = prey.pos - c.pos
            facing = (math.atan2(vec.y, vec.x) - c.angle + math.pi) % (2 * math.pi) - math.pi
            if abs(facing) > CARNIVORE_BITE_ANGLE / 2:
                continue
            c.last_bite_tick = tick
            if _confused(prey, world):
                return CONFUSED
            biters = prey.recent_biters
            for biter, t in list(biters.items()):
                if tick - t > PACK_WINDOW_TICKS:
                    del biters[biter]
            ratio = c.size / prey.size if prey.size > 0 else 2.0
            damage = BITE_DAMAGE * min(2.0, max(0.5, ratio))
            if any(biter is not c for biter in biters):
                damage *= PACK_BITE_BONUS
                world.tally.pack_bites += 1
            biters[c] = tick
            prey.health -= damage
            prey.stamina = min(prey.max_stamina, prey.stamina + ADRENALINE_STAMINA)
            prey.last_hit_tick = tick
            prey.hurt = 1.0
            prey.stop_eating()   # the rest of the interrupted meal is lost
            c.gain_meat(damage * BITE_HEALTH_TRANSFER)
            c.score += BITE_SCORE
            if prey.health <= 0:
                c.score += 1
                for biter in biters:
                    if biter is not c and biter.health > 0:
                        biter.score += PACK_ASSIST_SCORE
            # Large prey fights back, against everyone attacking it.
            retaliation = RETALIATION_DAMAGE * min(3.0, max(0.0, 1 / ratio - 1))
            if retaliation > 0:
                c.health -= retaliation / _attackers(c, prey, world)
                c.hurt = 1.0
            c.last_hit_tick = tick
            return BITTEN
        return None

    def _scavenge(self, c, nearby_carcasses):
        for carcass in nearby_carcasses:
            if (carcass.meat <= 0
                    or c.pos.distance_to(carcass.pos) >= c.size + carcass.radius + CONTACT_REACH):
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
