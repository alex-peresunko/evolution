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
    HERD_COURAGE_MAX,
    MOB_ASSIST_SCORE,
    MOB_BONUS,
    MOB_WINDOW_TICKS,
    PACK_ASSIST_SCORE,
    PACK_BITE_BONUS,
    PACK_WINDOW_TICKS,
    RETALIATION_DAMAGE,
    SCAVENGE_SCORE_FACTOR,
    TRAMPLE_ANGLE,
    TRAMPLE_COOLDOWN_TICKS,
    TRAMPLE_DAMAGE,
    TRAMPLE_MAX_SIZE_RATIO,
    TRAMPLE_MAX_SPEED_RATIO,
    TRAMPLE_MIN_SPEED,
    TRAMPLE_REF_SPEED,
    TRAMPLE_SCORE,
    TRAMPLE_STUN_TICKS,
)

_MAX_SIZE = GENE_MIN_MAX["size"][1]


def _reach(c):
    """Query radius that finds everything the creature could touch."""
    return c.size + _MAX_SIZE + CONTACT_REACH


def _touching(c, other):
    return c.pos.distance_to(other.pos) < c.size + other.size + CONTACT_REACH


def _in_front(c, other, arc):
    """Whether other's centre lies within arc (radians, centred on c's heading)."""
    vec = other.pos - c.pos
    facing = (math.atan2(vec.y, vec.x) - c.angle + math.pi) % (2 * math.pi) - math.pi
    return abs(facing) <= arc / 2


def _herd_size(c, world):
    """Other live creatures of c's species within CONFUSION_RADIUS."""
    return sum(1 for other in world.index.of(c.species).query(c.pos, CONFUSION_RADIUS)
               if other is not c and other.health > 0
               and c.pos.distance_to(other.pos) < CONFUSION_RADIUS)


def _prune(recent, tick, window):
    for attacker, t in list(recent.items()):
        if tick - t > window:
            del recent[attacker]


class GrazingDiet:
    """Eats the first food item it touches, then digests it over the species' eat_ticks.
    Charging head-on into a predator tramples it."""

    tracks_by_smell = False

    def feed(self, c, world):
        if c.eating_ticks > 0:
            c.digest()
            return
        if (c.is_adult and c.current_speed >= TRAMPLE_MIN_SPEED
                and (c.last_trample_tick is None
                     or world.tick - c.last_trample_tick >= TRAMPLE_COOLDOWN_TICKS)):
            self._trample(c, world, world.tick)
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

    @staticmethod
    def _trample(c, world, tick):
        """Trample the first live predator in contact and in front; returns whether one was hit."""
        for species in world.populations:
            if not species.is_predator:
                continue
            for foe in world.index.of(species).query(c.pos, _reach(c)):
                if (foe.health <= 0 or not _touching(c, foe)
                        or not _in_front(c, foe, TRAMPLE_ANGLE)):
                    continue
                size_ratio = c.size / foe.size if foe.size > 0 else TRAMPLE_MAX_SIZE_RATIO
                damage = (TRAMPLE_DAMAGE
                          * min(TRAMPLE_MAX_SIZE_RATIO, max(1.0, size_ratio))
                          * min(TRAMPLE_MAX_SPEED_RATIO, c.current_speed / TRAMPLE_REF_SPEED)
                          * (1 + HERD_COURAGE_MAX * min(1.0, _herd_size(c, world)
                                                         / CONFUSION_HERD_NORM)))
                tramplers = foe.recent_tramplers
                _prune(tramplers, tick, MOB_WINDOW_TICKS)
                if any(other is not c for other in tramplers):
                    damage *= MOB_BONUS
                    world.tally.mob_tramples += 1
                tramplers[c] = tick
                world.tally.tramples += 1
                c.last_trample_tick = tick
                foe.health -= damage
                foe.hurt = 1.0
                foe.last_hit_tick = foe.last_trampled_tick = tick
                foe.stunned_ticks = TRAMPLE_STUN_TICKS
                foe.stop_eating()
                c.score += TRAMPLE_SCORE
                if foe.health <= 0:
                    c.score += 1
                    for other in tramplers:
                        if other is not c and other.health > 0:
                            other.score += MOB_ASSIST_SCORE
                return True
        return False


BITTEN, CONFUSED = "bitten", "confused"


def _confused(prey, world):
    """Whether the herd around prey makes the bite miss."""
    if CONFUSION_MAX_MISS <= 0:
        return False
    herd = _herd_size(prey, world)
    if herd == 0:
        return False
    return random.random() < CONFUSION_MAX_MISS * min(1.0, herd / CONFUSION_HERD_NORM)


def _attackers(c, prey, world):
    """Carnivores of c's species in contact with prey, c included."""
    others = sum(1 for other in world.index.of(c.species).query(prey.pos, _reach(prey))
                 if other is not c and other.health > 0 and _touching(prey, other))
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
            if c.stunned_ticks == 0 and (c.last_bite_tick is None
                                         or tick - c.last_bite_tick >= BITE_COOLDOWN_TICKS):
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
            if (prey.health <= 0 or not _touching(c, prey)
                    or not _in_front(c, prey, CARNIVORE_BITE_ANGLE)):
                continue
            c.last_bite_tick = tick
            if _confused(prey, world):
                return CONFUSED
            biters = prey.recent_biters
            _prune(biters, tick, PACK_WINDOW_TICKS)
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
