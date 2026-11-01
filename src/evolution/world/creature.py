import math
import random

import numpy as np

from ..brain.layout import OUTPUTS as OUT
from ..brain.network import NeuralNetwork
from ..config import (
    ADULTHOOD_AGE,
    BRAIN_TOPOLOGY,
    CREATURE_ROTATION_SPEED,
    FAT_BURN_PER_TICK,
    FAT_MAX,
    FAT_SIZE_FACTOR,
    FAT_SPEED_PENALTY,
    FAT_STORE_EFFICIENCY,
    HEALTH_GAIN_SIZE_PENALTY,
    HEALTH_LOSS_PER_TICK,
    HEALTH_LOSS_SPEED_FACTOR,
    HEALTH_LOST_ON_HIT,
    HUNGER_HEALTH_PENALTY,
    HUNGER_MAX,
    HURT_FADE_TICKS,
    JUVENILE_SIZE_FACTOR,
    KLEIBER_EXPONENT,
    MEAL_BREAK_THRESHOLD,
    NUM_MEMORY,
    NUM_WHISKERS,
    SENESCENCE_RATE,
    SIGNAL_HEALTH_COST,
    SIZE_SPEED_PENALTY_FACTOR,
    SIZE_STAMINA_FACTOR,
    SPEED_STAMINA_EXPONENT,
    STAMINA_DEPLETION_RATE,
    STAMINA_EFFORT_THRESHOLD,
    STAMINA_EXHAUSTION_PENALTY,
    STAMINA_RECOVERY_SHARE,
    STAMINA_REGEN_RATE,
    UPKEEP_WEIGHTS,
    WATER_SPEED_FACTOR,
)
from ..genetics import clamp_genes
from .geometry import HEADLESS

# attrition: upkeep, effort, signalling and wall or rock hits, with no other cause in effect.
DEATH_CAUSES = ("bitten", "retaliation", "trampled", "impact", "starved", "old_age", "attrition")


class Creature:
    _id_counter = 1

    def __init__(self, species, world_bounds, brain=None, genes=None, *, geometry=HEADLESS):
        self.species = species
        self.geometry = geometry
        self.id = Creature._id_counter
        Creature._id_counter += 1
        self.world_bounds = world_bounds
        self.pos = geometry.vec2(random.uniform(0, world_bounds[0]),
                                 random.uniform(0, world_bounds[1]))
        self.angle = random.uniform(0, 2 * math.pi)
        self.health = species.max_health
        self.max_health = self.health
        self.score, self.is_best, self.reproductions = 0, False, 0
        self.birth_time: float | None = None
        self.last_reproduction_tick: int | None = None
        self.current_speed = 0.0
        self.hunger = 0
        self.fat = 0.0                      # surplus food; slows the creature, burned when starving
        self.winded = False                 # out of stamina: slow, recovering whatever it asks
        self.in_water = False
        self.age = 0
        self.is_adult = False
        self.respawn_immunity_ticks = 0
        self.last_bite_tick: int | None = None
        self.last_hit_tick: int | None = None
        self.last_impact_tick: int | None = None
        self.knock = geometry.vec2(0, 0)    # knockback velocity from collisions, px/tick
        self.hurt = 0.0                     # 1 when bitten or hit back, fading to 0
        self.recent_biters = {}             # prey: carnivore -> tick of its last bite
        self.last_trample_tick: int | None = None
        self.last_trampled_tick: int | None = None
        self.recent_tramplers = {}          # predator: herbivore -> tick of its last trample
        self.stunned_ticks = 0              # > 0 after being trampled: no bites
        self.eating_ticks = 0               # > 0 while standing still to eat
        self.meal_health = 0.0              # grazer: health still to absorb from the meal
        self.meal_carcass = None            # predator: carcass being eaten
        self.lineage = self.id              # founder's id, inherited from the first parent
        self.parent_ids = ()
        self.descent = 0                    # generations since the lineage's founder
        self.generation = 1                 # parents' generation + 1, carried across extinctions
        # sim.breeding's memo: other id -> (own genes, other's genes, compatible); the genes
        # dicts are compared by identity, so reassigning genes invalidates the entry.
        self.compatibility = {}

        # Written by perception.perceive_all each tick.
        self.whiskers = [0.0] * NUM_WHISKERS
        self.target_whiskers = [None] * NUM_WHISKERS
        self.effective_sight = 0.0
        self.nearest_target_dist: float | None = None
        self.nearest_smell_strength = 0.0
        # Written by the simulation after the brain runs.
        self.signal = 0.0                   # heard by kin
        self.memory = [0.0] * NUM_MEMORY    # fed back as inputs on the next tick
        self.brain_inputs = None
        self.brain_outputs = None
        self.mate_override = False          # last tick's outputs were steered toward a mate

        self._set_genes(genes)
        self.stamina = self.max_stamina
        self.size = self.genetic_size * JUVENILE_SIZE_FACTOR
        self.max_speed = self.effective_max_speed * JUVENILE_SIZE_FACTOR
        self.brain = brain if brain else NeuralNetwork(BRAIN_TOPOLOGY)

    def _set_genes(self, genes):
        """genes (missing ones from the species defaults, clamped) and the traits derived from
        them."""
        default_genes = self.species.default_genes
        self.genes = clamp_genes({**default_genes, **(genes or {})}, self.species)

        self.genetic_size = self.genes["size"]
        self.genetic_max_speed = self.genes["max_speed"]
        self.sight_distance = self.genes["sight_distance"]
        self.sight_angle = self.genes["sight_angle"]
        self.attractiveness = self.genes.get("attractiveness", 10)

        base_stamina = self.genes.get("max_stamina", 1000)
        self.max_stamina = base_stamina + (self.genetic_size * SIZE_STAMINA_FACTOR)

        size_diff = self.genetic_size - default_genes["size"]
        self.effective_max_speed = self.genetic_max_speed / (
            1 + max(0, size_diff * SIZE_SPEED_PENALTY_FACTOR))

        self.trait_upkeep = self._trait_upkeep(default_genes)

        self.whisker_angles = np.linspace(-self.sight_angle / 2, self.sight_angle / 2,
                                          NUM_WHISKERS)

    def adopt(self, brain, genes):
        """Take another brain and genes mid-life; position, health, age, fat and the rest of
        the creature's state stay. `brain` must be this creature's own copy (BatchBrain)."""
        self.brain = brain
        self._set_genes(genes)
        self.stamina = min(self.stamina, self.max_stamina)
        self._shape_body(self._growth_factor())

    @property
    def is_carnivore(self):
        return self.species.is_predator

    def is_alive(self):
        return self.health > 0

    def death_cause(self, tick):
        """The most likely killer of a creature that died during `tick`, one of DEATH_CAUSES."""
        if self.last_trampled_tick == tick:
            return "trampled"
        if self.last_hit_tick == tick:
            return "retaliation" if self.species.is_predator else "bitten"
        if self.last_impact_tick == tick:
            return "impact"
        if self.hunger > HUNGER_MAX:
            return "starved"
        if self.age > self.genes["lifespan"]:
            return "old_age"
        return "attrition"

    # --- costs of living -------------------------------------------------------------------

    def _trait_upkeep(self, default_genes):
        """Non-body share of the upkeep multiplier: each costly trait relative to its default."""
        g, d = self.genes, default_genes
        ratios = {
            "sight": ((g["sight_distance"] * g["sight_angle"])
                      / (d["sight_distance"] * d["sight_angle"])),
            "stamina": g["max_stamina"] / d["max_stamina"],
            "lifespan": g["lifespan"] / d["lifespan"],
            "attractiveness": g["attractiveness"] / d["attractiveness"],
        }
        return sum(UPKEEP_WEIGHTS[k] * r for k, r in ratios.items())

    def upkeep(self, growth_factor=1.0):
        """Per-tick health cost of staying alive: Kleiber-scaled lean body plus costly traits."""
        default_size = self.species.default_genes["size"]
        body = (self.genetic_size * growth_factor / default_size) ** KLEIBER_EXPONENT
        return HEALTH_LOSS_PER_TICK * (UPKEEP_WEIGHTS["body"] * body + self.trait_upkeep)

    def senescence(self):
        """Extra per-tick health loss of old age, growing linearly past the lifespan gene."""
        lifespan = self.genes["lifespan"]
        return SENESCENCE_RATE * max(0.0, self.age - lifespan) / lifespan

    # --- one tick ----------------------------------------------------------------------------

    def act(self, outputs, world):
        """Live one tick on the brain's `outputs`. The steps' order is part of the rules."""
        self.hurt = max(0.0, self.hurt - 1 / HURT_FADE_TICKS)
        self.stunned_ticks = max(0, self.stunned_ticks - 1)
        growth_factor = self._grow(world.terrain.near(self.pos.x, self.pos.y)[1])
        if self.eating_ticks > 0 and outputs[OUT.stop_eating] > MEAL_BREAK_THRESHOLD:
            self.stop_eating()
            world.tally.meals_abandoned[self.species.name] += 1
        speed = self._move(outputs)
        self._metabolize(growth_factor, speed)
        self.current_speed = speed
        self.species.diet.feed(self, world)
        self._collide(world.obstacles if self.size > world.terrain.reach
                      else world.terrain.near(self.pos.x, self.pos.y)[0])
        self._reward_pursuit()

    def _grow(self, lakes):
        self.age += 1
        self.in_water = any(lake.contains(self.pos.x, self.pos.y) for lake in lakes)
        if self.respawn_immunity_ticks > 0:
            self.respawn_immunity_ticks -= 1
        if not self.is_adult and self.age >= ADULTHOOD_AGE:
            self.is_adult = True
        growth_factor = self._growth_factor()
        self._shape_body(growth_factor)
        return growth_factor

    def _growth_factor(self):
        if self.is_adult:
            return 1.0
        growth_ratio = self.age / ADULTHOOD_AGE
        return JUVENILE_SIZE_FACTOR + (1 - JUVENILE_SIZE_FACTOR) * growth_ratio

    def _shape_body(self, growth_factor):
        fat_ratio = self.fat / FAT_MAX
        self.size = self.genetic_size * growth_factor * (1 + fat_ratio * FAT_SIZE_FACTOR)
        self.max_speed = self.effective_max_speed * growth_factor

    def _move(self, outputs):
        """Turn, run and bounce off the walls on outputs (rotation, effort); returns the speed."""
        if self.stamina <= 0:
            self.winded = True
        elif self.winded and self.stamina >= STAMINA_RECOVERY_SHARE * self.max_stamina:
            self.winded = False
        effort = 0.0
        if self.eating_ticks > 0:
            speed = 0.0
        else:
            effort = max(0.0, outputs[OUT.speed])
            self.angle = ((self.angle + outputs[OUT.rotation] * CREATURE_ROTATION_SPEED)
                          % (2 * math.pi))
            speed_multiplier = max(0.1, self.health / self.max_health)
            speed_penalty = STAMINA_EXHAUSTION_PENALTY if self.winded else 1.0
            weight_penalty = 1 / (1 + self.fat / FAT_MAX * FAT_SPEED_PENALTY)
            water_penalty = WATER_SPEED_FACTOR if self.in_water else 1.0
            speed = (effort * self.max_speed * speed_multiplier * speed_penalty
                     * weight_penalty * water_penalty)

        if self.in_water:
            self.stamina -= self.species.swim_stamina_cost
        elif effort > STAMINA_EFFORT_THRESHOLD and not self.winded:
            self.stamina -= (effort ** SPEED_STAMINA_EXPONENT) * STAMINA_DEPLETION_RATE
        else:
            self.stamina += STAMINA_REGEN_RATE
        self.stamina = max(0, min(self.stamina, self.max_stamina))

        heading = self.geometry.vec2(math.cos(self.angle), math.sin(self.angle))
        self.pos += heading * speed
        if self._outside_walls():
            self.health -= HEALTH_LOST_ON_HIT
            self.pos -= heading * speed * 3
            self.clamp_to_walls()
        return speed

    def _metabolize(self, growth_factor, speed):
        self.hunger += 1
        hunger_penalty = (
            HUNGER_HEALTH_PENALTY * min(1.0, max(0.0, (self.hunger - HUNGER_MAX) / HUNGER_MAX))
            if self.hunger > HUNGER_MAX else 0
        )
        self.fat = max(0.0, self.fat - FAT_BURN_PER_TICK)
        from_fat = min(self.fat, hunger_penalty)   # starvation is paid from fat first
        self.fat -= from_fat
        hunger_penalty -= from_fat
        self.health -= (self.upkeep(growth_factor) + self.senescence()
                        + (speed * HEALTH_LOSS_SPEED_FACTOR) + hunger_penalty
                        + self.signal * SIGNAL_HEALTH_COST)

    def _collide(self, obstacles):
        body = self.geometry.rect(self.pos.x - self.size, self.pos.y - self.size,
                                  self.size * 2, self.size * 2)
        for obs in obstacles:
            if obs.rect.colliderect(body):
                self.health -= HEALTH_LOST_ON_HIT
                self.pos -= (self.geometry.vec2(math.cos(self.angle), math.sin(self.angle))
                             * self.current_speed * 3)
                break

    def _reward_pursuit(self):
        if self.nearest_target_dist is not None:
            self.score += 0.001 * (1.0 - self.nearest_target_dist)
        elif self.species.diet.tracks_by_smell and self.nearest_smell_strength > 0:
            self.score += 0.0005 * self.nearest_smell_strength

    def _outside_walls(self):
        w, h = self.world_bounds
        return not (self.size <= self.pos.x <= w - self.size
                    and self.size <= self.pos.y <= h - self.size)

    def clamp_to_walls(self):
        w, h = self.world_bounds
        self.pos.x = min(max(self.pos.x, self.size), w - self.size)
        self.pos.y = min(max(self.pos.y, self.size), h - self.size)

    # --- eating, used by the species' diet -------------------------------------------------

    def size_adjusted_gain(self, amount):
        """Bodies above the species' default size get less out of the same food."""
        penalty = 1 + (self.genetic_size - self.species.default_genes["size"]) * \
            HEALTH_GAIN_SIZE_PENALTY
        return amount / max(0.1, penalty)

    def absorb(self, amount):
        """Add health up to max_health; the surplus is stored as fat. Returns the health gained."""
        gain = max(0.0, min(self.max_health - self.health, amount))
        self.health += gain
        self.fat = min(FAT_MAX, self.fat + (amount - gain) * FAT_STORE_EFFICIENCY)
        return gain

    def gain_meat(self, amount):
        gain = self.absorb(self.size_adjusted_gain(amount))
        self.hunger = 0
        return gain

    def digest(self):
        """One tick of a grazing meal: absorb an even share of the remaining health."""
        portion = self.meal_health / self.eating_ticks
        self.absorb(portion)
        self.meal_health -= portion
        self.eating_ticks -= 1
        self.hunger = 0

    def is_stuffed(self):
        return self.health >= self.max_health and self.fat >= FAT_MAX

    def stop_eating(self):
        self.eating_ticks, self.meal_health, self.meal_carcass = 0, 0.0, None
