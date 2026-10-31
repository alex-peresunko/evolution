import random
import math
from typing import Optional

import numpy as np

from .config import (
    CARNIVORE_HEALTH, HERBIVORE_HEALTH, ADULTHOOD_AGE, JUVENILE_SIZE_FACTOR,
    CREATURE_ROTATION_SPEED, STAMINA_DEPLETION_SPEED_THRESHOLD, STAMINA_DEPLETION_RATE,
    STAMINA_REGEN_RATE, STAMINA_EXHAUSTION_PENALTY, HUNGER_MAX, HUNGER_HEALTH_PENALTY,
    HEALTH_LOSS_PER_TICK, HEALTH_LOSS_SPEED_FACTOR, HEALTH_LOST_ON_HIT,
    GENE_MIN_MAX, CARNIVORE_GENE_MIN_MAX, DEFAULT_HERBIVORE_GENES, DEFAULT_CARNIVORE_GENES,
    CARNIVORE_HEALTH_PER_FOOD, HERBIVORE_HEALTH_PER_FOOD, FOOD_RADIUS,
    HEALTH_GAIN_SIZE_PENALTY,
    NUM_WHISKERS, BRAIN_TOPOLOGY, SIZE_SPEED_PENALTY_FACTOR, SPEED_STAMINA_EXPONENT,
    SIZE_STAMINA_FACTOR, RESPAWN_IMMUNITY_TICKS,
    FOOD_MAX_LIFETIME, FOOD_MIN_HEALTH_FACTOR, NUM_MEMORY, CARNIVORE_BITE_ANGLE,
    BITE_DAMAGE, BITE_COOLDOWN_TICKS, BITE_HEALTH_TRANSFER, BITE_SCORE, ADRENALINE_STAMINA,
    RETALIATION_DAMAGE, CARCASS_MEAT_PER_SIZE, CARCASS_LIFETIME, CARCASS_EAT_RATE,
    SCAVENGE_SCORE_FACTOR, SIGNAL_HEALTH_COST, HERBIVORE_EAT_TICKS, CARNIVORE_EAT_TICKS,
    FAT_MAX, FAT_STORE_EFFICIENCY, FAT_BURN_PER_TICK, FAT_SPEED_PENALTY, FAT_SIZE_FACTOR,
    UPKEEP_WEIGHTS, KLEIBER_EXPONENT, SENESCENCE_RATE,
    WATER_SPEED_FACTOR, HERBIVORE_SWIM_STAMINA_COST, CARNIVORE_SWIM_STAMINA_COST,
    TREE_TRUNK_FRACTION,
)
from .math_types import Vector2 as _HeadlessVector2, SimpleRect as _HeadlessRect
from .neural_network import NeuralNetwork
from .perception import perceive_all


class Obstacle:
    def __init__(self, x, y, width, height, *, rect_factory=None):
        _rect = rect_factory or _HeadlessRect
        self.rect = _rect(x, y, width, height)


class Tree(Obstacle):
    """Only the trunk (`rect`, centred on x, y) is solid; the crown is drawn over creatures."""

    def __init__(self, x, y, crown_radius, *, rect_factory=None):
        side = crown_radius * TREE_TRUNK_FRACTION
        super().__init__(x - side / 2, y - side / 2, side, side, rect_factory=rect_factory)
        self.x, self.y, self.crown_radius = x, y, crown_radius


class Lake:
    """Shore radius at angle t is radius * (1 + sum of a * cos(k * t + phase) over
    `harmonics` (k, a, phase)); no harmonics is a circle."""

    def __init__(self, x, y, radius, harmonics=()):
        self.x, self.y, self.radius = x, y, radius
        self.harmonics = tuple(harmonics)
        self.max_radius = radius * (1 + sum(abs(a) for _, a, _ in self.harmonics))

    def shore(self, theta, cos=math.cos):
        """Shore radius at `theta`; pass cos=np.cos for an array of angles."""
        return self.radius * (1 + sum(a * cos(k * theta + p) for k, a, p in self.harmonics))

    def contains(self, x, y):
        dx, dy = x - self.x, y - self.y
        d2 = dx * dx + dy * dy
        return d2 < self.max_radius ** 2 and d2 < self.shore(math.atan2(dy, dx)) ** 2

    def shore_gap(self, xs, ys):
        """Distance past the shore along the ray from the centre (< 0 = in water), for arrays."""
        dx, dy = xs - self.x, ys - self.y
        return np.hypot(dx, dy) - self.shore(np.arctan2(dy, dx), np.cos)


class Creature:
    _id_counter = 1

    def __init__(self, world_bounds, brain=None, is_carnivore=False, genes=None,
                 *, vec2_factory=None, rect_factory=None):
        self._vec2 = vec2_factory or _HeadlessVector2
        self._rect = rect_factory or _HeadlessRect

        self.id = Creature._id_counter
        Creature._id_counter += 1
        self.world_bounds = world_bounds
        self.pos = self._vec2(random.uniform(0, world_bounds[0]), random.uniform(0, world_bounds[1]))
        self.angle = random.uniform(0, 2 * math.pi)
        self.is_carnivore = is_carnivore
        self.health = CARNIVORE_HEALTH if is_carnivore else HERBIVORE_HEALTH
        self.max_health = self.health
        self.score, self.is_best, self.reproductions = 0, False, 0
        self.birth_time: Optional[float] = None
        self.last_reproduction_tick: Optional[int] = None
        self.current_speed = 0.0
        self._nearest_target_dist: Optional[float] = None
        self._nearest_smell_strength: float = 0.0
        self.hunger = 0
        self.signal = 0.0                   # emitted to kin, from the brain's 3rd output
        self.memory = [0.0] * NUM_MEMORY    # fed back as inputs on the next tick
        self.effective_sight = 0.0
        self.last_bite_tick: Optional[int] = None
        self.last_hit_tick: Optional[int] = None
        self.respawn_immunity_ticks = 0
        self.eating_ticks = 0               # > 0 while standing still to eat
        self.meal_health = 0.0              # herbivore: health still to absorb from the meal
        self.meal_carcass = None            # carnivore: carcass being eaten
        self.fat = 0.0                      # surplus food; slows the creature, burned when starving
        self.brain_inputs = None            # last tick's brain input/output rows (brain panel)
        self.brain_outputs = None
        self.mate_override = False          # last tick's outputs were steered toward a mate
        self.in_water = False
        self.age = 0
        self.is_adult = False
        self.lineage = self.id              # founder's id, inherited from the first parent
        self.parent_ids = ()
        self.descent = 0                    # generations since the lineage's founder

        default_genes = DEFAULT_CARNIVORE_GENES if is_carnivore else DEFAULT_HERBIVORE_GENES
        self.genes = {**default_genes, **(genes or {})}   # older saves lack the newer genes

        gene_constraints = CARNIVORE_GENE_MIN_MAX if is_carnivore else GENE_MIN_MAX
        for key, value in self.genes.items():
            if key in gene_constraints:
                min_val, max_val = gene_constraints[key]
                self.genes[key] = float(np.clip(value, min_val, max_val))

        self.genetic_size = self.genes["size"]
        self.genetic_max_speed = self.genes["max_speed"]
        self.sight_distance = self.genes["sight_distance"]
        self.sight_angle = self.genes["sight_angle"]
        self.attractiveness = self.genes.get("attractiveness", 10)

        base_stamina = self.genes.get("max_stamina", 1000)
        self.max_stamina = base_stamina + (self.genetic_size * SIZE_STAMINA_FACTOR)
        self.stamina = self.max_stamina

        base_size = default_genes["size"]
        size_diff = self.genetic_size - base_size
        self.effective_max_speed = self.genetic_max_speed / (1 + max(0, size_diff * SIZE_SPEED_PENALTY_FACTOR))

        self.size = self.genetic_size * JUVENILE_SIZE_FACTOR
        self.max_speed = self.effective_max_speed * JUVENILE_SIZE_FACTOR

        self.trait_upkeep = self._trait_upkeep(default_genes)

        self.whiskers = [0.0] * NUM_WHISKERS
        self.target_whiskers = [None] * NUM_WHISKERS
        self.whisker_angles = np.linspace(-self.sight_angle / 2, self.sight_angle / 2, NUM_WHISKERS)
        self.brain = brain if brain else NeuralNetwork(BRAIN_TOPOLOGY)

    def _trait_upkeep(self, default_genes):
        """Non-body share of the upkeep multiplier: each costly trait relative to its default."""
        g, d = self.genes, default_genes
        ratios = {
            "sight": (g["sight_distance"] * g["sight_angle"]) / (d["sight_distance"] * d["sight_angle"]),
            "stamina": g["max_stamina"] / d["max_stamina"],
            "lifespan": g["lifespan"] / d["lifespan"],
            "attractiveness": g["attractiveness"] / d["attractiveness"],
        }
        return sum(UPKEEP_WEIGHTS[k] * r for k, r in ratios.items())

    def upkeep(self, growth_factor=1.0):
        """Per-tick health cost of staying alive: Kleiber-scaled lean body plus costly traits."""
        default_size = (DEFAULT_CARNIVORE_GENES if self.is_carnivore else DEFAULT_HERBIVORE_GENES)["size"]
        body = (self.genetic_size * growth_factor / default_size) ** KLEIBER_EXPONENT
        return HEALTH_LOSS_PER_TICK * (UPKEEP_WEIGHTS["body"] * body + self.trait_upkeep)

    def senescence(self):
        """Extra per-tick health loss of old age, growing linearly past the lifespan gene."""
        lifespan = self.genes["lifespan"]
        return SENESCENCE_RATE * max(0.0, self.age - lifespan) / lifespan

    def perceive_and_think(self, local_food, local_herbivores, local_carnivores, obstacles):
        inputs = self.see(local_food, obstacles, local_herbivores, local_carnivores)
        return self.think(inputs)

    def act(self, outputs, global_food_list, global_herbivore_list, obstacles, grid,
            food_grid=None, herb_grid=None, carcass_grid=None, tick=0, lakes=()):
        self.age += 1
        self.in_water = any(lake.contains(self.pos.x, self.pos.y) for lake in lakes)
        if self.respawn_immunity_ticks > 0:
            self.respawn_immunity_ticks -= 1
        if not self.is_adult and self.age >= ADULTHOOD_AGE:
            self.is_adult = True
        if self.is_adult:
            growth_factor = 1.0
        else:
            growth_ratio = self.age / ADULTHOOD_AGE
            growth_factor = JUVENILE_SIZE_FACTOR + (1 - JUVENILE_SIZE_FACTOR) * growth_ratio
        fat_ratio = self.fat / FAT_MAX
        self.size = self.genetic_size * growth_factor * (1 + fat_ratio * FAT_SIZE_FACTOR)
        self.max_speed = self.effective_max_speed * growth_factor

        if self.eating_ticks > 0:
            new_speed = 0.0
        else:
            self.angle = (self.angle + outputs[0] * CREATURE_ROTATION_SPEED) % (2 * math.pi)
            speed_multiplier = max(0.1, self.health / self.max_health)
            is_exhausted = self.stamina <= 0
            speed_penalty = STAMINA_EXHAUSTION_PENALTY if is_exhausted else 1.0
            weight_penalty = 1 / (1 + fat_ratio * FAT_SPEED_PENALTY)
            water_penalty = WATER_SPEED_FACTOR if self.in_water else 1.0
            new_speed = (max(0, outputs[1]) * self.max_speed * speed_multiplier * speed_penalty
                         * weight_penalty * water_penalty)

        if self.in_water:
            self.stamina -= (CARNIVORE_SWIM_STAMINA_COST if self.is_carnivore
                             else HERBIVORE_SWIM_STAMINA_COST)
        elif self.max_speed > 0 and new_speed / self.max_speed > STAMINA_DEPLETION_SPEED_THRESHOLD:
            speed_ratio = new_speed / self.max_speed
            depletion_amount = (speed_ratio ** SPEED_STAMINA_EXPONENT) * STAMINA_DEPLETION_RATE
            self.stamina -= depletion_amount
        else:
            self.stamina += STAMINA_REGEN_RATE
        self.stamina = max(0, min(self.stamina, self.max_stamina))

        heading = self._vec2(math.cos(self.angle), math.sin(self.angle))
        self.pos += heading * new_speed
        if self._outside_walls():  # world edges are walls, hit like an obstacle
            self.health -= HEALTH_LOST_ON_HIT
            self.pos -= heading * new_speed * 3
            self._clamp_to_walls()
        self.hunger += 1
        hunger_penalty = (
            HUNGER_HEALTH_PENALTY * min(1.0, max(0.0, (self.hunger - HUNGER_MAX) / HUNGER_MAX))
            if self.hunger > HUNGER_MAX else 0
        )
        self.fat = max(0.0, self.fat - FAT_BURN_PER_TICK)
        from_fat = min(self.fat, hunger_penalty)   # starvation is paid from fat first
        self.fat -= from_fat
        hunger_penalty -= from_fat
        self.health -= (self.upkeep(growth_factor) + self.senescence() + (new_speed * HEALTH_LOSS_SPEED_FACTOR) + hunger_penalty
                        + self.signal * SIGNAL_HEALTH_COST)
        self.current_speed = new_speed

        interaction_radius = self.size + GENE_MIN_MAX["size"][1]

        if self.is_carnivore:
            if self.eating_ticks > 0:
                self._continue_scavenging()
            elif self.respawn_immunity_ticks == 0:
                bit = (self.last_bite_tick is None or tick - self.last_bite_tick >= BITE_COOLDOWN_TICKS) \
                    and self._try_bite((herb_grid or grid).query(self.pos, interaction_radius), tick)
                if not bit and carcass_grid is not None:
                    self._scavenge(carcass_grid.query(self.pos, interaction_radius))
        elif self.eating_ticks > 0:
            self._digest()
        else:
            base_size = DEFAULT_HERBIVORE_GENES["size"]
            nearby_food = (food_grid or grid).query(self.pos, interaction_radius)
            for food in nearby_food:
                if food.consumed:
                    continue
                if self.pos.distance_to(food.pos) < (self.size + FOOD_RADIUS):
                    food.consumed = True
                    base_health_gain = food.get_health_value()
                    size_penalty_denom = 1 + (self.genetic_size - base_size) * HEALTH_GAIN_SIZE_PENALTY
                    gain = base_health_gain / max(0.1, size_penalty_denom)
                    if HERBIVORE_EAT_TICKS > 0:
                        self.meal_health, self.eating_ticks = gain, HERBIVORE_EAT_TICKS
                    else:
                        self._absorb(gain)
                    self.hunger = 0
                    self.score += 1
                    break

        creature_rect = self._rect(
            self.pos.x - self.size, self.pos.y - self.size, self.size * 2, self.size * 2
        )
        for obs in obstacles:
            if obs.rect.colliderect(creature_rect):
                self.health -= HEALTH_LOST_ON_HIT
                self.pos -= self._vec2(math.cos(self.angle), math.sin(self.angle)) * self.current_speed * 3
                break

        if self._nearest_target_dist is not None:
            self.score += 0.001 * (1.0 - self._nearest_target_dist)
        elif self.is_carnivore and self._nearest_smell_strength > 0:
            self.score += 0.0005 * self._nearest_smell_strength

    def _outside_walls(self):
        w, h = self.world_bounds
        return not (self.size <= self.pos.x <= w - self.size
                    and self.size <= self.pos.y <= h - self.size)

    def _clamp_to_walls(self):
        w, h = self.world_bounds
        self.pos.x = min(max(self.pos.x, self.size), w - self.size)
        self.pos.y = min(max(self.pos.y, self.size), h - self.size)

    def _digest(self):
        """One tick of a herbivore meal: absorb an even share of the remaining health."""
        portion = self.meal_health / self.eating_ticks
        self._absorb(portion)
        self.meal_health -= portion
        self.eating_ticks -= 1
        self.hunger = 0

    def _absorb(self, amount):
        """Add health up to max_health; the surplus is stored as fat. Returns the health gained."""
        gain = max(0.0, min(self.max_health - self.health, amount))
        self.health += gain
        self.fat = min(FAT_MAX, self.fat + (amount - gain) * FAT_STORE_EFFICIENCY)
        return gain

    def _is_stuffed(self):
        return self.health >= self.max_health and self.fat >= FAT_MAX

    def _stop_eating(self):
        self.eating_ticks, self.meal_health, self.meal_carcass = 0, 0.0, None

    def _gain_meat(self, amount):
        size_penalty_denom = 1 + (self.genetic_size - DEFAULT_CARNIVORE_GENES["size"]) * HEALTH_GAIN_SIZE_PENALTY
        gain = self._absorb(amount / max(0.1, size_penalty_denom))
        self.hunger = 0
        return gain

    def _try_bite(self, nearby_prey, tick):
        """Bite the first live prey in contact and in front; returns whether a bite landed."""
        for prey in nearby_prey:
            if prey.health <= 0 or self.pos.distance_to(prey.pos) >= self.size + prey.size:
                continue
            vec = prey.pos - self.pos
            facing = (math.atan2(vec.y, vec.x) - self.angle + math.pi) % (2 * math.pi) - math.pi
            if abs(facing) > CARNIVORE_BITE_ANGLE / 2:
                continue
            ratio = self.size / prey.size if prey.size > 0 else 2.0
            damage = BITE_DAMAGE * min(2.0, max(0.5, ratio))
            prey.health -= damage
            prey.stamina = min(prey.max_stamina, prey.stamina + ADRENALINE_STAMINA)
            prey.last_hit_tick = tick
            prey._stop_eating()   # a bite interrupts the meal; the rest of it is lost
            self._gain_meat(damage * BITE_HEALTH_TRANSFER)
            self.score += BITE_SCORE + (1 if prey.health <= 0 else 0)
            # Large prey fights back.
            self.health -= RETALIATION_DAMAGE * min(3.0, max(0.0, 1 / ratio - 1))
            self.last_bite_tick = tick
            self.last_hit_tick = tick
            return True
        return False

    def _scavenge(self, nearby_carcasses):
        for carcass in nearby_carcasses:
            if carcass.meat <= 0 or self.pos.distance_to(carcass.pos) >= self.size + carcass.radius:
                continue
            self._eat_from(carcass)
            if CARNIVORE_EAT_TICKS > 0 and not carcass.is_gone() and not self._is_stuffed():
                self.meal_carcass, self.eating_ticks = carcass, CARNIVORE_EAT_TICKS
            return

    def _continue_scavenging(self):
        """One tick of a carnivore meal; ends early once the carcass is gone or the carnivore is stuffed."""
        carcass = self.meal_carcass
        if carcass is None or carcass.is_gone() or self._is_stuffed():
            self._stop_eating()
            return
        self._eat_from(carcass)
        self.eating_ticks -= 1
        if self.eating_ticks == 0:
            self.meal_carcass = None

    def _eat_from(self, carcass):
        eaten = min(CARCASS_EAT_RATE, carcass.meat)
        carcass.meat -= eaten
        gain = self._gain_meat(eaten)
        self.score += gain / CARNIVORE_HEALTH_PER_FOOD * SCAVENGE_SCORE_FACTOR

    def see(self, food_items, obstacles, herbivores=None, carnivores=None, carcasses=(), light=1.0,
            season=0.0):
        return perceive_all([self], food_items, herbivores or [], carnivores or [], obstacles,
                            carcasses, light, season)[0].tolist()

    def think(self, inputs):
        return self.brain.forward(inputs)

    def is_alive(self):
        return self.health > 0

    def get_info(self):
        status = "Adult" if self.is_adult else f"Juvenile ({self.age}/{ADULTHOOD_AGE})"
        info = {
            "ID": self.id,
            "Type": "Carnivore" if self.is_carnivore else "Herbivore",
            "Status": status,
            "Health": int(self.health),
            "Stamina": f"{int(self.stamina)}/{int(self.max_stamina)}",
            "Swimming": self.in_water,
            "Score": round(self.score, 2),
            "Eating": self.eating_ticks,
            "Fat": f"{int(self.fat)}/{FAT_MAX}",
            "Upkeep": round(self.upkeep(), 2),
            "Signal": round(self.signal, 2),
            "Memory": " ".join(f"{m:+.2f}" for m in self.memory),
            "Reproductions": self.reproductions,
            "Lineage": f"#{self.lineage} (descent {self.descent})",
            "Is Best": self.is_best,
        }
        for gene, value in self.genes.items():
            info[f"Gene: {gene}"] = round(value, 2)
        return info


class Food:
    _id_counter = -1

    def __init__(self, world_bounds, *, vec2_factory=None, pos=None):
        self._vec2 = vec2_factory or _HeadlessVector2
        self.id = Food._id_counter
        Food._id_counter -= 1
        self.world_bounds = world_bounds
        if pos is None:
            pos = (random.uniform(0, world_bounds[0]), random.uniform(0, world_bounds[1]))
        self.pos = self._vec2(*pos)
        self.lifetime = 0
        self.max_lifetime = FOOD_MAX_LIFETIME
        self.health_value_base = HERBIVORE_HEALTH_PER_FOOD
        self.min_health_factor = FOOD_MIN_HEALTH_FACTOR
        self.consumed = False

    def update(self):
        self.lifetime += 1

    def is_rotten(self):
        return self.lifetime >= self.max_lifetime

    def get_health_value(self):
        if self.is_rotten():
            return 0
        freshness = 1.0 - (self.lifetime / self.max_lifetime)
        return int(self.health_value_base * (self.min_health_factor + (1 - self.min_health_factor) * freshness))


class Carcass:
    """Meat left where a creature died; carnivores scavenge it until it is eaten or rots."""
    _id_counter = -10**9

    def __init__(self, creature):
        self._vec2 = creature._vec2
        self.id = Carcass._id_counter
        Carcass._id_counter -= 1
        self.pos = self._vec2(creature.pos.x, creature.pos.y)
        self.radius = max(3.0, creature.size)
        self.is_carnivore = creature.is_carnivore
        self.max_meat = CARCASS_MEAT_PER_SIZE * creature.size
        self.meat = self.max_meat
        self.age = 0

    def update(self):
        self.age += 1

    def is_gone(self):
        return self.meat <= 0 or self.age >= CARCASS_LIFETIME
