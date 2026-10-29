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
    HEALTH_GAIN_SIZE_PENALTY, SMELL_DISTANCE, MAX_NORMALIZED_SIGHT_DISTANCE,
    NUM_WHISKERS, BRAIN_TOPOLOGY, SIZE_SPEED_PENALTY_FACTOR, SPEED_STAMINA_EXPONENT,
    SIZE_STAMINA_FACTOR, RESPAWN_IMMUNITY_TICKS,
    FOOD_MAX_LIFETIME, FOOD_MIN_HEALTH_FACTOR,
)
from .math_types import Vector2 as _HeadlessVector2, SimpleRect as _HeadlessRect
from .neural_network import NeuralNetwork


class Obstacle:
    def __init__(self, x, y, width, height, *, rect_factory=None):
        _rect = rect_factory or _HeadlessRect
        self.rect = _rect(x, y, width, height)


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
        self.last_reproduction_time: Optional[float] = None
        self.current_speed = 0.0
        self._nearest_target_dist: Optional[float] = None
        self._nearest_smell_strength: float = 0.0
        self.dead = False
        self.hunger = 0
        self.respawn_immunity_ticks = 0
        self.age = 0
        self.is_adult = False

        if genes:
            self.genes = genes
        else:
            self.genes = DEFAULT_CARNIVORE_GENES.copy() if is_carnivore else DEFAULT_HERBIVORE_GENES.copy()

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

        base_size = DEFAULT_CARNIVORE_GENES["size"] if is_carnivore else DEFAULT_HERBIVORE_GENES["size"]
        size_diff = self.genetic_size - base_size
        self.effective_max_speed = self.genetic_max_speed / (1 + max(0, size_diff * SIZE_SPEED_PENALTY_FACTOR))

        self.size = self.genetic_size * JUVENILE_SIZE_FACTOR
        self.max_speed = self.effective_max_speed * JUVENILE_SIZE_FACTOR

        self.whiskers = [0.0] * NUM_WHISKERS
        self.whisker_angles = np.linspace(-self.sight_angle / 2, self.sight_angle / 2, NUM_WHISKERS)
        self.brain = brain if brain else NeuralNetwork(BRAIN_TOPOLOGY)

    def perceive_and_think(self, local_food, local_herbivores, local_carnivores, obstacles):
        inputs = self.see(local_food, obstacles, local_herbivores, local_carnivores)
        return self.think(inputs)

    def act(self, outputs, global_food_list, global_herbivore_list, obstacles, grid,
            food_grid=None, herb_grid=None):
        self.age += 1
        if self.respawn_immunity_ticks > 0:
            self.respawn_immunity_ticks -= 1
        if not self.is_adult:
            if self.age >= ADULTHOOD_AGE:
                self.is_adult = True
                self.size = self.genetic_size
                self.max_speed = self.effective_max_speed
            else:
                growth_ratio = self.age / ADULTHOOD_AGE
                growth_factor = JUVENILE_SIZE_FACTOR + (1 - JUVENILE_SIZE_FACTOR) * growth_ratio
                self.size = self.genetic_size * growth_factor
                self.max_speed = self.effective_max_speed * growth_factor

        self.angle = (self.angle + outputs[0] * CREATURE_ROTATION_SPEED) % (2 * math.pi)
        speed_multiplier = max(0.1, self.health / self.max_health)
        is_exhausted = self.stamina <= 0
        speed_penalty = STAMINA_EXHAUSTION_PENALTY if is_exhausted else 1.0
        new_speed = max(0, outputs[1]) * self.max_speed * speed_multiplier * speed_penalty

        if self.max_speed > 0 and new_speed / self.max_speed > STAMINA_DEPLETION_SPEED_THRESHOLD:
            speed_ratio = new_speed / self.max_speed
            depletion_amount = (speed_ratio ** SPEED_STAMINA_EXPONENT) * STAMINA_DEPLETION_RATE
            self.stamina -= depletion_amount
        else:
            self.stamina += STAMINA_REGEN_RATE
        self.stamina = max(0, min(self.stamina, self.max_stamina))

        self.pos += self._vec2(math.cos(self.angle), math.sin(self.angle)) * new_speed
        self.pos.x %= self.world_bounds[0]
        self.pos.y %= self.world_bounds[1]
        self.hunger += 1
        hunger_penalty = (
            HUNGER_HEALTH_PENALTY * min(1.0, max(0.0, (self.hunger - HUNGER_MAX) / HUNGER_MAX))
            if self.hunger > HUNGER_MAX else 0
        )
        self.health -= HEALTH_LOSS_PER_TICK + (new_speed * HEALTH_LOSS_SPEED_FACTOR) + hunger_penalty
        self.current_speed = new_speed

        interaction_radius = self.size + GENE_MIN_MAX["size"][1]

        if self.is_carnivore and self.respawn_immunity_ticks == 0:
            base_size = DEFAULT_CARNIVORE_GENES["size"]
            nearby_prey = (herb_grid or grid).query(self.pos, interaction_radius)
            for prey in nearby_prey:
                if prey.dead:
                    continue
                if self.pos.distance_to(prey.pos) < (self.size + prey.size):
                    prey.dead = True
                    base_health_gain = CARNIVORE_HEALTH_PER_FOOD
                    size_penalty_denom = 1 + (self.genetic_size - base_size) * HEALTH_GAIN_SIZE_PENALTY
                    self.health = min(self.max_health, self.health + base_health_gain / max(0.1, size_penalty_denom))
                    self.hunger = 0
                    self.score += 1
                    break
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
                    self.health = min(self.max_health, self.health + base_health_gain / max(0.1, size_penalty_denom))
                    self.hunger = 0
                    self.score += 1
                    break

        for obs in obstacles:
            creature_rect = self._rect(
                self.pos.x - self.size, self.pos.y - self.size, self.size * 2, self.size * 2
            )
            if obs.rect.colliderect(creature_rect):
                self.health -= HEALTH_LOST_ON_HIT
                self.pos -= self._vec2(math.cos(self.angle), math.sin(self.angle)) * self.current_speed * 3
                break

        if self._nearest_target_dist is not None:
            self.score += 0.001 * (1.0 - self._nearest_target_dist)
        elif self.is_carnivore and self._nearest_smell_strength > 0:
            self.score += 0.0005 * self._nearest_smell_strength

    def see(self, food_items, obstacles, herbivores=None, carnivores=None):
        inputs = []
        for i, whisker_angle in enumerate(self.whisker_angles):
            end_point = self.pos + self._vec2(
                math.cos(self.angle + whisker_angle),
                math.sin(self.angle + whisker_angle),
            ) * self.sight_distance
            closest_dist = self.sight_distance
            for obs in obstacles:
                clipped_line = obs.rect.clipline(self.pos, end_point)
                if clipped_line:
                    closest_dist = min(closest_dist, self.pos.distance_to(clipped_line[0]))
            self.whiskers[i] = closest_dist
            inputs.append(1.0 - (closest_dist / self.sight_distance if self.sight_distance > 0 else 1))

        def in_sight(target):
            if target.id == self.id:
                return False
            vec = target.pos - self.pos
            if 0 < vec.length() <= self.sight_distance:
                angle_to_target = (math.atan2(vec.y, vec.x) - self.angle + math.pi) % (2 * math.pi) - math.pi
                return abs(angle_to_target) <= self.sight_angle / 2
            return False

        nearest_prey = None
        if self.is_carnivore:
            visible_prey = [h for h in (herbivores or []) if in_sight(h)]
            if visible_prey:
                nearest_prey = min(visible_prey, key=lambda h: self.pos.distance_to(h.pos))
                target_info = self.get_target_info(nearest_prey)
            else:
                target_info = (0, 1)
            self._nearest_target_dist = target_info[1] if visible_prey else None
            inputs.extend(target_info)
        else:
            visible_food = [f for f in food_items if in_sight(f)]
            food_angle, food_dist = (
                self.get_target_info(min(visible_food, key=lambda f: self.pos.distance_to(f.pos)))
                if visible_food else (0, 1)
            )
            self._nearest_target_dist = food_dist if visible_food else None
            visible_preds = [c for c in (carnivores or []) if in_sight(c)]
            predator_angle, predator_dist = (
                self.get_target_info(min(visible_preds, key=lambda c: self.pos.distance_to(c.pos)))
                if visible_preds else (0, 1)
            )
            avoid_angle = food_angle - predator_angle * (1 - predator_dist) ** 2
            inputs.extend([np.clip(avoid_angle, -1, 1), food_dist])

        smell_angle, smell_strength = 0.0, 0.0
        self._nearest_smell_strength = 0.0
        if self.is_carnivore:
            smellable_prey = [h for h in (herbivores or []) if h.id != self.id]
            if smellable_prey:
                closest_prey = min(smellable_prey, key=lambda p: self.pos.distance_to(p.pos))
                dist = self.pos.distance_to(closest_prey.pos)
                if dist < SMELL_DISTANCE:
                    angle = (
                        math.atan2(closest_prey.pos.y - self.pos.y, closest_prey.pos.x - self.pos.x)
                        - self.angle + math.pi
                    ) % (2 * math.pi) - math.pi
                    smell_angle = angle / math.pi
                    smell_strength = 1.0 - (dist / SMELL_DISTANCE)
                    self._nearest_smell_strength = smell_strength
        else:
            smellable_predators = [c for c in (carnivores or []) if c.id != self.id]
            if smellable_predators:
                closest_predator = min(smellable_predators, key=lambda p: self.pos.distance_to(p.pos))
                dist = self.pos.distance_to(closest_predator.pos)
                if dist < SMELL_DISTANCE:
                    angle = (
                        math.atan2(closest_predator.pos.y - self.pos.y, closest_predator.pos.x - self.pos.x)
                        - self.angle + math.pi
                    ) % (2 * math.pi) - math.pi
                    smell_angle = angle / math.pi
                    smell_strength = 1.0 - (dist / SMELL_DISTANCE)
        inputs.extend([smell_angle, smell_strength])

        pvx, pvy = 0.0, 0.0
        if nearest_prey is not None:
            spd_norm = nearest_prey.current_speed / nearest_prey.max_speed if nearest_prey.max_speed > 0 else 0
            pvx = math.cos(nearest_prey.angle) * spd_norm
            pvy = math.sin(nearest_prey.angle) * spd_norm
        inputs.extend([pvx, pvy])

        inputs.append(self.health / self.max_health)
        inputs.append(math.sin(self.angle))
        inputs.append(math.cos(self.angle))
        inputs.append(self.size / GENE_MIN_MAX["size"][1])
        inputs.append(self.current_speed / self.genetic_max_speed if self.genetic_max_speed > 0 else 0)
        inputs.append(self.sight_distance / MAX_NORMALIZED_SIGHT_DISTANCE)
        inputs.append(self.sight_angle / (2 * math.pi))
        inputs.append(self.stamina / self.max_stamina if self.max_stamina > 0 else 0)
        inputs.append(min(1.0, self.hunger / HUNGER_MAX))
        return inputs

    def get_target_info(self, target):
        dist = self.pos.distance_to(target.pos)
        angle = (
            math.atan2(target.pos.y - self.pos.y, target.pos.x - self.pos.x)
            - self.angle + math.pi
        ) % (2 * math.pi) - math.pi
        return (
            angle / (self.sight_angle / 2 if self.sight_angle > 0 else 1),
            dist / (self.sight_distance if self.sight_distance > 0 else 1),
        )

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
            "Score": self.score,
            "Reproductions": self.reproductions,
            "Is Best": self.is_best,
        }
        for gene, value in self.genes.items():
            info[f"Gene: {gene}"] = round(value, 2)
        return info


class Food:
    _id_counter = -1

    def __init__(self, world_bounds, *, vec2_factory=None):
        self._vec2 = vec2_factory or _HeadlessVector2
        self.id = Food._id_counter
        Food._id_counter -= 1
        self.world_bounds = world_bounds
        self.pos = self._vec2(random.uniform(0, world_bounds[0]), random.uniform(0, world_bounds[1]))
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
