import numpy as np
import random
import math
import json
import time
import os
import sys
from typing import Optional
from prometheus_client import start_http_server
from prometheus_client import Counter, Gauge

# =============================================================================
# Headless Replacements for Pygame functionality
# =============================================================================
class Vector2:
    """A 2D vector class to replace pygame.math.Vector2."""
    def __init__(self, x=0.0, y=0.0):
        self.x = float(x)
        self.y = float(y)

    def __add__(self, other):
        return Vector2(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Vector2(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar):
        return Vector2(self.x * scalar, self.y * scalar)

    def __truediv__(self, scalar):
        if scalar == 0:
            return Vector2(0, 0)
        return Vector2(self.x / scalar, self.y / scalar)
    
    def __iadd__(self, other):
        self.x += other.x
        self.y += other.y
        return self
        
    def __isub__(self, other):
        self.x -= other.x
        self.y -= other.y
        return self

    def distance_to(self, other):
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
        
    def length(self):
        return math.sqrt(self.x**2 + self.y**2)
        
    def __repr__(self):
        return f"Vector2({self.x}, {self.y})"

class SimpleRect:
    """A simple rectangle class to replace pygame.Rect."""
    def __init__(self, x, y, width, height):
        self.x, self.y, self.w, self.h = x, y, width, height
        self.left, self.right = x, x + width
        self.top, self.bottom = y, y + height

    def colliderect(self, other_rect):
        return (self.x < other_rect.x + other_rect.w and
                self.x + self.w > other_rect.x and
                self.y < other_rect.y + other_rect.h and
                self.y + self.h > other_rect.y)

    def clipline(self, p1, p2):
        """
        Clips a line segment to this rectangle.
        Returns a list containing the first intersection point as a Vector2 if it intersects, or None.
        This is a simplified replacement for pygame.Rect.clipline, tailored for whisker detection.
        """
        p1_x, p1_y = p1.x, p1.y
        p2_x, p2_y = p2.x, p2.y
        dx, dy = p2_x - p1_x, p2_y - p1_y
        
        t0, t1 = 0.0, 1.0
        
        checks = [
            (-dx, p1_x - self.left),    # Left
            (dx, self.right - p1_x),    # Right
            (-dy, p1_y - self.top),     # Top
            (dy, self.bottom - p1_y)    # Bottom
        ]
        
        for p, q in checks:
            if p == 0:
                if q < 0: return None # Parallel and outside
            else:
                r = q / p
                if p < 0:
                    if r > t1: return None
                    t0 = max(t0, r)
                else: # p > 0
                    if r < t0: return None
                    t1 = min(t1, r)

        if t0 < t1:
            if t0 > 0:
                ix = p1_x + t0 * dx
                iy = p1_y + t0 * dy
                return [Vector2(ix, iy)]
        return None


# --- Main Simulation Configuration ---
SCREEN_WIDTH = 2400  # Width of the simulation world
SCREEN_HEIGHT = 1300  # Height of the simulation world
NUM_HERBIVOROUS = 80  # Initial number of herbivores
NUM_CARNIVORES = 35  # Initial number of carnivores
NUM_FOOD = 60  # Initial number of food items
FOOD_RADIUS = 5  # Radius of food items
FOOD_RESPAWN_RATE = 0.2  # Probability of food respawning per tick
NUM_OBSTACLES = 5  # Number of obstacles in the world
GENERATION_TIME = 600  # Duration of one generation in ticks

# --- Spatial Grid Configuration ---
GRID_CELL_SIZE = 500  # Size of each cell in the spatial grid for optimization

# --- GENE DEFAULTS: Base values for the first generation ---
DEFAULT_HERBIVORE_GENES = {  # Default genetic traits for herbivores
    "size": 10, "max_speed": 5, "sight_distance": 150, "sight_angle": math.radians(180), 
    "max_stamina": 1000, "attractiveness": 10
}
DEFAULT_CARNIVORE_GENES = {  # Default genetic traits for carnivores
    "size": 6, "max_speed": 5, "sight_distance": 500, "sight_angle": math.radians(30), 
    "max_stamina": 1200, "attractiveness": 10
}

# --- Food Configuration ---
FOOD_MAX_LIFETIME = 2000  # Maximum lifetime of food before it rots
FOOD_MIN_HEALTH_FACTOR = 0.1  # Minimum health value of rotting food

# --- Creature Configuration ---
HERBIVORE_HEALTH = 1500  # Initial health of herbivores
CARNIVORE_HEALTH = 1500  # Initial health of carnivores
HERBIVORE_HEALTH_PER_FOOD = 1000  # Health gained by herbivores per food
CARNIVORE_HEALTH_PER_FOOD = 1000  # Health gained by carnivores per prey
CREATURE_ROTATION_SPEED = 0.8  # Rotation speed of creatures
HEALTH_LOST_ON_HIT = 50  # Health lost when colliding with obstacles
CARNIVORE_BITE_ANGLE = math.radians(60) # NEW: The angle of the carnivore's attack cone
REPRODUCTION_HEALTH_THRESHOLD = 0.2  # Minimum health ratio for reproduction
MAX_REPRODUCTIONS = 50  # Maximum number of reproductions per creature
HEALTH_LOSS_PER_TICK = 2  # Health lost per tick
HEALTH_LOSS_SPEED_FACTOR = 1  # Additional health loss based on speed
REPRODUCTION_COOLDOWN = 3  # Cooldown (in seconds) between reproductions
REPRODUCTION_POP_CAP_FACTOR = 10  # Population cap multiplier for reproduction
MATE_SELECTION_RADIUS = 50  # Radius for finding mates
SMELL_DISTANCE = 400  # Maximum distance for smell perception
MAX_NORMALIZED_SIGHT_DISTANCE = 2000  # Normalization factor for sight distance
HEALTH_GAIN_SIZE_PENALTY = 0.01  # Penalty for health gain based on size

# --- Life Stage Configuration ---
ADULTHOOD_AGE = 100  # Ticks until a creature becomes an adult
JUVENILE_SIZE_FACTOR = 0.4  # Juvenile size as a fraction of adult size

# --- Stamina Configuration ---
STAMINA_DEPLETION_SPEED_THRESHOLD = 0.7  # Speed threshold for stamina depletion
STAMINA_DEPLETION_RATE = 2.5  # Stamina lost per tick at high speed
STAMINA_REGEN_RATE = 1.0  # Stamina regained per tick when resting
STAMINA_EXHAUSTION_PENALTY = 0.2  # Speed penalty when stamina is exhausted

# --- Neural Network Configuration ---
NUM_WHISKERS = 3  # Number of whiskers for obstacle detection
BRAIN_TOPOLOGY = [NUM_WHISKERS + 2 + 2 + 3 + 4 + 1, 8, 2]  # Neural network structure

# --- Evolution Configuration ---
MUTATION_RATE = 0.02  # Probability of mutation per gene
MUTATION_AMOUNT = 0.02  # Magnitude of mutation
GENE_MUTATION_AMOUNT = 0.10  # Mutation magnitude for genes
SURVIVAL_RATE = 0.05  # Fraction of creatures that survive each generation
AUTOSAVE_INTERVAL = 100  # Interval for autosaving brains


# =============================================================================
# CLASS: SpatialHashGrid (FOR PERFORMANCE)
# =============================================================================
class SpatialHashGrid:
    def __init__(self, width, height, cell_size):
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.cols = int(math.ceil(width / cell_size))
        self.rows = int(math.ceil(height / cell_size))
        self.grid = {}

    def clear(self):
        self.grid = {}

    def insert(self, obj):
        key = (int(obj.pos.x / self.cell_size), int(obj.pos.y / self.cell_size))
        self.grid.setdefault(key, []).append(obj)

    def query(self, pos, radius):
        min_x = int((pos.x - radius) / self.cell_size)
        max_x = int((pos.x + radius) / self.cell_size)
        min_y = int((pos.y - radius) / self.cell_size)
        max_y = int((pos.y + radius) / self.cell_size)
        
        results = []
        for x in range(min_x, max_x + 1):
            for y in range(min_y, max_y + 1):
                results.extend(self.grid.get((x, y), []))
        return results

# =============================================================================
# CLASS: NeuralNetwork
# =============================================================================
class NeuralNetwork:
    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        self.weights, self.biases = [], []
        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.1
            self.weights.append(w)
            b = np.random.randn(1, layer_sizes[i+1]) * 0.1
            if i == len(layer_sizes) - 2: b[0, 0], b[0, 1] = 0.0, 1.0
            self.biases.append(b)

    def forward(self, inputs):
        x = np.array(inputs, dtype=float).reshape(1, -1)
        for i in range(len(self.weights)):
            x = np.dot(x, self.weights[i]) + self.biases[i]
            if i < len(self.weights) - 1: x = np.tanh(x)
        return np.tanh(x)[0]

    def mutate(self, mutation_rate, mutation_amount):
        for i in range(len(self.weights)):
            w_mask = np.random.rand(*self.weights[i].shape) < mutation_rate
            self.weights[i] += w_mask * np.random.uniform(-mutation_amount, mutation_amount, self.weights[i].shape)
            b_mask = np.random.rand(*self.biases[i].shape) < mutation_rate
            self.biases[i] += b_mask * np.random.uniform(-mutation_amount, mutation_amount, self.biases[i].shape)

    def copy(self):
        new_nn = NeuralNetwork(self.layer_sizes)
        new_nn.weights = [w.copy() for w in self.weights]
        new_nn.biases = [b.copy() for b in self.biases]
        return new_nn

    def save(self, filename, prometheus_metrics, generation=None, genes=None):
        save_dir = "saved_brains"
        os.makedirs(save_dir, exist_ok=True)
        data = {
            "layer_sizes": self.layer_sizes,
            "weights": [w.tolist() for w in self.weights], 
            "biases": [b.tolist() for b in self.biases]
        }
        if generation is not None: data["generation"] = generation
        if genes is not None: data["genes"] = genes
        if prometheus_metrics and prometheus_metrics.metric_game_uptime_seconds._value.get() is not None: 
            data["metric_game_uptime_seconds"] = prometheus_metrics.metric_game_uptime_seconds._value.get()
        with open(os.path.join(save_dir, filename), 'w') as f: json.dump(data, f, indent=4, sort_keys=True)
        print(f"Brain saved to {filename}")

    @staticmethod
    def load(filename, target_topology):
        filepath = os.path.join("saved_brains", filename)
        if not os.path.exists(filepath): raise FileNotFoundError(f"Brain file not found at {filepath}")
        with open(filepath, 'r') as f: data = json.load(f)
        saved_weights = [np.array(w) for w in data['weights']]
        saved_biases = [np.array(b) for b in data['biases']]
        nn = NeuralNetwork(target_topology)
        for i in range(len(nn.weights)):
            if i < len(saved_weights):
                target_w_shape, source_w_shape = nn.weights[i].shape, saved_weights[i].shape
                target_b_shape, source_b_shape = nn.biases[i].shape, saved_biases[i].shape
                rows_w, cols_w = min(target_w_shape[0], source_w_shape[0]), min(target_w_shape[1], source_w_shape[1])
                cols_b = min(target_b_shape[1], source_b_shape[1])
                nn.weights[i][:rows_w, :cols_w] = saved_weights[i][:rows_w, :cols_w]
                nn.biases[i][:, :cols_b] = saved_biases[i][:, :cols_b]
        print(f"Brain intelligently loaded from {filepath} into new topology {target_topology}")
        loaded_genes = data.get("genes")
        return nn, data.get("generation", 1), loaded_genes

# =============================================================================
# CLASS: Obstacle and Creature
# =============================================================================
class Obstacle:
    def __init__(self, x, y, width, height): self.rect = SimpleRect(x, y, width, height)

class Creature:
    _id_counter = 1
    def __init__(self, world_bounds, brain=None, is_carnivore=False, genes=None):
        self.id = Creature._id_counter; Creature._id_counter += 1
        self.world_bounds = world_bounds
        self.pos = Vector2(random.uniform(0, world_bounds[0]), random.uniform(0, world_bounds[1]))
        self.angle = random.uniform(0, 2 * math.pi)
        self.is_carnivore = is_carnivore
        self.health = CARNIVORE_HEALTH if is_carnivore else HERBIVORE_HEALTH
        self.max_health = self.health
        self.score, self.is_best, self.reproductions = 0, False, 0
        self.birth_time: Optional[float] = None
        self.last_reproduction_time: Optional[float] = None
        self.current_speed = 0.0

        # --- Life Stage Attributes ---
        self.age = 0
        self.is_adult = False

        if genes: self.genes = genes
        else: self.genes = DEFAULT_CARNIVORE_GENES.copy() if is_carnivore else DEFAULT_HERBIVORE_GENES.copy()
        
        base_size = (DEFAULT_CARNIVORE_GENES['size'] if self.is_carnivore else DEFAULT_HERBIVORE_GENES['size'])
        self.max_size_cap = base_size * 3.0
        self.genes["size"] = max(0.1, min(self.genes["size"], self.max_size_cap))

        # --- Distinguish Genetic Traits from Current Physical Traits ---
        self.genetic_size = self.genes["size"]
        self.genetic_max_speed = self.genes["max_speed"]
        self.sight_distance = self.genes["sight_distance"]
        self.sight_angle = self.genes["sight_angle"]
        self.max_stamina = self.genes.get("max_stamina", 1000)
        self.attractiveness = self.genes.get("attractiveness", 10)
        self.stamina = self.max_stamina

        # --- Initialize as a Juvenile ---
        self.size = self.genetic_size * JUVENILE_SIZE_FACTOR
        self.max_speed = self.genetic_max_speed * JUVENILE_SIZE_FACTOR

        self.whiskers = [0.0] * NUM_WHISKERS
        self.whisker_angles = np.linspace(-self.sight_angle / 2, self.sight_angle / 2, NUM_WHISKERS)
        self.brain = brain if brain else NeuralNetwork(BRAIN_TOPOLOGY)

    def perceive_and_think(self, local_food, local_herbivores, local_carnivores, obstacles):
        inputs = self.see(local_food, obstacles, local_herbivores, local_carnivores)
        return self.think(inputs)

    def act(self, outputs, global_food_list, global_herbivore_list, obstacles, grid):
        # --- Aging and Growth ---
        self.age += 1
        if not self.is_adult:
            if self.age >= ADULTHOOD_AGE:
                self.is_adult = True
                self.size = self.genetic_size
                self.max_speed = self.genetic_max_speed
            else:
                growth_ratio = self.age / ADULTHOOD_AGE
                growth_factor = JUVENILE_SIZE_FACTOR + (1 - JUVENILE_SIZE_FACTOR) * growth_ratio
                self.size = self.genetic_size * growth_factor
                self.max_speed = self.genetic_max_speed * growth_factor

        self.angle = (self.angle + outputs[0] * CREATURE_ROTATION_SPEED) % (2 * math.pi)
        speed_multiplier = max(0.1, self.health / self.max_health)
        is_exhausted = self.stamina <= 0
        speed_penalty = STAMINA_EXHAUSTION_PENALTY if is_exhausted else 1.0
        new_speed = max(0, outputs[1]) * self.max_speed * speed_multiplier * speed_penalty
        
        if self.max_speed > 0 and new_speed / self.max_speed > STAMINA_DEPLETION_SPEED_THRESHOLD:
            depletion_amount = (new_speed / self.max_speed) * STAMINA_DEPLETION_RATE
            self.stamina -= depletion_amount
        else:
            self.stamina += STAMINA_REGEN_RATE
        self.stamina = max(0, min(self.stamina, self.max_stamina))

        self.pos += Vector2(math.cos(self.angle), math.sin(self.angle)) * new_speed
        self.pos.x %= self.world_bounds[0]
        self.pos.y %= self.world_bounds[1]
        self.health -= (HEALTH_LOSS_PER_TICK + (new_speed * HEALTH_LOSS_SPEED_FACTOR))
        self.current_speed = new_speed

        # --- CORRECTED & EFFICIENT INTERACTION LOGIC ---
        interaction_radius = self.size * 2 
        nearby_objects = grid.query(self.pos, interaction_radius)

        if self.is_carnivore:
            base_size = DEFAULT_CARNIVORE_GENES['size']
            for obj in nearby_objects:
                if isinstance(obj, Creature) and not obj.is_carnivore:
                    prey = obj
                    is_close_enough = self.pos.distance_to(prey.pos) < (self.size + prey.size)
                    if is_close_enough:
                        vec_to_prey = prey.pos - self.pos
                        if vec_to_prey.length() > 0:
                            angle_to_prey = math.atan2(vec_to_prey.y, vec_to_prey.x)
                            angle_diff = (angle_to_prey - self.angle + math.pi) % (2 * math.pi) - math.pi
                            if abs(angle_diff) <= CARNIVORE_BITE_ANGLE / 2:
                                if prey in global_herbivore_list:
                                    global_herbivore_list.remove(prey)
                                    base_health_gain = CARNIVORE_HEALTH_PER_FOOD
                                    size_penalty_denominator = 1 + (self.genetic_size - base_size) * HEALTH_GAIN_SIZE_PENALTY
                                    actual_health_gain = base_health_gain / max(0.1, size_penalty_denominator)
                                    self.health = min(self.max_health, self.health + actual_health_gain)
                                    self.score += 1
                                    break
        else: # Is Herbivore
            base_size = DEFAULT_HERBIVORE_GENES['size']
            for obj in nearby_objects:
                if isinstance(obj, Food):
                    food = obj
                    if self.pos.distance_to(food.pos) < (self.size + FOOD_RADIUS):
                        if food in global_food_list:
                            global_food_list.remove(food)
                            base_health_gain = food.get_health_value()
                            size_penalty_denominator = 1 + (self.genetic_size - base_size) * HEALTH_GAIN_SIZE_PENALTY
                            actual_health_gain = base_health_gain / max(0.1, size_penalty_denominator)
                            self.health = min(self.max_health, self.health + actual_health_gain)
                            self.score += 1
                            break

        # --- CORRECTED OBSTACLE COLLISION ---
        for obs in obstacles:
            creature_rect = SimpleRect(self.pos.x - self.size, self.pos.y - self.size, self.size * 2, self.size * 2)
            if obs.rect.colliderect(creature_rect):
                self.health -= HEALTH_LOST_ON_HIT
                self.pos -= Vector2(math.cos(self.angle), math.sin(self.angle)) * self.current_speed * 3
                break 

    def see(self, food_items, obstacles, herbivores=None, carnivores=None):
        inputs = []
        for i, whisker_angle in enumerate(self.whisker_angles):
            end_point = self.pos + Vector2(math.cos(self.angle + whisker_angle), math.sin(self.angle + whisker_angle)) * self.sight_distance
            closest_dist = self.sight_distance
            for obs in obstacles:
                clipped_line = obs.rect.clipline(self.pos, end_point)
                if clipped_line: closest_dist = min(closest_dist, self.pos.distance_to(clipped_line[0]))
            self.whiskers[i] = closest_dist
            inputs.append(1.0 - (closest_dist / self.sight_distance if self.sight_distance > 0 else 1))
        
        def in_sight(target):
            if target.id == self.id: return False
            vec = target.pos - self.pos
            if 0 < vec.length() <= self.sight_distance:
                angle_to_target = (math.atan2(vec.y, vec.x) - self.angle + math.pi) % (2 * math.pi) - math.pi
                return abs(angle_to_target) <= self.sight_angle / 2
            return False

        if self.is_carnivore:
            visible_prey = [h for h in herbivores or [] if in_sight(h)]
            inputs.extend(self.get_target_info(min(visible_prey, key=lambda h: self.pos.distance_to(h.pos))) if visible_prey else [0, 1])
        else:
            visible_food = [f for f in food_items if in_sight(f)]
            food_angle, food_dist = self.get_target_info(min(visible_food, key=lambda f: self.pos.distance_to(f.pos))) if visible_food else (0, 1)
            visible_preds = [c for c in carnivores or [] if in_sight(c)]
            predator_angle, predator_dist = self.get_target_info(min(visible_preds, key=lambda c: self.pos.distance_to(c.pos))) if visible_preds else (0, 1)
            avoid_angle = food_angle - predator_angle * (1 - predator_dist)**2
            inputs.extend([np.clip(avoid_angle, -1, 1), food_dist])

        smell_angle, smell_strength = 0.0, 0.0
        if self.is_carnivore:
            smellable_prey = [h for h in herbivores or [] if h.id != self.id]
            if smellable_prey:
                closest_prey = min(smellable_prey, key=lambda p: self.pos.distance_to(p.pos))
                dist = self.pos.distance_to(closest_prey.pos)
                if dist < SMELL_DISTANCE:
                    angle = (math.atan2(closest_prey.pos.y - self.pos.y, closest_prey.pos.x - self.pos.x) - self.angle + math.pi) % (2 * math.pi) - math.pi
                    smell_angle = angle / math.pi
                    smell_strength = 1.0 - (dist / SMELL_DISTANCE)
        else: # Herbivore
            smellable_predators = [c for c in carnivores or [] if c.id != self.id]
            if smellable_predators:
                closest_predator = min(smellable_predators, key=lambda p: self.pos.distance_to(p.pos))
                dist = self.pos.distance_to(closest_predator.pos)
                if dist < SMELL_DISTANCE:
                    angle = (math.atan2(closest_predator.pos.y - self.pos.y, closest_predator.pos.x - self.pos.x) - self.angle + math.pi) % (2 * math.pi) - math.pi
                    smell_angle = angle / math.pi
                    smell_strength = 1.0 - (dist / SMELL_DISTANCE)
        inputs.extend([smell_angle, smell_strength])

        inputs.append(self.health / self.max_health)
        inputs.append(math.sin(self.angle))
        inputs.append(math.cos(self.angle))
        inputs.append(self.size / self.max_size_cap)
        inputs.append(self.current_speed / self.genetic_max_speed if self.genetic_max_speed > 0 else 0)
        inputs.append(self.sight_distance / MAX_NORMALIZED_SIGHT_DISTANCE)
        inputs.append(self.sight_angle / (2 * math.pi))
        inputs.append(self.stamina / self.max_stamina if self.max_stamina > 0 else 0)
        return inputs

    def get_target_info(self, target):
        dist = self.pos.distance_to(target.pos)
        angle = (math.atan2(target.pos.y - self.pos.y, target.pos.x - self.pos.x) - self.angle + math.pi) % (2 * math.pi) - math.pi
        return angle / (self.sight_angle / 2 if self.sight_angle > 0 else 1), dist / (self.sight_distance if self.sight_distance > 0 else 1)
    
    def think(self, inputs): return self.brain.forward(inputs)
    def is_alive(self): return self.health > 0
    
    def get_info(self):
        status = "Adult" if self.is_adult else f"Juvenile ({self.age}/{ADULTHOOD_AGE})"
        info = {"ID": self.id, "Type": "Carnivore" if self.is_carnivore else "Herbivore", "Status": status,
                "Health": int(self.health), "Stamina": f"{int(self.stamina)}/{int(self.max_stamina)}",
                "Score": self.score, "Reproductions": self.reproductions, "Is Best": self.is_best}
        for gene, value in self.genes.items():
            info[f"Gene: {gene}"] = round(value, 2)
        return info

# =============================================================================
# CLASS: Food
# =============================================================================
class Food:
    _id_counter = -1
    def __init__(self, world_bounds):
        self.id = Food._id_counter; Food._id_counter -= 1
        self.world_bounds = world_bounds
        self.pos = Vector2(random.uniform(0, world_bounds[0]), random.uniform(0, world_bounds[1]))
        self.lifetime = 0
        self.max_lifetime = FOOD_MAX_LIFETIME
        self.health_value_base = HERBIVORE_HEALTH_PER_FOOD
        self.min_health_factor = FOOD_MIN_HEALTH_FACTOR

    def update(self):
        self.lifetime += 1

    def is_rotten(self):
        return self.lifetime >= self.max_lifetime

    def get_health_value(self):
        if self.is_rotten(): return 0
        freshness = 1.0 - (self.lifetime / self.max_lifetime)
        return int(self.health_value_base * (self.min_health_factor + (1 - self.min_health_factor) * freshness))

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def combine_brains(brain1, brain2):
    child = brain1.copy()
    for i in range(len(child.weights)):
        mask = np.random.rand(*child.weights[i].shape) < 0.5
        child.weights[i][mask] = brain2.weights[i][mask]
        mask_b = np.random.rand(*child.biases[i].shape) < 0.5
        child.biases[i][mask_b] = brain2.biases[i][mask_b]
    return child

def combine_genes(genes1, genes2, default_genes):
    child_genes = {}
    for key, default_value in default_genes.items():
        p1_gene = genes1.get(key, default_value)
        p2_gene = genes2.get(key, default_value)
        avg_gene = (p1_gene + p2_gene) / 2.0
        # FIX: The mutation amount should be proportional to the new average gene, not the default.
        mutation_base = avg_gene
        mutation = mutation_base * random.uniform(-GENE_MUTATION_AMOUNT, GENE_MUTATION_AMOUNT)
        child_genes[key] = max(0.1, avg_gene + mutation)
    return child_genes

def mutate_genes(genes, default_genes):
    mutated_genes = {}
    for key, default_value in default_genes.items():
        parent_value = genes.get(key, default_value)
        
        # FIX: The mutation is now proportional to the PARENT's value.
        mutation_base = parent_value 
        
        mutation = mutation_base * random.uniform(-GENE_MUTATION_AMOUNT, GENE_MUTATION_AMOUNT)
        mutated_genes[key] = max(0.1, parent_value + mutation)
    return mutated_genes

# =============================================================================
# SIMULATION LOGIC FUNCTIONS
# =============================================================================
def update_world(sim_state):
    sim_state['generation_timer'] += 1
    grid = sim_state['grid']
    for food in sim_state['food_items']: food.update()
    sim_state['food_items'][:] = [f for f in sim_state['food_items'] if not f.is_rotten()]
    if random.random() < FOOD_RESPAWN_RATE and len(sim_state['food_items']) < NUM_FOOD * 1.5:
        sim_state['food_items'].append(Food(sim_state['world_bounds']))
    all_creatures = sim_state['creatures'] + sim_state['carnivores']
    grid.clear()
    for obj in all_creatures + sim_state['food_items']:
        grid.insert(obj)
    for c in all_creatures:
        local_objects = grid.query(c.pos, max(c.sight_distance, SMELL_DISTANCE))
        local_food = [obj for obj in local_objects if isinstance(obj, Food)]
        local_herbivores = [obj for obj in local_objects if isinstance(obj, Creature) and not obj.is_carnivore]
        local_carnivores = [obj for obj in local_objects if isinstance(obj, Creature) and obj.is_carnivore]
        outputs = c.perceive_and_think(local_food, local_herbivores, local_carnivores, sim_state['obstacles'])
        c.act(outputs, sim_state['food_items'], sim_state['creatures'], sim_state['obstacles'], grid)
    sim_state['creatures'][:] = [c for c in sim_state['creatures'] if c.is_alive()]
    sim_state['carnivores'][:] = [c for c in sim_state['carnivores'] if c.is_alive()]
    if sim_state['creatures']:
        for c in sim_state['creatures']: c.is_best = False
        best_herbivore = max(sim_state['creatures'], key=lambda c: c.score, default=None)
        if best_herbivore:
            best_herbivore.is_best = True
    if sim_state['carnivores']:
        for c in sim_state['carnivores']: c.is_best = False
        best_carnivore = max(sim_state['carnivores'], key=lambda c: c.score, default=None)
        if best_carnivore:
            best_carnivore.is_best = True
    
    new_offspring = []
    reproduced_ids = set()
    now = time.time()
    for group, pop_limit, max_health, default_genes in [
        (sim_state['creatures'], NUM_HERBIVOROUS, HERBIVORE_HEALTH, DEFAULT_HERBIVORE_GENES), 
        (sim_state['carnivores'], NUM_CARNIVORES, CARNIVORE_HEALTH, DEFAULT_CARNIVORE_GENES)
    ]:
        pop_cap = pop_limit * REPRODUCTION_POP_CAP_FACTOR
        for p1 in group:
            if p1.id in reproduced_ids or len(group) + len(new_offspring) >= pop_cap:
                continue

            p1_is_eligible = (p1.is_adult and p1.health > max_health * REPRODUCTION_HEALTH_THRESHOLD and 
                              p1.reproductions < MAX_REPRODUCTIONS and 
                              (p1.last_reproduction_time is None or now - p1.last_reproduction_time > REPRODUCTION_COOLDOWN))
            
            if not p1_is_eligible:
                continue

            potential_mates = []
            for p2 in group:
                if p1.id == p2.id or p2.id in reproduced_ids:
                    continue
                
                p2_is_eligible = (p2.is_adult and p2.health > max_health * REPRODUCTION_HEALTH_THRESHOLD and 
                                  p2.reproductions < MAX_REPRODUCTIONS and 
                                  (p2.last_reproduction_time is None or now - p2.last_reproduction_time > REPRODUCTION_COOLDOWN))
                
                if p2_is_eligible and p1.pos.distance_to(p2.pos) < MATE_SELECTION_RADIUS:
                    potential_mates.append(p2)

            if potential_mates:
                best_mate = max(potential_mates, key=lambda mate: mate.attractiveness)
                child_genes = combine_genes(p1.genes, best_mate.genes, default_genes)
                child_brain = combine_brains(p1.brain, best_mate.brain)
                child = Creature(sim_state['world_bounds'], brain=child_brain, is_carnivore=p1.is_carnivore, genes=child_genes)
                child.pos = (p1.pos + best_mate.pos) / 2
                child.birth_time = now
                new_offspring.append(child)
                p1.health *= 0.7
                best_mate.health *= 0.7
                p1.reproductions += 1
                best_mate.reproductions += 1
                p1.last_reproduction_time = now
                best_mate.last_reproduction_time = now
                reproduced_ids.add(p1.id)
                reproduced_ids.add(best_mate.id)
                
    sim_state['creatures'].extend([c for c in new_offspring if not c.is_carnivore])
    sim_state['carnivores'].extend([c for c in new_offspring if c.is_carnivore])

def evolve_population(sim_state, prometheus_metrics):
    sim_state['herbivore_score_history'].append(max([c.score for c in sim_state['creatures']] + [0]))
    sim_state['carnivore_score_history'].append(max([c.score for c in sim_state['carnivores']] + [0]))

    hall_of_fame = sim_state['hall_of_fame']
    best_herbivore_of_gen = max(sim_state['creatures'], key=lambda c: c.score, default=None)
    if best_herbivore_of_gen and best_herbivore_of_gen.score > hall_of_fame['herbivore']['score']:
        print(f"\nNew Herbivore record! Gen {sim_state['generation']}, Score: {best_herbivore_of_gen.score}")
        hall_of_fame['herbivore']['score'] = best_herbivore_of_gen.score
        hall_of_fame['herbivore']['genes'] = best_herbivore_of_gen.genes.copy()
        hall_of_fame['herbivore']['brain'] = best_herbivore_of_gen.brain.copy()

    best_carnivore_of_gen = max(sim_state['carnivores'], key=lambda c: c.score, default=None)
    if best_carnivore_of_gen and best_carnivore_of_gen.score > hall_of_fame['carnivore']['score']:
        print(f"\nNew Carnivore record! Gen {sim_state['generation']}, Score: {best_carnivore_of_gen.score}")
        hall_of_fame['carnivore']['score'] = best_carnivore_of_gen.score
        hall_of_fame['carnivore']['genes'] = best_carnivore_of_gen.genes.copy()
        hall_of_fame['carnivore']['brain'] = best_carnivore_of_gen.brain.copy()
    
    def get_new_population(survivors, count, is_carnivore, default_genes):
        species = 'carnivore' if is_carnivore else 'herbivore'
        
        if not survivors:
            print(f"\n--- {species.capitalize()} extinction event! Repopulating from Hall of Fame and defaults. ---")
            progenitor_record = hall_of_fame[species]
            new_pop = []
            
            if progenitor_record['genes'] and progenitor_record['brain']:
                progenitor = Creature(sim_state['world_bounds'], brain=progenitor_record['brain'].copy(), 
                                      is_carnivore=is_carnivore, genes=progenitor_record['genes'].copy())
                new_pop.append(progenitor)
                
                num_mutated_descendants = (count - 1) // 2
                for _ in range(num_mutated_descendants):
                    new_brain = progenitor_record['brain'].copy()
                    new_brain.mutate(MUTATION_RATE, MUTATION_AMOUNT)
                    new_genes = mutate_genes(progenitor_record['genes'], default_genes)
                    new_pop.append(Creature(sim_state['world_bounds'], brain=new_brain, is_carnivore=is_carnivore, genes=new_genes))
            
            num_defaults = count - len(new_pop)
            for _ in range(num_defaults):
                new_pop.append(Creature(sim_state['world_bounds'], is_carnivore=is_carnivore, genes=default_genes))
            
            return new_pop
        
        new_pop = []
        for _ in range(count):
            parent = random.choice(survivors)
            new_brain = parent.brain.copy()
            new_brain.mutate(MUTATION_RATE, MUTATION_AMOUNT)
            new_genes = mutate_genes(parent.genes, default_genes)
            new_pop.append(Creature(sim_state['world_bounds'], brain=new_brain, is_carnivore=is_carnivore, genes=new_genes))
        return new_pop
    
    sim_state['creatures'].sort(key=lambda c: c.score, reverse=True)
    num_to_select_h = max(1, int(len(sim_state['creatures']) * SURVIVAL_RATE)) if sim_state['creatures'] else 0
    survivors_h = sim_state['creatures'][:num_to_select_h]
    
    sim_state['carnivores'].sort(key=lambda c: c.score, reverse=True)
    num_to_select_c = max(1, int(len(sim_state['carnivores']) * SURVIVAL_RATE)) if sim_state['carnivores'] else 0
    survivors_c = sim_state['carnivores'][:num_to_select_c]

    if sim_state['generation'] % AUTOSAVE_INTERVAL == 0:
        print(f"\n--- AUTOSAVING BRAINS FOR END OF GENERATION {sim_state['generation']} ---")
        if survivors_h:
            survivors_h[0].brain.save(f"autosave_herbivore_gen_{sim_state['generation']}.json", prometheus_metrics,
                                      generation=sim_state['generation'], genes=survivors_h[0].genes)
        if survivors_c:
            survivors_c[0].brain.save(f"autosave_carnivore_gen_{sim_state['generation']}.json", prometheus_metrics,
                                      generation=sim_state['generation'], genes=survivors_c[0].genes)

    sim_state['creatures'] = get_new_population(survivors_h, NUM_HERBIVOROUS, False, DEFAULT_HERBIVORE_GENES)
    sim_state['carnivores'] = get_new_population(survivors_c, NUM_CARNIVORES, True, DEFAULT_CARNIVORE_GENES)
    
    sim_state['generation'] += 1
    sim_state['generation_timer'] = 0

class GameMetrics:
    def __init__(self, sim_state):
        self.metric_game_uptime_seconds = Counter('evolution_running_seconds', 'Time the game has been running in seconds')
        self.game_start_time = time.time()
        self.metric_current_generation_number = Gauge('evolution_current_generation_number', 'Current generation number in the simulation')
        self.metric_best_herbivore_score = Gauge('evolution_best_herbivore_score', 'Best herbivore score in the current generation')
        self.metric_best_carnivore_score = Gauge('evolution_best_carnivore_score', 'Best carnivore score in the current generation')
        self.metric_current_herbivore_count = Gauge('evolution_current_herbivore_count', 'Current number of herbivores in the simulation')
        self.metric_current_carnivore_count = Gauge('evolution_current_carnivore_count', 'Current number of carnivores in the simulation')
        self.metric_current_food_count = Gauge('evolution_current_food_count', 'Current number of food items in the simulation')
        self.metric_best_herbivore_gene = Gauge('evolution_best_herbivore_gene', 'Best herbivore gene in the current generation', ['gene'])
        self.metric_best_carnivore_gene = Gauge('evolution_best_carnivore_gene', 'Best carnivore gene in the current generation', ['gene'])
        self.metric_creature_brain_topology_input_size = Gauge('evolution_creature_brain_topology_input_size', 'Input size of the creature brain topology')
        self.metric_creature_brain_topology_hidden_layers = Gauge('evolution_creature_brain_topology_hidden_layer_sizes', 'Hidden layer sizes of the creature brain topology')
        self.metric_creature_brain_topology_output_size = Gauge('evolution_creature_brain_topology_output_size', 'Output size of the creature brain topology')
        self.metric_creature_brain_topology_hidden_layer_neurons = Gauge('evolution_creature_brain_topology_hidden_layer_neurons', 'Number of neurons in each hidden layer of the creature brain topology')
        self.metric_best_herbivore_current_stamina = Gauge('evolution_best_herbivore_current_stamina', 'Current stamina of the best herbivore')
        self.metric_best_carnivore_current_stamina = Gauge('evolution_best_carnivore_current_stamina', 'Current stamina of the best carnivore')
        self.metric_current_herbivore_adult_count = Gauge('evolution_current_herbivore_adult_count', 'Current number of adult herbivores')
        self.metric_current_carnivore_adult_count = Gauge('evolution_current_carnivore_adult_count', 'Current number of adult carnivores')
        self.metric_hall_of_fame_herbivore_score = Gauge('evolution_hall_of_fame_herbivore_score', 'All-time best score record for a herbivore')
        self.metric_hall_of_fame_carnivore_score = Gauge('evolution_hall_of_fame_carnivore_score', 'All-time best score record for a carnivore')

        best_herbivore = max(sim_state['creatures'], key=lambda c: c.score, default=None)
        best_carnivore = max(sim_state['carnivores'], key=lambda c: c.score, default=None)
        self.metric_game_uptime_seconds.inc(0)
        self.metric_current_generation_number.set(0)
        self.metric_best_herbivore_score.set(0)
        self.metric_best_carnivore_score.set(0)
        self.metric_current_herbivore_count.set(len(sim_state['creatures']))
        self.metric_current_carnivore_count.set(len(sim_state['carnivores']))
        self.metric_current_food_count.set(len(sim_state['food_items']))
        self.metric_best_herbivore_current_stamina.set(0)
        self.metric_best_carnivore_current_stamina.set(0)
        self.metric_current_herbivore_adult_count.set(0)
        self.metric_current_carnivore_adult_count.set(0)
        self.metric_hall_of_fame_herbivore_score.set(0)
        self.metric_hall_of_fame_carnivore_score.set(0)

        for gene, value in best_herbivore.genes.items() if best_herbivore else DEFAULT_HERBIVORE_GENES.items():
            self.metric_best_herbivore_gene.labels(gene).set(value)
        for gene, value in best_carnivore.genes.items() if best_carnivore else DEFAULT_CARNIVORE_GENES.items():
            self.metric_best_carnivore_gene.labels(gene).set(value)
        self.metric_creature_brain_topology_input_size.set(BRAIN_TOPOLOGY[0])
        self.metric_creature_brain_topology_hidden_layers.set(len(BRAIN_TOPOLOGY) - 2)
        self.metric_creature_brain_topology_output_size.set(BRAIN_TOPOLOGY[-1])
        self.metric_creature_brain_topology_hidden_layer_neurons.set((BRAIN_TOPOLOGY[1]))

    def update(self, sim_state, dt):
        best_herbivore = max(sim_state['creatures'], key=lambda c: c.score, default=None)
        best_carnivore = max(sim_state['carnivores'], key=lambda c: c.score, default=None)
        self.metric_game_uptime_seconds.inc(dt)
        self.metric_current_generation_number.set(sim_state['generation'])
        self.metric_best_herbivore_score.set(best_herbivore.score if best_herbivore else 0)
        self.metric_best_carnivore_score.set(best_carnivore.score if best_carnivore else 0)
        self.metric_current_herbivore_count.set(len(sim_state['creatures']))
        self.metric_current_carnivore_count.set(len(sim_state['carnivores']))
        self.metric_current_food_count.set(len(sim_state['food_items']))
        self.metric_best_herbivore_current_stamina.set(best_herbivore.stamina if best_herbivore else 0)
        self.metric_best_carnivore_current_stamina.set(best_carnivore.stamina if best_carnivore else 0)
        self.metric_current_herbivore_adult_count.set(sum(1 for c in sim_state['creatures'] if c.is_adult))
        self.metric_current_carnivore_adult_count.set(sum(1 for c in sim_state['carnivores'] if c.is_adult))
        self.metric_hall_of_fame_herbivore_score.set(max(0, sim_state['hall_of_fame']['herbivore']['score']))
        self.metric_hall_of_fame_carnivore_score.set(max(0, sim_state['hall_of_fame']['carnivore']['score']))

        for gene, value in best_herbivore.genes.items() if best_herbivore else DEFAULT_HERBIVORE_GENES.items():
            self.metric_best_herbivore_gene.labels(gene).set(value)
        for gene, value in best_carnivore.genes.items() if best_carnivore else DEFAULT_CARNIVORE_GENES.items():
            self.metric_best_carnivore_gene.labels(gene).set(value)
        self.metric_creature_brain_topology_input_size.set(BRAIN_TOPOLOGY[0])
        self.metric_creature_brain_topology_hidden_layers.set(len(BRAIN_TOPOLOGY) - 2)
        self.metric_creature_brain_topology_output_size.set(BRAIN_TOPOLOGY[-1])
        self.metric_creature_brain_topology_hidden_layer_neurons.set((BRAIN_TOPOLOGY[1]))

# =============================================================================
# Main Simulation
# =============================================================================
def create_initial_population(count, is_carnivore, world_bounds, brain_file):
    """Creates an initial population, loading from a file if it exists."""
    default_genes = DEFAULT_CARNIVORE_GENES if is_carnivore else DEFAULT_HERBIVORE_GENES
    species_name = "carnivore" if is_carnivore else "herbivore"
    
    try:
        progenitor_brain, gen, progenitor_genes = NeuralNetwork.load(brain_file, BRAIN_TOPOLOGY)
        if progenitor_genes is None: progenitor_genes = default_genes # Fallback
        
        print(f"Loading initial {species_name} population from {brain_file} (Gen {gen}).")
        
        new_pop = []
        # Add one pure progenitor
        progenitor_creature = Creature(world_bounds, brain=progenitor_brain, is_carnivore=is_carnivore, genes=progenitor_genes)
        new_pop.append(progenitor_creature)
        
        # Fill the rest with mutated descendants
        for _ in range(count - 1):
            mutated_brain = progenitor_brain.copy()
            mutated_brain.mutate(MUTATION_RATE, MUTATION_AMOUNT)
            mutated_genes = mutate_genes(progenitor_genes, default_genes)
            new_pop.append(Creature(world_bounds, brain=mutated_brain, is_carnivore=is_carnivore, genes=mutated_genes))
        
        return new_pop, gen
        
    except FileNotFoundError:
        print(f"No saved brain found for {species_name}s. Creating a new random population.")
        return [Creature(world_bounds, is_carnivore=is_carnivore) for _ in range(count)], 1

def main():
    # --- NEW: Load populations from files or create new ones ---
    herbivores, h_gen = create_initial_population(NUM_HERBIVOROUS, False, (SCREEN_WIDTH, SCREEN_HEIGHT), "best_brain_herbivore.json")
    carnivores, c_gen = create_initial_population(NUM_CARNIVORES, True, (SCREEN_WIDTH, SCREEN_HEIGHT), "best_brain_carnivore.json")

    sim_state = {
        'running': True,
        'world_bounds': (SCREEN_WIDTH, SCREEN_HEIGHT),
        'grid': SpatialHashGrid(SCREEN_WIDTH, SCREEN_HEIGHT, GRID_CELL_SIZE),
        'creatures': herbivores,
        'carnivores': carnivores,
        'food_items': [Food((SCREEN_WIDTH, SCREEN_HEIGHT)) for _ in range(NUM_FOOD)],
        'obstacles': [Obstacle(random.randint(100, SCREEN_WIDTH-100), random.randint(100, SCREEN_HEIGHT-100), random.randint(50, 150), random.randint(50, 150)) for _ in range(NUM_OBSTACLES)],
        'herbivore_score_history': [], 'carnivore_score_history': [],
        'generation': max(h_gen, c_gen), 
        'generation_timer': 0,
        'hall_of_fame': {
            'herbivore': {'score': -1, 'genes': None, 'brain': None},
            'carnivore': {'score': -1, 'genes': None, 'brain': None}
        },
    }
    prometheus_metrics = GameMetrics(sim_state)

    last_time = time.time()
    ticks_this_second = 0
    last_tps_report_time = time.time()

    print(f"Headless simulation starting at Generation {sim_state['generation']}... Press Ctrl+C to stop.")
    try:
        while sim_state['running']:
            now = time.time()
            dt = now - last_time
            last_time = now

            update_world(sim_state)

            if sim_state['generation_timer'] > GENERATION_TIME:
                evolve_population(sim_state, prometheus_metrics)
                print(f"\n--- Generation {sim_state['generation']-1} complete ---")
                print(f"    Herbivore best score: {sim_state['herbivore_score_history'][-1]}")
                print(f"    Carnivore best score: {sim_state['carnivore_score_history'][-1]}")
                print(f"    HoF Herbivore: {sim_state['hall_of_fame']['herbivore']['score']}")
                print(f"    HoF Carnivore: {sim_state['hall_of_fame']['carnivore']['score']}")

            prometheus_metrics.update(sim_state, dt)

            ticks_this_second += 1
            if now - last_tps_report_time >= 1.0:
                tps = ticks_this_second
                ticks_this_second = 0
                last_tps_report_time = now
                progress = sim_state['generation_timer'] / GENERATION_TIME
                progress_bar = '#' * int(progress * 20)
                progress_bar = progress_bar.ljust(20, '-')
                sys.stdout.write(f"\rGen {sim_state['generation']} [{progress_bar}] {int(progress*100)}% | TPS: {tps} | Herb: {len(sim_state['creatures'])} | Carn: {len(sim_state['carnivores'])}   ")
                sys.stdout.flush()
    
    except KeyboardInterrupt:
        print("\nSimulation stopped by user.")
    finally:
        # --- NEW: Save best brains from Hall of Fame on exit ---
        print("\nSaving best brains from Hall of Fame on exit...")
        
        # Save best herbivore
        hof_herbivore = sim_state['hall_of_fame']['herbivore']
        if hof_herbivore['brain'] is not None:
            hof_herbivore['brain'].save(
                "best_brain_herbivore.json", 
                prometheus_metrics,
                generation=sim_state['generation'], 
                genes=hof_herbivore['genes']
            )
        else:
            print("No herbivore in Hall of Fame to save.")

        # Save best carnivore
        hof_carnivore = sim_state['hall_of_fame']['carnivore']
        if hof_carnivore['brain'] is not None:
            hof_carnivore['brain'].save(
                "best_brain_carnivore.json", 
                prometheus_metrics,
                generation=sim_state['generation'], 
                genes=hof_carnivore['genes']
            )
        else:
            print("No carnivore in Hall of Fame to save.")
            
        print("Exiting.")


if __name__ == "__main__":
    start_http_server(8000)
    main()