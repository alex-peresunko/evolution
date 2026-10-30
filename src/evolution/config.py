import math

# --- Display / World ---
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 900
WORLD_WIDTH = 1920
WORLD_HEIGHT = 900
MINIMAP_WIDTH = 200
MINIMAP_HEIGHT = int(WORLD_HEIGHT / WORLD_WIDTH * MINIMAP_WIDTH)
MINIMAP_X, MINIMAP_Y = 10, 10

# --- Population ---
NUM_HERBIVOROUS = 20
NUM_CARNIVORES = 30
NUM_FOOD = 50
NUM_OBSTACLES = 5
FOOD_RADIUS = 3
FOOD_RESPAWN_RATE = 2

# --- Spatial Grid ---
GRID_CELL_SIZE = 200

# --- Score / Worker sync ---
SCORE_SAMPLE_INTERVAL = 200
CHART_HISTORY_SAVE_MAX = 500  # chart samples per series kept in a saved brain file
WORKER_SYNC_INTERVAL = 300       # ticks between island brain exchanges (background mode)
ISLAND_IMMIGRANTS = 4            # weakest creatures replaced by an imported better brain

# --- Gene Defaults ---
DEFAULT_HERBIVORE_GENES = {
    "size": 10, "max_speed": 5, "sight_distance": 150,
    "sight_angle": math.radians(180), "max_stamina": 1000, "attractiveness": 10,
}
DEFAULT_CARNIVORE_GENES = {
    "size": 6, "max_speed": 10, "sight_distance": 600,
    "sight_angle": math.radians(120), "max_stamina": 1200, "attractiveness": 10,
}

# --- Gene Constraints ---
GENE_MIN_MAX = {
    "size": (5, 25),
    "max_speed": (2, 10),
    "sight_distance": (50, 800),
    "sight_angle": (math.radians(10), math.radians(360)),
    "max_stamina": (500, 2500),
    "attractiveness": (1, 100),
}
CARNIVORE_GENE_MIN_MAX = {**GENE_MIN_MAX, "max_speed": (2, 15)}

# --- Food ---
FOOD_MAX_LIFETIME = 1500
FOOD_MIN_HEALTH_FACTOR = 0.2

# --- Creature ---
HERBIVORE_HEALTH = 5000
CARNIVORE_HEALTH = 5000
HERBIVORE_HEALTH_PER_FOOD = 500
CARNIVORE_HEALTH_PER_FOOD = 1200
CREATURE_ROTATION_SPEED = 0.8
HEALTH_LOST_ON_HIT = 50
CARNIVORE_BITE_ANGLE = math.radians(60)
HUNGER_MAX = 400
HUNGER_HEALTH_PENALTY = 5
REPRODUCTION_HEALTH_THRESHOLD = 0.6
MAX_REPRODUCTIONS = 50
HEALTH_LOSS_PER_TICK = 1
HEALTH_LOSS_SPEED_FACTOR = 0.1
REPRODUCTION_COOLDOWN_TICKS = 180   # 3 s at 60 TPS
REPRODUCTION_POP_CAP_FACTOR = 10
MATE_SELECTION_RADIUS = 50
MATE_ATTRACTION_RADIUS = 300
MATE_SEEK_HUNGER_THRESHOLD = 0.5
MATE_SEEK_DANGER_THRESHOLD = 0.35
MATE_SEEK_BLEND = 0.75
MATE_SEEK_MIN_SPEED = 0.4
SMELL_DISTANCE = 400
MAX_NORMALIZED_SIGHT_DISTANCE = 2000
HEALTH_GAIN_SIZE_PENALTY = 0.01

# --- Trait Dependency Factors ---
SIZE_SPEED_PENALTY_FACTOR = 0.08
SPEED_STAMINA_EXPONENT = 1.2
SIZE_STAMINA_FACTOR = 80

# --- Life Stage ---
ADULTHOOD_AGE = 200
JUVENILE_SIZE_FACTOR = 0.2

# --- Stamina ---
STAMINA_DEPLETION_SPEED_THRESHOLD = 0.7
STAMINA_DEPLETION_RATE = 2.5
STAMINA_REGEN_RATE = 1.0
STAMINA_EXHAUSTION_PENALTY = 0.2

# --- Chases & fights ---
BITE_DAMAGE = 1800              # vs. 5000 health: ~3 bites kill a healthy adult
BITE_COOLDOWN_TICKS = 25
BITE_HEALTH_TRANSFER = 0.25     # share of the damage the carnivore gains as health
BITE_SCORE = 0.25               # per bite landed; a kill adds 1
ADRENALINE_STAMINA = 300        # stamina a bitten herbivore gains, so it can escape
RETALIATION_DAMAGE = 600        # times (prey.size / carn.size - 1), capped at 3x

# --- Carcasses ---
CARCASS_MEAT_PER_SIZE = 120
CARCASS_LIFETIME = 1200
CARCASS_EAT_RATE = 40           # meat eaten per tick of contact
SCAVENGE_SCORE_FACTOR = 0.5

# --- Meals (creatures stand still while eating; 0 = instant eating on the move) ---
HERBIVORE_EAT_TICKS = 45        # food's health is absorbed gradually over the meal
CARNIVORE_EAT_TICKS = 60        # ticks spent at a carcass, eating CARCASS_EAT_RATE per tick

# --- Weight (food eaten at full health is stored as fat) ---
FAT_MAX = 3000                  # fat cap, in health units
FAT_STORE_EFFICIENCY = 0.5      # fat gained per unit of surplus health
FAT_BURN_PER_TICK = 0.5         # basal metabolism
FAT_SPEED_PENALTY = 0.6         # speed / (1 + 0.6 * fat/FAT_MAX): full fat = 62% speed
FAT_SIZE_FACTOR = 0.4           # body size * (1 + 0.4 * fat/FAT_MAX)

# --- Kin sense / signals ---
KIN_SENSE_RADIUS = 250
KIN_DENSITY_NORM = 8
SIGNAL_RADIUS = 300
SIGNAL_HEALTH_COST = 0.5        # per tick at full signal

# --- Food patches ---
FOOD_PATCH_COUNT = 3
FOOD_PATCH_SHARE = 0.8          # share of respawned food placed in a patch
FOOD_PATCH_RADIUS = 110         # Gaussian sigma
FOOD_PATCH_SPEED = 0.3
FOOD_PATCH_TURN = 0.05

# --- Day/night ---
DAY_LENGTH_TICKS = 3600
NIGHT_SIGHT_FACTOR = 0.35       # share of sight distance left at midnight

# --- Neural Network ---
NUM_WHISKERS = 3
NUM_MEMORY = 2
WHISKER_TARGET_WIDTH = 12   # px; a target this close to a whisker ray is hit by it
BRAIN_INPUTS = (
    NUM_WHISKERS  # obstacle proximity
    + 2 + 2 + 2   # target, smell, prey velocity
    + 3 + 6       # health, sin/cos angle, normalized genes/stamina/hunger
    + 3 + 2       # kin angle/dist/density, signal angle/strength
    + 2 + 1       # carcass angle/dist, light
    + NUM_MEMORY
    + NUM_WHISKERS  # target whiskers (food / prey along each ray)
)
BRAIN_OUTPUTS = 3 + NUM_MEMORY  # rotation, speed, signal, memory
BRAIN_TOPOLOGY = [BRAIN_INPUTS, 16, 8, BRAIN_OUTPUTS]

# --- Evolution ---
MUTATION_RATE = 0.08
MUTATION_AMOUNT = 0.15
GENE_MUTATION_AMOUNT = 0.10
SURVIVAL_RATE = 0.15
AUTOSAVE_INTERVAL = 100
RESPAWN_IMMUNITY_TICKS = 500

# --- Colors (pygame only) ---
COLOR_CREATURE = (0, 255, 0)
COLOR_CARNIVORE = (255, 120, 120)
COLOR_BEST_CREATURE = (255, 255, 0)
COLOR_OBSTACLE = (60, 60, 60)
COLOR_WHISKER = (0, 125, 0)
COLOR_CARNIVORE_WHISKER = (125, 0, 0)
COLOR_TARGET_WHISKER = (255, 220, 60)
COLOR_TEXT = (255, 255, 255)
COLOR_BIRTH = (0, 255, 255)
CHART_FONT_SIZE = 15         # at chart scale 1.0
CHART_SIZE = (150, 75)       # at chart scale 1.0
CHART_SCALE_DEFAULT = 2.0    # [ / ] in game.py zoom the charts
CHART_SCALE_MIN, CHART_SCALE_MAX, CHART_SCALE_STEP = 1.0, 3.0, 0.25
HUD_FONT_SIZE = 18

# --- Brain panel (game.py, N toggles) ---
BRAIN_VIEW_EDGE_FRACTION = 0.15  # strongest share of connections drawn
BRAIN_VIEW_HISTORY = 200         # ticks of output history in the sparklines
BRAIN_PANEL_SIZE = (430, 520)
COLOR_BRAIN_NEG = (70, 140, 255)
COLOR_BRAIN_ZERO = (90, 90, 90)
COLOR_BRAIN_POS = (255, 150, 40)

# --- Sprite rendering (render.py) ---
SPRITE_ANGLE_STEPS = 72      # rotation buckets for cached creature sprites (5 degrees)
SPRITE_CACHE_LIMIT = 8000    # rotated sprites kept before the cache is flushed
NIGHT_MAX_ALPHA = 150
SIGNAL_RENDER_THRESHOLD = 0.6
COLOR_HIT = (255, 40, 30)
