import math

# --- Display / World ---
SCREEN_WIDTH = 1400
SCREEN_HEIGHT = 900
WORLD_WIDTH = 3000
WORLD_HEIGHT = 2000
MINIMAP_WIDTH = 200
MINIMAP_HEIGHT = int(WORLD_HEIGHT / WORLD_WIDTH * MINIMAP_WIDTH)
MINIMAP_X, MINIMAP_Y = 10, 10
ZOOM_MAX = 4.0               # mouse wheel zoom in game.py; the minimum fits the world
ZOOM_STEP = 1.15             # zoom factor per wheel notch

# --- Population ---
NUM_HERBIVOROUS = 120
NUM_CARNIVORES = 30
NUM_FOOD = 400
NUM_OBSTACLES = 5
NUM_TREES = 14
TREE_CROWN_RADIUS_RANGE = (34, 60)  # px; dinosaurs are ~25-45 px long
TREE_TRUNK_FRACTION = 0.28          # solid trunk side as a share of the crown radius
TREE_CROWN_ALPHA = 215              # crowns are drawn over creatures
FOOD_RADIUS = 3
FOOD_RESPAWN_RATE = 2

# --- Spatial Grid ---
GRID_CELL_SIZE = 200

# --- Gene-drift chart (G in game.py): species mean of each gene / its default ---
GENE_CHART_COLORS = {
    "size": (120, 200, 255), "max_speed": (255, 210, 80), "sight_distance": (160, 255, 140),
    "sight_angle": (80, 220, 200), "max_stamina": (255, 150, 90), "attractiveness": (255, 120, 200),
    "mutation_rate": (200, 160, 255), "lifespan": (230, 230, 230), "litter_size": (255, 90, 90),
}

# --- Score / Worker sync ---
SCORE_SAMPLE_INTERVAL = 200
CHART_HISTORY_SAVE_MAX = 500  # chart samples per series kept in a saved brain file
WORKER_SYNC_INTERVAL = 300       # ticks between island brain exchanges (background mode)
WATCH_POLL_SECONDS = 1.0         # how often watch mode (game.py --watch) checks brain files
ISLAND_IMMIGRANTS = 4            # weakest creatures replaced by an imported better brain

# --- Gene Defaults ---
# Shared by both species: brain mutation of the offspring, aging, litter size and a neutral
# colour marker (drawn as a hue shift, used for speciation).
_LIFE_HISTORY_GENES = {
    "mutation_rate": 0.08, "mutation_amount": 0.15,
    "lifespan": 3000, "litter_size": 1.0, "color": 0.5,
}
DEFAULT_HERBIVORE_GENES = {
    "size": 10, "max_speed": 5, "sight_distance": 150,
    "sight_angle": math.radians(180), "max_stamina": 1000, "attractiveness": 10,
    **_LIFE_HISTORY_GENES,
}
DEFAULT_CARNIVORE_GENES = {
    "size": 6, "max_speed": 10, "sight_distance": 600,
    "sight_angle": math.radians(120), "max_stamina": 1200, "attractiveness": 10,
    **_LIFE_HISTORY_GENES,
}

# --- Gene Constraints ---
GENE_MIN_MAX = {
    "size": (5, 25),
    "max_speed": (2, 10),
    "sight_distance": (50, 800),
    "sight_angle": (math.radians(10), math.radians(360)),
    "max_stamina": (500, 2500),
    "attractiveness": (1, 100),
    "mutation_rate": (0.005, 0.4),
    "mutation_amount": (0.02, 0.6),
    "lifespan": (1000, 20000),
    "litter_size": (1, 4),
    "color": (0, 1),
}
CARNIVORE_GENE_MIN_MAX = {**GENE_MIN_MAX, "max_speed": (2, 15)}

# --- Food ---
FOOD_MAX_LIFETIME = 1500
FOOD_MIN_HEALTH_FACTOR = 0.2

# --- Creature ---
HERBIVORE_HEALTH = 5000
CARNIVORE_HEALTH = 5000
HERBIVORE_HEALTH_PER_FOOD = 600
CARNIVORE_HEALTH_PER_FOOD = 1200
CREATURE_ROTATION_SPEED = 0.25
HEALTH_LOST_ON_HIT = 50
CARNIVORE_BITE_ANGLE = math.radians(60)
HUNGER_MAX = 400
HUNGER_HEALTH_PENALTY = 5
REPRODUCTION_HEALTH_THRESHOLD = 0.5
MAX_REPRODUCTIONS = 50
HEALTH_LOSS_PER_TICK = 1
HEALTH_LOSS_SPEED_FACTOR = 0.1
# Upkeep: HEALTH_LOSS_PER_TICK times a weighted sum of trait / species-default ratios. The
# weights sum to 1, so a creature with default genes pays exactly HEALTH_LOSS_PER_TICK.
UPKEEP_WEIGHTS = {
    "body": 0.5,            # (lean size / default size) ** KLEIBER_EXPONENT
    "sight": 0.25,          # sight_distance * sight_angle
    "stamina": 0.1,
    "lifespan": 0.1,        # somatic maintenance
    "attractiveness": 0.05,  # costly display (handicap)
}
KLEIBER_EXPONENT = 0.75
REPRODUCTION_COOLDOWN_TICKS = 180   # 3 s at 60 TPS
REPRODUCTION_POP_CAP_FACTOR = 2.5
MATE_SELECTION_RADIUS = 60
MATE_ATTRACTION_RADIUS = 500
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
SENESCENCE_RATE = 3             # extra health loss per tick at twice the lifespan gene, linear
LITTER_ENDOWMENT = 1.0          # share of max health split among a litter's young
LITTER_SPREAD = 6               # px; littermates are placed around the parents' midpoint
JUVENILE_SIZE_FACTOR = 0.2

# --- Stamina ---
STAMINA_EFFORT_THRESHOLD = 0.6  # brain speed output above which running costs stamina
STAMINA_DEPLETION_RATE = 3.0    # per tick at full effort
STAMINA_REGEN_RATE = 1.0
STAMINA_EXHAUSTION_PENALTY = 0.2
STAMINA_RECOVERY_SHARE = 0.2    # a winded creature runs again at this share of max stamina

# --- Lakes (swimmable: slow and tiring) ---
NUM_LAKES = 4
LAKE_RADIUS_RANGE = (80, 200)          # mean shore radius
LAKE_LOBES = (1, 4)                    # overlapping blobs merged into one lake
LAKE_SHORE_ROUGHNESS = 0.45            # blob shore radius varies within (1 +- this) * mean
LAKE_SHORE_HARMONICS = 10              # bays and headlands per blob, coarse to fine
WATER_SPEED_FACTOR = 0.35
HERBIVORE_SWIM_STAMINA_COST = 3.0   # per tick in water, replaces regen
CARNIVORE_SWIM_STAMINA_COST = 8.0

# --- Chases & fights ---
BITE_DAMAGE = 1800              # vs. 5000 health: ~3 bites kill a healthy adult
BITE_COOLDOWN_TICKS = 25
BITE_HEALTH_TRANSFER = 0.25     # share of the damage the carnivore gains as health
BITE_SCORE = 0.25               # per bite landed; a kill adds 1
ADRENALINE_STAMINA = 300        # stamina a bitten herbivore gains, so it can escape
HURT_FADE_TICKS = 30            # the hurt input falls from 1 to 0 this many ticks after a bite
RETALIATION_DAMAGE = 600        # times (prey.size / carn.size - 1), capped at 3x

# --- Carcasses ---
CARCASS_MEAT_PER_SIZE = 120
CARCASS_LIFETIME = 1200
CARCASS_EAT_RATE = 40           # meat eaten per tick of contact
SCAVENGE_SCORE_FACTOR = 0.5

# --- Meals (creatures stand still while eating; 0 = instant eating on the move) ---
HERBIVORE_EAT_TICKS = 45        # food's health is absorbed gradually over the meal
CARNIVORE_EAT_TICKS = 60        # ticks spent at a carcass, eating CARCASS_EAT_RATE per tick
MEAL_BREAK_THRESHOLD = 0.5      # stop_eating output above which a meal is abandoned, rest lost

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

# --- Seasons ---
SEASON_LENGTH_TICKS = 8 * DAY_LENGTH_TICKS   # one year; a run starts in spring
SEASON_FOOD_AMPLITUDE = 0.6     # food cap x (1 + 0.6 * season): summer 1.6x, winter 0.4x

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
    + 1             # season
    + 2             # carcass smell angle/strength
    + 1             # in water
    + 1             # eating
    + 1             # hurt
)
BRAIN_OUTPUTS = 4 + NUM_MEMORY  # rotation, speed, signal, memory, stop eating
BRAIN_TOPOLOGY = [BRAIN_INPUTS, 16, 8, BRAIN_OUTPUTS]

# --- Evolution ---
MUTATION_RATE = 0.08
MUTATION_AMOUNT = 0.15
GENE_MUTATION_AMOUNT = 0.10      # multiplicative, +-10% of the gene
ADDITIVE_GENE_MUTATION = {"color": 0.03}  # genes mutated by +-amount instead
# Speciation: mates must be within this RMS gene distance (each gene / its GENE_MIN_MAX range).
SPECIATION_DISTANCE = 0.12
SURVIVAL_RATE = 0.15
# Every ELITE_REINJECT_INTERVAL ticks the lowest-scoring ELITE_REINJECT_COUNT creatures of each
# species are replaced by the Hall of Fame brain and mutated copies (0 disables).
ELITE_REINJECT_INTERVAL = 0
ELITE_REINJECT_COUNT = 2
# Mate choice: attractiveness / default plus this weight times score / best candidate's score.
MATE_SCORE_WEIGHT = 0.0
HOF_SCORE_HALF_LIFE = 10000      # ticks; Hall of Fame and elite scores fade so newer brains win
ELITE_POOL_SIZE = 10             # best recently retired brains per species, bred after extinction
REPOPULATE_RANDOM_SHARE = 0.2    # share of an extinct species respawned with random brains
AUTOSAVE_INTERVAL = 100
BRAIN_HISTORY_KEEP = 10           # snapshots of earlier best brains kept in saved_brains/history
RESPAWN_IMMUNITY_TICKS = 500

# --- Colors (pygame only) ---
COLOR_CREATURE = (0, 255, 0)
COLOR_CARNIVORE = (255, 120, 120)
COLOR_BEST_CREATURE = (255, 255, 0)
COLOR_OBSTACLE = (60, 60, 60)
COLOR_TREE = (46, 110, 52)
COLOR_WHISKER = (0, 125, 0)
COLOR_CARNIVORE_WHISKER = (125, 0, 0)
COLOR_TARGET_WHISKER = (255, 220, 60)
COLOR_TEXT = (255, 255, 255)
COLOR_BIRTH = (0, 255, 255)
CHART_FONT_SIZE = 15         # at chart scale 1.0
CHART_SIZE = (150, 75)       # at chart scale 1.0
CHART_SCALE_DEFAULT = 2.0    # [ / ] in game.py zoom the charts
CHART_SCALE_MIN, CHART_SCALE_MAX, CHART_SCALE_STEP = 1.0, 3.0, 0.25
HUD_ALPHA = 60               # opacity of every HUD panel and chart (background, text, lines),
                             # 0 (clear) .. 255 (solid)
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
COLOR_BUCKETS = 11           # sprite tints for the color gene (odd: the default 0.5 is untinted)
COLOR_HUE_RANGE = 100        # degrees of hue shift across the color gene's range
NIGHT_MAX_ALPHA = 150
SIGNAL_RENDER_THRESHOLD = 0.6
COLOR_HIT = (255, 40, 30)
