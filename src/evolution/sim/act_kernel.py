"""The act loop of Simulation._think_and_act, compiled with Numba.

For every creature in turn, the Python loop runs the mate-seeking drive (breeding), then
Creature.act, which calls the species' diet. act_all does the same in one Numba call over arrays
gathered from the creatures, food, carcasses and terrain, then writes the state back.

The Python code stays the reference and the specification: act_all must leave every creature,
food item and carcass, the tally and the `random` stream exactly as the Python loop would, bit
for bit and type for type (tests/sim/test_act_kernel.py runs both side by side). A rule changed
in one has to change in the other. What keeps them identical:
- grid queries visit cells and members in SpatialHashGrid.query's order (cells by x, then y;
  members by insertion), with the cells as of the last reindex;
- floats are computed with the same operations in the same order, using Python's float % and
  //; `**` goes through math.pow with a runtime exponent, because Numba would turn x ** 2 into
  x * x and x ** 0.5 into sqrt, which CPython does not;
- where Python can leave an int in a field (stamina, fat, score) an is-int flag follows it;
- random.random() comes from Numba's generator, which is the same Mersenne Twister: it is loaded
  with Python's state before the call and the state is handed back after;
- the geometry's arithmetic: Vector2.distance_to squares with ** (headless) or * (pygame), and
  pygame.Rect truncates to ints.
Config values reach the kernel as arguments, read when it is called (Numba would freeze module
globals, and its disk cache does not notice changes to other modules).
"""
import math
import random
from collections import Counter

import numpy as np
from numba import _helperlib, njit

from .. import diet as diet_module
from .. import genetics
from ..brain.layout import OUTPUTS as OUT
from ..diet import GrazingDiet, PredatorDiet
from ..species import ALL_SPECIES
from ..world import creature as creature_module
from ..world import food as food_module
from . import breeding

_JIT = {"cache": True, "error_model": "numpy"}
_INF = math.inf
_PI = math.pi
_TWO_PI = 2 * math.pi
NONE = -(2 ** 62)          # an int field that is None
GRAZER, PREDATOR = 0, 1    # diet kinds

# Float fields of a creature (array `fl`).
(X, Y, ANGLE, TURN, HEALTH, MAX_HEALTH, FAT, STAMINA, MAX_STAMINA, SIZE, MAX_SPEED, GEN_SIZE,
 EFF_MAX_SPEED, SPEED, HURT, MEAL_HEALTH, SCORE, TRAIT_UPKEEP, LIFESPAN, MATE_INSTINCT,
 TARGET_DIST, SMELL, SIGNAL) = range(23)
N_FL = 23
# Int fields of a creature (array `it`); bools as 0/1, None as NONE.
(SPECIES, AGE, THIRST, HUNGER, EATING, STUNNED, IMMUNE, ADULT, WINDED, IN_WATER, WILLING,
 MATE_OVERRIDE, LAST_BITE, LAST_HIT, LAST_TRAMPLE, LAST_TRAMPLED, MEAL_CARCASS, STAMINA_INT,
 FAT_INT, SCORE_INT, ACTS) = range(21)
N_IT = 21
CARCASS_OFF_LIST = -2      # MEAL_CARCASS of a carcass no longer in world.carcasses: it is gone

# Tally counters (array `tally`).
T_BITES, T_CONFUSED, T_PACK, T_TRAMPLES, T_MOB, T_FOOD = range(6)
# Outcomes of a bite.
_MISSED, _BITTEN, _CONFUSED = 0, 1, 2

# Config values (array `k`): (name, module the Python code reads it from).
_TUNABLES = [
    ("ADULTHOOD_AGE", creature_module), ("CONTACT_REACH", creature_module),
    ("CREATURE_ROTATION_SPEED", creature_module), ("DRINK_RATE", creature_module),
    ("FAT_BURN_PER_TICK", creature_module), ("FAT_MAX", creature_module),
    ("FAT_SIZE_FACTOR", creature_module), ("FAT_SPEED_PENALTY", creature_module),
    ("FAT_STORE_EFFICIENCY", creature_module), ("HEALTH_GAIN_SIZE_PENALTY", creature_module),
    ("HEALTH_LOSS_PER_TICK", creature_module), ("HEALTH_LOSS_SPEED_FACTOR", creature_module),
    ("HEALTH_LOST_ON_HIT", creature_module), ("HEALTH_SPEED_FLOOR", creature_module),
    ("HUNGER_HEALTH_PENALTY", creature_module), ("HUNGER_MAX", creature_module),
    ("JUVENILE_SIZE_FACTOR", creature_module), ("KLEIBER_EXPONENT", creature_module),
    ("MEAL_BREAK_THRESHOLD", creature_module), ("SENESCENCE_RATE", creature_module),
    ("SIGNAL_HEALTH_COST", creature_module), ("SPEED_STAMINA_EXPONENT", creature_module),
    ("STAMINA_DEPLETION_RATE", creature_module), ("STAMINA_EFFORT_THRESHOLD", creature_module),
    ("STAMINA_EXHAUSTION_PENALTY", creature_module), ("STAMINA_RECOVERY_SHARE", creature_module),
    ("STAMINA_REGEN_RATE", creature_module), ("THIRST_HEALTH_PENALTY", creature_module),
    ("THIRST_MAX", creature_module), ("TURN_RESPONSE", creature_module),
    ("WATER_SPEED_FACTOR", creature_module), ("WINTER_COLD_COST", creature_module),
    ("WINTER_FAT_INSULATION", creature_module),
    ("ADRENALINE_STAMINA", diet_module), ("BITE_COOLDOWN_TICKS", diet_module),
    ("BITE_DAMAGE", diet_module), ("BITE_HEALTH_TRANSFER", diet_module),
    ("BITE_SCORE", diet_module), ("CARCASS_EAT_RATE", diet_module),
    ("CARNIVORE_BITE_ANGLE", diet_module), ("CONFUSION_HERD_NORM", diet_module),
    ("CONFUSION_MAX_MISS", diet_module), ("CONFUSION_RADIUS", diet_module),
    ("CONTACT_REACH", diet_module), ("FOOD_RADIUS", diet_module),
    ("HERD_COURAGE_MAX", diet_module), ("MOB_ASSIST_SCORE", diet_module),
    ("MOB_BONUS", diet_module), ("MOB_WINDOW_TICKS", diet_module),
    ("PACK_ASSIST_SCORE", diet_module), ("PACK_BITE_BONUS", diet_module),
    ("PACK_WINDOW_TICKS", diet_module), ("RETALIATION_DAMAGE", diet_module),
    ("SCAVENGE_SCORE_FACTOR", diet_module), ("TRAMPLE_ANGLE", diet_module),
    ("TRAMPLE_COOLDOWN_TICKS", diet_module), ("TRAMPLE_DAMAGE", diet_module),
    ("TRAMPLE_MAX_SIZE_RATIO", diet_module), ("TRAMPLE_MAX_SPEED_RATIO", diet_module),
    ("TRAMPLE_MIN_SPEED", diet_module), ("TRAMPLE_REF_SPEED", diet_module),
    ("TRAMPLE_STUN_TICKS", diet_module), ("TRAMPLE_SCORE", diet_module),
    ("_MAX_SIZE", diet_module),
    ("HUNGER_MAX", breeding), ("MATE_ATTRACTION_RADIUS", breeding),
    ("MATE_SEEK_DANGER_THRESHOLD", breeding), ("MATE_SEEK_HUNGER_THRESHOLD", breeding),
    ("MATE_SEEK_MIN_SPEED", breeding), ("REPRODUCTION_HEALTH_THRESHOLD", breeding),
    ("SPECIATION_DISTANCE", genetics),
    ("FOOD_MAX_LIFETIME", food_module), ("FOOD_MIN_HEALTH_FACTOR", food_module),
    ("HERBIVORE_HEALTH_PER_FOOD", food_module), ("CARCASS_LIFETIME", food_module),
]
(K_ADULTHOOD_AGE, K_CONTACT_REACH, K_ROTATION_SPEED, K_DRINK_RATE, K_FAT_BURN, K_FAT_MAX,
 K_FAT_SIZE, K_FAT_SPEED, K_FAT_STORE, K_GAIN_SIZE_PENALTY, K_HEALTH_LOSS, K_LOSS_SPEED,
 K_LOST_ON_HIT, K_SPEED_FLOOR, K_HUNGER_PENALTY, K_HUNGER_MAX, K_JUVENILE, K_KLEIBER,
 K_MEAL_BREAK, K_SENESCENCE, K_SIGNAL_COST, K_STAMINA_EXPONENT, K_STAMINA_DEPLETION,
 K_EFFORT_THRESHOLD, K_EXHAUSTION, K_RECOVERY_SHARE, K_STAMINA_REGEN, K_THIRST_PENALTY,
 K_THIRST_MAX, K_TURN_RESPONSE, K_WATER_SPEED, K_WINTER_COLD, K_FAT_INSULATION,
 K_ADRENALINE, K_BITE_COOLDOWN, K_BITE_DAMAGE, K_BITE_TRANSFER, K_BITE_SCORE, K_CARCASS_EAT,
 K_BITE_ANGLE, K_HERD_NORM, K_MAX_MISS, K_CONFUSION_RADIUS, K_DIET_CONTACT, K_FOOD_RADIUS,
 K_HERD_COURAGE, K_MOB_ASSIST, K_MOB_BONUS, K_MOB_WINDOW, K_PACK_ASSIST, K_PACK_BONUS,
 K_PACK_WINDOW, K_RETALIATION, K_SCAVENGE_SCORE, K_TRAMPLE_ANGLE, K_TRAMPLE_COOLDOWN,
 K_TRAMPLE_DAMAGE, K_TRAMPLE_SIZE_RATIO, K_TRAMPLE_SPEED_RATIO, K_TRAMPLE_MIN_SPEED,
 K_TRAMPLE_REF_SPEED, K_TRAMPLE_STUN, K_TRAMPLE_SCORE, K_DIET_MAX_SIZE, K_SEEK_HUNGER_MAX,
 K_MATE_RADIUS,
 K_DANGER, K_SEEK_HUNGER, K_SEEK_MIN_SPEED, K_REPRO_HEALTH, K_SPECIATION, K_FOOD_LIFETIME,
 K_FOOD_MIN_HEALTH, K_FOOD_HEALTH, K_CARCASS_LIFETIME) = range(len(_TUNABLES))
# Derived in Python, as the Python code computes them.
(K_HURT_STEP, K_UPKEEP_BODY, K_REFUSE, K_TWO, K_HALF) = range(len(_TUNABLES),
                                                             len(_TUNABLES) + 5)


def _tunables():
    from . import simulation
    values = [getattr(module, name) for name, module in _TUNABLES]
    values += [1 / creature_module.HURT_FADE_TICKS, creature_module.UPKEEP_WEIGHTS["body"],
               simulation.MATE_REFUSE_THRESHOLD, 2.0, 0.5]
    return np.array(values, dtype=np.float64)


# --- arithmetic as the Python code does it ---------------------------------------------------

@njit(**_JIT)
def _sq(v, k):
    """CPython's float v ** 2: pow(|v|, 2.0), which is not always v * v."""
    return math.pow(abs(v), k[K_TWO])


@njit(**_JIT)
def _dist(ax, ay, bx, by, pygame, k):
    """Vector2.distance_to."""
    dx = ax - bx
    dy = ay - by
    if pygame:
        return math.sqrt(dx * dx + dy * dy)
    return math.sqrt(_sq(dx, k) + _sq(dy, k))


@njit(**_JIT)
def _pmax(a, b):
    """Python's max(a, b) for floats: a unless b > a."""
    return b if b > a else a


@njit(**_JIT)
def _pmin(a, b):
    """Python's min(a, b) for floats: a unless b < a."""
    return b if b < a else a


@njit(**_JIT)
def _query(px, py, radius, cell, cols, rows):
    """SpatialHashGrid.cell_range: (x0, x1, y0, y1)."""
    return (max(0, int((px - radius) / cell)), min(cols - 1, int((px + radius) / cell)),
            max(0, int((py - radius) / cell)), min(rows - 1, int((py + radius) / cell)))


@njit(**_JIT)
def _grid(xs, ys, member, cell, cols, rows):
    """A SpatialHashGrid of the rows where `member` holds, built in row order: (start, items),
    where the rows in cell (cx, cy) are items[start[cx * rows + cy]:start[cx * rows + cy + 1]].
    Cells are ordered by x, then y, so a query reads cells cx * rows + y0 .. y1 in one run."""
    n = xs.shape[0]
    cells = np.empty(n, dtype=np.int64)
    start = np.zeros(cols * rows + 1, dtype=np.int64)
    for i in range(n):
        if member[i]:
            cx = min(cols - 1, max(0, int(xs[i] / cell)))
            cy = min(rows - 1, max(0, int(ys[i] / cell)))
            cells[i] = cx * rows + cy
            start[cells[i] + 1] += 1
    for c in range(cols * rows):
        start[c + 1] += start[c]
    fill = start[:-1].copy()
    items = np.empty(start[-1], dtype=np.int64)
    for i in range(n):
        if member[i]:
            items[fill[cells[i]]] = i
            fill[cells[i]] += 1
    return start, items


# --- terrain -----------------------------------------------------------------------------------

@njit(**_JIT)
def _shore(lakes, waves, lk, theta):
    ripple = 0.0
    for h in range(waves.shape[1]):
        ripple = ripple + waves[lk, h, 1] * math.cos(waves[lk, h, 0] * theta + waves[lk, h, 2])
    return lakes[lk, 2] * (1 + ripple)


@njit(**_JIT)
def _lake_contains(lakes, waves, lk, x, y, k):
    dx = x - lakes[lk, 0]
    dy = y - lakes[lk, 1]
    d2 = dx * dx + dy * dy
    return d2 < lakes[lk, 4] and d2 < _sq(_shore(lakes, waves, lk, math.atan2(dy, dx)), k)


@njit(**_JIT)
def _lake_reaches(lakes, waves, lk, x, y, margin):
    dx = x - lakes[lk, 0]
    dy = y - lakes[lk, 1]
    d = math.hypot(dx, dy)
    return (d < lakes[lk, 3] + margin
            and d - _shore(lakes, waves, lk, math.atan2(dy, dx)) < margin)


@njit(**_JIT)
def _terrain_cell(x, y, t_cell, t_cols, t_rows):
    """TerrainGrid.near's cell."""
    return (min(t_cols - 1, max(0, int(x // t_cell))) * t_rows
            + min(t_rows - 1, max(0, int(y // t_cell))))


@njit(**_JIT)
def _hits_rect(rect, x, y, size, pygame):
    """rect.colliderect(the body's bounding rect)."""
    bx, by, bw, bh = x - size, y - size, size * 2, size * 2
    if pygame:
        bx, by, bw, bh = float(int(bx)), float(int(by)), float(int(bw)), float(int(bh))
        if bw == 0 or bh == 0 or rect[2] == 0 or rect[3] == 0:
            return False
    return (rect[0] < bx + bw and rect[0] + rect[2] > bx
            and rect[1] < by + bh and rect[1] + rect[3] > by)


# --- creature helpers --------------------------------------------------------------------------

@njit(**_JIT)
def _add_score(fl, it, i, amount, is_int):
    fl[i, SCORE] += amount
    if not is_int:
        it[i, SCORE_INT] = 0


@njit(**_JIT)
def _stop_eating(fl, it, i):
    it[i, EATING] = 0
    fl[i, MEAL_HEALTH] = 0.0
    it[i, MEAL_CARCASS] = -1


@njit(**_JIT)
def _absorb(fl, it, i, amount, k):
    """Creature.absorb: health up to max_health, the surplus to fat. Returns the gain."""
    gain = _pmax(0.0, _pmin(fl[i, MAX_HEALTH] - fl[i, HEALTH], amount))
    fl[i, HEALTH] += gain
    fat = fl[i, FAT] + (amount - gain) * k[K_FAT_STORE]
    if fat < k[K_FAT_MAX]:   # min(FAT_MAX, fat): the int FAT_MAX unless fat is below it
        fl[i, FAT] = fat
        it[i, FAT_INT] = 0
    else:
        fl[i, FAT] = k[K_FAT_MAX]
        it[i, FAT_INT] = 1
    return gain


@njit(**_JIT)
def _size_adjusted_gain(fl, i, amount, default_size, k):
    penalty = 1 + (fl[i, GEN_SIZE] - default_size) * k[K_GAIN_SIZE_PENALTY]
    return amount / _pmax(0.1, penalty)


@njit(**_JIT)
def _gain_meat(fl, it, i, amount, default_size, k):
    gain = _absorb(fl, it, i, _size_adjusted_gain(fl, i, amount, default_size, k), k)
    it[i, HUNGER] = 0
    return gain


@njit(**_JIT)
def _is_stuffed(fl, i, k):
    return fl[i, HEALTH] >= fl[i, MAX_HEALTH] and fl[i, FAT] >= k[K_FAT_MAX]


@njit(**_JIT)
def _touching(fl, a, b, pygame, k):
    return (_dist(fl[a, X], fl[a, Y], fl[b, X], fl[b, Y], pygame, k)
            < fl[a, SIZE] + fl[b, SIZE] + k[K_DIET_CONTACT])


@njit(**_JIT)
def _in_front(fl, a, b, arc):
    facing = (math.atan2(fl[b, Y] - fl[a, Y], fl[b, X] - fl[a, X]) - fl[a, ANGLE] + _PI) \
        % _TWO_PI - _PI
    return abs(facing) <= arc / 2


@njit(**_JIT)
def _reach(fl, i, k):
    """diet._reach."""
    return fl[i, SIZE] + k[K_DIET_MAX_SIZE] + k[K_DIET_CONTACT]


@njit(**_JIT)
def _herd_size(fl, i, start, items, cell, cols, rows, pygame, k):
    """Other live members of i's species grid within CONFUSION_RADIUS of i."""
    radius = k[K_CONFUSION_RADIUS]
    x0, x1, y0, y1 = _query(fl[i, X], fl[i, Y], radius, cell, cols, rows)
    count = 0
    if x0 > x1 or y0 > y1:
        return count
    for gx in range(x0, x1 + 1):
        for at in range(start[gx * rows + y0], start[gx * rows + y1 + 1]):
            o = items[at]
            if (o != i and fl[o, HEALTH] > 0
                    and _dist(fl[i, X], fl[i, Y], fl[o, X], fl[o, Y], pygame, k) < radius):
                count += 1
    return count


@njit(**_JIT)
def _compatible(genes, gene_span, n_genes, a, b, k):
    """genetics.compatible: RMS of the gene differences, each over its range."""
    if n_genes == 0:
        return 0.0 <= k[K_SPECIATION]
    total = 0.0
    for g in range(n_genes):
        total += _sq((genes[a, g] - genes[b, g]) / gene_span[g], k)
    return math.pow(total / n_genes, k[K_HALF]) <= k[K_SPECIATION]


@njit(**_JIT)
def _attackers(fl, i, prey, start, items, cell, cols, rows, pygame, k):
    """diet._attackers: carnivores of i's species grid in contact with prey, i included."""
    x0, x1, y0, y1 = _query(fl[prey, X], fl[prey, Y], _reach(fl, prey, k), cell, cols, rows)
    others = 0
    if x0 <= x1 and y0 <= y1:
        for gx in range(x0, x1 + 1):
            for at in range(start[gx * rows + y0], start[gx * rows + y1 + 1]):
                o = items[at]
                if o != i and fl[o, HEALTH] > 0 and _touching(fl, prey, o, pygame, k):
                    others += 1
    return 1 + others


@njit(**_JIT)
def _eat_from(fl, it, i, carcasses, kc, default_size, health_per_food, k):
    """PredatorDiet._eat_from."""
    eaten = _pmin(k[K_CARCASS_EAT], carcasses[kc, 3])
    carcasses[kc, 3] -= eaten
    gain = _gain_meat(fl, it, i, eaten, default_size, k)
    _add_score(fl, it, i, gain / health_per_food * k[K_SCAVENGE_SCORE], False)


@njit(**_JIT)
def _ready_to_seek(fl, it, m, k):
    return (fl[m, HEALTH] >= fl[m, MAX_HEALTH] * k[K_REPRO_HEALTH]
            and it[m, HUNGER] < k[K_SEEK_HUNGER_MAX] * k[K_SEEK_HUNGER])


# --- recent attackers (Creature.recent_biters / recent_tramplers as arrays) -------------------

@njit(**_JIT)
def _prune(who, when, count, j, tick, window):
    """diet._prune on row j: drop the entries older than window, keeping the order."""
    kept = 0
    for e in range(count[j]):
        if not tick - when[j, e] > window:
            who[j, kept] = who[j, e]
            when[j, kept] = when[j, e]
            kept += 1
    count[j] = kept


@njit(**_JIT)
def _any_other(who, count, j, i):
    for e in range(count[j]):
        if who[j, e] != i:
            return True
    return False


@njit(**_JIT)
def _record(who, when, count, j, i, tick):
    """recent[i] = tick on row j: updates i's entry in place or appends one. Returns the arrays,
    grown when row j was full."""
    for e in range(count[j]):
        if who[j, e] == i:
            when[j, e] = tick
            return who, when
    if count[j] == who.shape[1]:
        grown_who = np.empty((who.shape[0], who.shape[1] * 2), dtype=np.int64)
        grown_when = np.empty((who.shape[0], who.shape[1] * 2), dtype=np.int64)
        grown_who[:, :who.shape[1]] = who
        grown_when[:, :who.shape[1]] = when
        who, when = grown_who, grown_when
    who[j, count[j]] = i
    when[j, count[j]] = tick
    count[j] += 1
    return who, when


@njit(**_JIT)
def _assist(fl, it, who, count, j, i, score):
    for e in range(count[j]):
        other = who[j, e]
        if other != i and fl[other, HEALTH] > 0:
            _add_score(fl, it, other, score, False)


# --- the loop ----------------------------------------------------------------------------------

@njit(**_JIT)
def _act_all(rows, out_cols, fl, it, genes, gene_span, gene_count, n_acting,
             sp_kind, sp_prey, sp_predator, sp_eat_ticks, sp_swim, sp_default_size,
             sp_health_per_food, sp_tracks_smell, tick, season, world_w, world_h, pygame,
             cell, cols, rows_, food_xy, food_life, carcasses,
             t_cell, t_cols, t_rows, t_reach, t_obs_start, t_obs, t_lake_start, t_lake,
             rects, lakes, waves, rb_who, rb_when, rb_n, rt_who, rt_when, rt_n,
             rb_changed, rt_changed, k):
    """Mate-seeking drive and Creature.act for rows 0..n_acting-1 of fl/it, in order.
    Returns (eaten food, abandoned meals per species, order species first abandoned one, tally,
    random draws, rb_who, rb_when, rt_who, rt_when)."""
    out_rotation, out_speed, out_stop, out_mate = out_cols[0], out_cols[1], out_cols[2], \
        out_cols[3]
    n_species = sp_kind.shape[0]
    m_total = fl.shape[0]
    tally = np.zeros(6, dtype=np.int64)
    abandoned = np.zeros(n_species, dtype=np.int64)
    abandon_order = np.full(n_species, -1, dtype=np.int64)
    n_abandon_order = 0
    draws = 0
    eaten = np.zeros(food_xy.shape[0], dtype=np.bool_)

    # The grids as of the reindex: each species, food and carcasses.
    grid_start = np.zeros((n_species, cols * rows_ + 1), dtype=np.int64)
    grid_items = np.full((n_species, m_total), -1, dtype=np.int64)
    for s in range(n_species):
        member = np.zeros(m_total, dtype=np.bool_)
        for i in range(m_total):
            member[i] = it[i, ACTS] == 1 and it[i, SPECIES] == s
        start, items = _grid(fl[:, X], fl[:, Y], member, cell, cols, rows_)
        grid_start[s] = start
        grid_items[s, :items.shape[0]] = items
    food_start, food_items = _grid(food_xy[:, 0], food_xy[:, 1],
                                   np.ones(food_xy.shape[0], dtype=np.bool_), cell, cols, rows_)
    carc_start, carc_items = _grid(carcasses[:, 0], carcasses[:, 1],
                                   np.ones(carcasses.shape[0], dtype=np.bool_), cell, cols,
                                   rows_)

    for i in range(n_acting):
        s = it[i, SPECIES]
        rotation = rows[i, out_rotation]
        speed_out = rows[i, out_speed]

        # --- Simulation: willing, then breeding.mate_seeking_drive
        willing = rows[i, out_mate] >= k[K_REFUSE]
        it[i, WILLING] = 1 if willing else 0
        steer_rotation, steer_speed = rotation, speed_out
        blend = fl[i, MATE_INSTINCT]
        if (blend > 0 and willing and it[i, ADULT] == 1
                and it[i, HUNGER] < k[K_SEEK_HUNGER_MAX] * k[K_SEEK_HUNGER]
                and (sp_predator[s] or fl[i, SMELL] < k[K_DANGER])
                and fl[i, HEALTH] >= fl[i, MAX_HEALTH] * k[K_REPRO_HEALTH]):
            nearest, nearest_dist = -1, _INF
            x0, x1, y0, y1 = _query(fl[i, X], fl[i, Y], k[K_MATE_RADIUS], cell, cols, rows_)
            if x0 <= x1 and y0 <= y1:
                for gx in range(x0, x1 + 1):
                    for at in range(grid_start[s, gx * rows_ + y0],
                                    grid_start[s, gx * rows_ + y1 + 1]):
                        m = grid_items[s, at]
                        if (m != i and it[m, ADULT] == 1 and _ready_to_seek(fl, it, m, k)
                                and _compatible(genes, gene_span[s], gene_count[s], i, m, k)):
                            d = _dist(fl[i, X], fl[i, Y], fl[m, X], fl[m, Y], pygame, k)
                            if d < nearest_dist:
                                nearest, nearest_dist = m, d
            if nearest >= 0 and not nearest_dist < 1:
                angle_diff = (math.atan2(fl[nearest, Y] - fl[i, Y], fl[nearest, X] - fl[i, X])
                              - fl[i, ANGLE] + _PI) % _TWO_PI - _PI
                steer_rotation = rotation * (1 - blend) + angle_diff / _PI * blend
                steer_speed = _pmax(speed_out, k[K_SEEK_MIN_SPEED] * blend)
        it[i, MATE_OVERRIDE] = 1 if (steer_rotation != rotation
                                     or steer_speed != speed_out) else 0

        # --- Creature.act
        if fl[i, HURT] > 0:
            fl[i, HURT] = _pmax(0.0, fl[i, HURT] - k[K_HURT_STEP])
        if it[i, STUNNED] > 0:
            it[i, STUNNED] -= 1
        tc = _terrain_cell(fl[i, X], fl[i, Y], t_cell, t_cols, t_rows)

        # _grow
        it[i, AGE] += 1
        in_water = False
        for at in range(t_lake_start[tc], t_lake_start[tc + 1]):
            if _lake_contains(lakes, waves, t_lake[at], fl[i, X], fl[i, Y], k):
                in_water = True
                break
        it[i, IN_WATER] = 1 if in_water else 0
        if it[i, IMMUNE] > 0:
            it[i, IMMUNE] -= 1
        if it[i, ADULT] == 0 and it[i, AGE] >= k[K_ADULTHOOD_AGE]:
            it[i, ADULT] = 1
        if it[i, ADULT] == 1:
            growth = 1.0
        else:
            growth = k[K_JUVENILE] + (1 - k[K_JUVENILE]) * (it[i, AGE] / k[K_ADULTHOOD_AGE])
        fat_ratio = fl[i, FAT] / k[K_FAT_MAX]
        fl[i, SIZE] = fl[i, GEN_SIZE] * growth * (1 + fat_ratio * k[K_FAT_SIZE])
        fl[i, MAX_SPEED] = fl[i, EFF_MAX_SPEED] * growth

        # _drink
        drinks = in_water
        if not drinks:
            margin = fl[i, SIZE] + k[K_CONTACT_REACH]
            for at in range(t_lake_start[tc], t_lake_start[tc + 1]):
                if _lake_reaches(lakes, waves, t_lake[at], fl[i, X], fl[i, Y], margin):
                    drinks = True
                    break
        if drinks:
            it[i, THIRST] = max(0, it[i, THIRST] - int(k[K_DRINK_RATE]))
        else:
            it[i, THIRST] += 1

        if it[i, EATING] > 0 and rows[i, out_stop] > k[K_MEAL_BREAK]:
            _stop_eating(fl, it, i)
            if abandoned[s] == 0:
                abandon_order[n_abandon_order] = s
                n_abandon_order += 1
            abandoned[s] += 1

        # _move
        if fl[i, STAMINA] <= 0:
            it[i, WINDED] = 1
        elif it[i, WINDED] == 1 and fl[i, STAMINA] >= k[K_RECOVERY_SHARE] * fl[i, MAX_STAMINA]:
            it[i, WINDED] = 0
        effort = 0.0
        if it[i, EATING] > 0:
            speed = 0.0
            fl[i, TURN] = 0.0
        else:
            effort = _pmax(0.0, steer_speed)
            fl[i, TURN] += (steer_rotation - fl[i, TURN]) * k[K_TURN_RESPONSE]
            fl[i, ANGLE] = (fl[i, ANGLE] + fl[i, TURN] * k[K_ROTATION_SPEED]) % _TWO_PI
            multiplier = (k[K_SPEED_FLOOR]
                          + (1 - k[K_SPEED_FLOOR]) * fl[i, HEALTH] / fl[i, MAX_HEALTH])
            penalty = k[K_EXHAUSTION] if it[i, WINDED] == 1 else 1.0
            weight = 1 / (1 + fl[i, FAT] / k[K_FAT_MAX] * k[K_FAT_SPEED])
            water = k[K_WATER_SPEED] if in_water else 1.0
            speed = effort * fl[i, MAX_SPEED] * multiplier * penalty * weight * water
        if in_water:
            fl[i, STAMINA] -= sp_swim[s]
        elif effort > k[K_EFFORT_THRESHOLD] and it[i, WINDED] == 0:
            fl[i, STAMINA] -= math.pow(effort, k[K_STAMINA_EXPONENT]) * k[K_STAMINA_DEPLETION]
        else:
            fl[i, STAMINA] += k[K_STAMINA_REGEN]
        clamped = _pmin(fl[i, STAMINA], fl[i, MAX_STAMINA])
        if clamped > 0:                 # max(0, clamped): the int 0 unless clamped is above it
            fl[i, STAMINA] = clamped
            it[i, STAMINA_INT] = 0
        else:
            fl[i, STAMINA] = 0.0
            it[i, STAMINA_INT] = 1
        step_x = math.cos(fl[i, ANGLE]) * speed
        step_y = math.sin(fl[i, ANGLE]) * speed
        fl[i, X] += step_x
        fl[i, Y] += step_y
        size = fl[i, SIZE]
        if not (size <= fl[i, X] <= world_w - size and size <= fl[i, Y] <= world_h - size):
            fl[i, HEALTH] -= k[K_LOST_ON_HIT]
            fl[i, X] -= step_x * 3
            fl[i, Y] -= step_y * 3
            fl[i, X] = _pmin(_pmax(fl[i, X], size), world_w - size)
            fl[i, Y] = _pmin(_pmax(fl[i, Y], size), world_h - size)

        # _metabolize
        it[i, HUNGER] += 1
        hunger = it[i, HUNGER]
        if hunger > k[K_HUNGER_MAX]:
            hunger_penalty = k[K_HUNGER_PENALTY] * _pmin(1.0, _pmax(
                0.0, (hunger - k[K_HUNGER_MAX]) / k[K_HUNGER_MAX]))
        else:
            hunger_penalty = 0.0
        fl[i, FAT] = _pmax(0.0, fl[i, FAT] - k[K_FAT_BURN])
        it[i, FAT_INT] = 0
        from_fat = _pmin(fl[i, FAT], hunger_penalty)
        fl[i, FAT] -= from_fat
        hunger_penalty -= from_fat
        dehydration = k[K_THIRST_PENALTY] * _pmin(1.0, _pmax(
            0.0, it[i, THIRST] / k[K_THIRST_MAX] - 1))
        body = math.pow(fl[i, GEN_SIZE] * growth / sp_default_size[s], k[K_KLEIBER])
        upkeep = k[K_HEALTH_LOSS] * (k[K_UPKEEP_BODY] * body + fl[i, TRAIT_UPKEEP])
        lifespan = fl[i, LIFESPAN]
        senescence = k[K_SENESCENCE] * _pmax(0.0, it[i, AGE] - lifespan) / lifespan
        cold = (k[K_WINTER_COLD] * _pmax(0.0, -season)
                * (1 - k[K_FAT_INSULATION] * fl[i, FAT] / k[K_FAT_MAX]))
        fl[i, HEALTH] -= (upkeep + senescence + cold + (speed * k[K_LOSS_SPEED])
                          + hunger_penalty + dehydration + fl[i, SIGNAL] * k[K_SIGNAL_COST])
        fl[i, SPEED] = speed

        # --- diet.feed
        if sp_kind[s] == GRAZER:
            if it[i, EATING] > 0:                       # Creature.digest
                portion = fl[i, MEAL_HEALTH] / it[i, EATING]
                _absorb(fl, it, i, portion, k)
                fl[i, MEAL_HEALTH] -= portion
                it[i, EATING] -= 1
                it[i, HUNGER] = 0
            else:
                if (it[i, ADULT] == 1 and fl[i, SPEED] >= k[K_TRAMPLE_MIN_SPEED]
                        and (it[i, LAST_TRAMPLE] == NONE
                             or tick - it[i, LAST_TRAMPLE] >= k[K_TRAMPLE_COOLDOWN])):
                    # GrazingDiet._trample: the first live predator in contact and in front
                    trampled = False
                    reach = _reach(fl, i, k)
                    x0, x1, y0, y1 = _query(fl[i, X], fl[i, Y], reach, cell, cols, rows_)
                    for ps in range(n_species):
                        if trampled or not sp_predator[ps] or x0 > x1 or y0 > y1:
                            continue
                        for gx in range(x0, x1 + 1):
                            if trampled:
                                break
                            for at in range(grid_start[ps, gx * rows_ + y0],
                                            grid_start[ps, gx * rows_ + y1 + 1]):
                                foe = grid_items[ps, at]
                                if (fl[foe, HEALTH] <= 0 or not _touching(fl, i, foe, pygame, k)
                                        or not _in_front(fl, i, foe, k[K_TRAMPLE_ANGLE])):
                                    continue
                                if fl[foe, SIZE] > 0:
                                    size_ratio = fl[i, SIZE] / fl[foe, SIZE]
                                else:
                                    size_ratio = k[K_TRAMPLE_SIZE_RATIO]
                                herd = _herd_size(fl, i, grid_start[s], grid_items[s], cell,
                                                  cols, rows_, pygame, k)
                                damage = (k[K_TRAMPLE_DAMAGE]
                                          * _pmin(k[K_TRAMPLE_SIZE_RATIO], _pmax(1.0, size_ratio))
                                          * _pmin(k[K_TRAMPLE_SPEED_RATIO],
                                                  fl[i, SPEED] / k[K_TRAMPLE_REF_SPEED])
                                          * (1 + k[K_HERD_COURAGE]
                                             * _pmin(1.0, herd / k[K_HERD_NORM])))
                                _prune(rt_who, rt_when, rt_n, foe, tick, k[K_MOB_WINDOW])
                                rt_changed[foe] = True
                                if _any_other(rt_who, rt_n, foe, i):
                                    damage *= k[K_MOB_BONUS]
                                    tally[T_MOB] += 1
                                rt_who, rt_when = _record(rt_who, rt_when, rt_n, foe, i, tick)
                                tally[T_TRAMPLES] += 1
                                it[i, LAST_TRAMPLE] = tick
                                fl[foe, HEALTH] -= damage
                                fl[foe, HURT] = 1.0
                                it[foe, LAST_HIT] = tick
                                it[foe, LAST_TRAMPLED] = tick
                                it[foe, STUNNED] = int(k[K_TRAMPLE_STUN])
                                _stop_eating(fl, it, foe)
                                _add_score(fl, it, i, k[K_TRAMPLE_SCORE], False)
                                if fl[foe, HEALTH] <= 0:
                                    _add_score(fl, it, i, 1.0, True)
                                    _assist(fl, it, rt_who, rt_n, foe, i, k[K_MOB_ASSIST])
                                trampled = True
                                break
                # the first uneaten food in reach
                x0, x1, y0, y1 = _query(fl[i, X], fl[i, Y], _reach(fl, i, k), cell, cols, rows_)
                found = -1
                if x0 <= x1 and y0 <= y1:
                    for gx in range(x0, x1 + 1):
                        if found >= 0:
                            break
                        for at in range(food_start[gx * rows_ + y0],
                                        food_start[gx * rows_ + y1 + 1]):
                            f = food_items[at]
                            if eaten[f]:
                                continue
                            if (_dist(fl[i, X], fl[i, Y], food_xy[f, 0], food_xy[f, 1], pygame,
                                      k) < fl[i, SIZE] + k[K_FOOD_RADIUS]):
                                found = f
                                break
                if found >= 0:
                    eaten[found] = True
                    tally[T_FOOD] += 1
                    life = food_life[found]
                    if life >= k[K_FOOD_LIFETIME]:
                        value = 0.0
                    else:
                        freshness = 1.0 - (life / k[K_FOOD_LIFETIME])
                        value = float(int(k[K_FOOD_HEALTH] * (
                            k[K_FOOD_MIN_HEALTH] + (1 - k[K_FOOD_MIN_HEALTH]) * freshness)))
                    gain = _size_adjusted_gain(fl, i, value, sp_default_size[s], k)
                    if sp_eat_ticks[s] > 0:
                        fl[i, MEAL_HEALTH] = gain
                        it[i, EATING] = sp_eat_ticks[s]
                    else:
                        _absorb(fl, it, i, gain, k)
                    it[i, HUNGER] = 0
                    _add_score(fl, it, i, 1.0, True)
        else:
            if it[i, EATING] > 0:                       # PredatorDiet._continue_meal
                kc = it[i, MEAL_CARCASS]
                if (kc < 0 or carcasses[kc, 3] <= 0 or carcasses[kc, 4] >= k[K_CARCASS_LIFETIME]
                        or _is_stuffed(fl, i, k)):
                    _stop_eating(fl, it, i)
                else:
                    _eat_from(fl, it, i, carcasses, kc, sp_default_size[s],
                              sp_health_per_food[s], k)
                    it[i, EATING] -= 1
                    if it[i, EATING] == 0:
                        it[i, MEAL_CARCASS] = -1
            elif it[i, IMMUNE] == 0:
                outcome = _MISSED
                if it[i, STUNNED] == 0 and (it[i, LAST_BITE] == NONE
                                            or tick - it[i, LAST_BITE] >= k[K_BITE_COOLDOWN]):
                    # PredatorDiet._bite: the first live prey in contact and in front
                    p = sp_prey[s]
                    x0, x1, y0, y1 = _query(fl[i, X], fl[i, Y], _reach(fl, i, k), cell, cols,
                                            rows_)
                    if x0 <= x1 and y0 <= y1:
                        for gx in range(x0, x1 + 1):
                            if outcome != _MISSED:
                                break
                            for at in range(grid_start[p, gx * rows_ + y0],
                                            grid_start[p, gx * rows_ + y1 + 1]):
                                prey = grid_items[p, at]
                                if (fl[prey, HEALTH] <= 0
                                        or not _touching(fl, i, prey, pygame, k)
                                        or not _in_front(fl, i, prey, k[K_BITE_ANGLE])):
                                    continue
                                it[i, LAST_BITE] = tick
                                # _confused
                                if k[K_MAX_MISS] > 0:
                                    herd = _herd_size(fl, prey, grid_start[p], grid_items[p],
                                                      cell, cols, rows_, pygame, k)
                                    if herd != 0:
                                        draws += 1
                                        if random.random() < k[K_MAX_MISS] * _pmin(
                                                1.0, herd / k[K_HERD_NORM]):
                                            outcome = _CONFUSED
                                            break
                                _prune(rb_who, rb_when, rb_n, prey, tick, k[K_PACK_WINDOW])
                                rb_changed[prey] = True
                                if fl[prey, SIZE] > 0:
                                    ratio = fl[i, SIZE] / fl[prey, SIZE]
                                else:
                                    ratio = 2.0
                                damage = k[K_BITE_DAMAGE] * _pmin(2.0, _pmax(0.5, ratio))
                                if _any_other(rb_who, rb_n, prey, i):
                                    damage *= k[K_PACK_BONUS]
                                    tally[T_PACK] += 1
                                rb_who, rb_when = _record(rb_who, rb_when, rb_n, prey, i, tick)
                                fl[prey, HEALTH] -= damage
                                boosted = fl[prey, STAMINA] + k[K_ADRENALINE]
                                if boosted < fl[prey, MAX_STAMINA]:
                                    fl[prey, STAMINA] = boosted   # keeps the int-ness
                                else:
                                    fl[prey, STAMINA] = fl[prey, MAX_STAMINA]
                                    it[prey, STAMINA_INT] = 0
                                it[prey, LAST_HIT] = tick
                                fl[prey, HURT] = 1.0
                                _stop_eating(fl, it, prey)
                                _gain_meat(fl, it, i, damage * k[K_BITE_TRANSFER],
                                           sp_default_size[s], k)
                                _add_score(fl, it, i, k[K_BITE_SCORE], False)
                                if fl[prey, HEALTH] <= 0:
                                    _add_score(fl, it, i, 1.0, True)
                                    _assist(fl, it, rb_who, rb_n, prey, i, k[K_PACK_ASSIST])
                                retaliation = k[K_RETALIATION] * _pmin(
                                    3.0, _pmax(0.0, 1 / ratio - 1))
                                if retaliation > 0:
                                    fl[i, HEALTH] -= retaliation / _attackers(
                                        fl, i, prey, grid_start[s], grid_items[s], cell, cols,
                                        rows_, pygame, k)
                                    fl[i, HURT] = 1.0
                                it[i, LAST_HIT] = tick
                                outcome = _BITTEN
                                break
                if outcome == _BITTEN:
                    tally[T_BITES] += 1
                elif outcome == _CONFUSED:
                    tally[T_CONFUSED] += 1
                else:
                    # PredatorDiet._scavenge: the first carcass in contact
                    x0, x1, y0, y1 = _query(fl[i, X], fl[i, Y], _reach(fl, i, k), cell, cols,
                                            rows_)
                    done = False
                    if x0 <= x1 and y0 <= y1:
                        for gx in range(x0, x1 + 1):
                            if done:
                                break
                            for at in range(carc_start[gx * rows_ + y0],
                                            carc_start[gx * rows_ + y1 + 1]):
                                kc = carc_items[at]
                                if (carcasses[kc, 3] <= 0
                                        or _dist(fl[i, X], fl[i, Y], carcasses[kc, 0],
                                                 carcasses[kc, 1], pygame, k)
                                        >= fl[i, SIZE] + carcasses[kc, 2] + k[K_DIET_CONTACT]):
                                    continue
                                _eat_from(fl, it, i, carcasses, kc, sp_default_size[s],
                                          sp_health_per_food[s], k)
                                if (sp_eat_ticks[s] > 0 and not (
                                        carcasses[kc, 3] <= 0
                                        or carcasses[kc, 4] >= k[K_CARCASS_LIFETIME])
                                        and not _is_stuffed(fl, i, k)):
                                    it[i, MEAL_CARCASS] = kc
                                    it[i, EATING] = sp_eat_ticks[s]
                                done = True
                                break

        # --- Creature._collide
        size = fl[i, SIZE]
        x, y = fl[i, X], fl[i, Y]
        if size > t_reach:
            first, last, use_all = 0, rects.shape[0], True
        else:
            tc = _terrain_cell(x, y, t_cell, t_cols, t_rows)
            first, last, use_all = t_obs_start[tc], t_obs_start[tc + 1], False
        for at in range(first, last):
            o = at if use_all else t_obs[at]
            if _hits_rect(rects[o], x, y, size, pygame):
                fl[i, HEALTH] -= k[K_LOST_ON_HIT]
                fl[i, X] -= math.cos(fl[i, ANGLE]) * fl[i, SPEED] * 3
                fl[i, Y] -= math.sin(fl[i, ANGLE]) * fl[i, SPEED] * 3
                break

        # --- Creature._reward_pursuit
        if fl[i, TARGET_DIST] == fl[i, TARGET_DIST]:     # not NaN: a target in sight
            _add_score(fl, it, i, 0.001 * (1.0 - fl[i, TARGET_DIST]), False)
        elif sp_tracks_smell[s] and fl[i, SMELL] > 0:
            _add_score(fl, it, i, 0.0005 * fl[i, SMELL], False)

    return (eaten, abandoned, abandon_order[:n_abandon_order], tally, draws,
            rb_who, rb_when, rt_who, rt_when)


# --- Python side: gather, run, write back ------------------------------------------------------

class _Terrain:
    """The world's obstacles, lakes and TerrainGrid as arrays, rebuilt when the terrain object
    changes (World rebuilds it when obstacles or lakes are replaced)."""

    def __init__(self):
        self._terrain = None

    def get(self, world):
        terrain = world.terrain
        if terrain is not self._terrain:
            self._terrain = terrain
            obstacles, lakes = world.obstacles, world.lakes
            obstacle_at = {id(o): n for n, o in enumerate(obstacles)}
            lake_at = {id(lake): n for n, lake in enumerate(lakes)}
            obs_start, obs, lake_start, lake = [0], [], [0], []
            for cx in range(terrain.cols):
                for cy in range(terrain.rows):
                    near_obstacles, near_lakes = terrain.cells[cx, cy]
                    obs += [obstacle_at[id(o)] for o in near_obstacles]
                    lake += [lake_at[id(blob)] for blob in near_lakes]
                    obs_start.append(len(obs))
                    lake_start.append(len(lake))
            self.obs_start = np.array(obs_start, dtype=np.int64)
            self.obs = np.array(obs, dtype=np.int64)
            self.lake_start = np.array(lake_start, dtype=np.int64)
            self.lake = np.array(lake, dtype=np.int64)
            self.rects = np.array([(o.rect.x, o.rect.y, o.rect.w, o.rect.h) for o in obstacles],
                                  dtype=np.float64).reshape(-1, 4)
            # max_radius ** 2 as Python computes it (pow, not a product).
            self.lakes = np.array([(b.x, b.y, b.radius, b.max_radius, b.max_radius ** 2)
                                   for b in lakes], dtype=np.float64).reshape(-1, 5)
            harmonics = max((len(b.harmonics) for b in lakes), default=0)
            self.waves = np.zeros((len(lakes), harmonics, 3))
            for n, b in enumerate(lakes):
                if b.harmonics:
                    self.waves[n, :len(b.harmonics)] = b.harmonics
        return self


_terrain = _Terrain()
_SPECIES_AT = {id(s): n for n, s in enumerate(ALL_SPECIES)}
_OUT_COLUMNS = np.array([OUT.rotation, OUT.speed, OUT.stop_eating, OUT.mate], dtype=np.int64)


def _species_table():
    kind, prey, predator, eat_ticks, swim, default_size, health_per_food, smell = \
        [], [], [], [], [], [], [], []
    for s in ALL_SPECIES:
        if type(s.diet) is GrazingDiet:
            kind.append(GRAZER)
            prey.append(-1)
        elif type(s.diet) is PredatorDiet and id(s.diet.prey) in _SPECIES_AT:
            kind.append(PREDATOR)
            prey.append(_SPECIES_AT[id(s.diet.prey)])
        else:
            return None
        predator.append(s.is_predator)
        eat_ticks.append(s.eat_ticks)
        swim.append(s.swim_stamina_cost)
        default_size.append(s.default_genes["size"])
        health_per_food.append(s.health_per_food)
        smell.append(s.diet.tracks_by_smell)
    return (np.array(kind, dtype=np.int64), np.array(prey, dtype=np.int64),
            np.array(predator, dtype=np.bool_), np.array(eat_ticks, dtype=np.int64),
            np.array(swim, dtype=np.float64), np.array(default_size, dtype=np.float64),
            np.array(health_per_food, dtype=np.float64), np.array(smell, dtype=np.bool_))


def supported(world):
    """Whether act_all reproduces the Python loop for this world: a geometry whose arithmetic
    it knows and the diets it implements."""
    return world.geometry.kind in ("headless", "pygame") and _species_table() is not None


def _tick(value):
    return NONE if value is None else value


def _gene_keys():
    return [list(s.gene_bounds) for s in ALL_SPECIES]


def _gather(world, creatures):
    """(members, fl, it, genes): the acting creatures, then every other creature their
    recent_biters / recent_tramplers name (their score can change)."""
    members = list(creatures)
    slot = {id(c): n for n, c in enumerate(members)}
    for c in creatures:
        for recent in (c.recent_biters, c.recent_tramplers):
            for other in recent:
                if id(other) not in slot:
                    slot[id(other)] = len(members)
                    members.append(other)
    acting = len(creatures)
    carcass_at = {id(k): n for n, k in enumerate(world.carcasses)}
    nan = math.nan
    fl = np.array([
        (c.pos.x, c.pos.y, c.angle, c.turn, c.health, c.max_health, c.fat, c.stamina,
         c.max_stamina, c.size, c.max_speed, c.genetic_size, c.effective_max_speed,
         c.current_speed, c.hurt, c.meal_health, c.score, c.trait_upkeep, c.genes["lifespan"],
         c.genes["mate_instinct"],
         nan if c.nearest_target_dist is None else c.nearest_target_dist,
         c.nearest_smell_strength, c.signal)
        for c in members], dtype=np.float64).reshape(-1, N_FL)
    it = np.array([
        (_SPECIES_AT[id(c.species)], c.age, c.thirst, c.hunger, c.eating_ticks, c.stunned_ticks,
         c.respawn_immunity_ticks, c.is_adult, c.winded, c.in_water, c.willing, c.mate_override,
         _tick(c.last_bite_tick), _tick(c.last_hit_tick), _tick(c.last_trample_tick),
         _tick(c.last_trampled_tick),
         -1 if c.meal_carcass is None else carcass_at.get(id(c.meal_carcass), CARCASS_OFF_LIST),
         type(c.stamina) is int, type(c.fat) is int, type(c.score) is int, n < acting)
        for n, c in enumerate(members)], dtype=np.int64).reshape(-1, N_IT)
    keys = _gene_keys()
    width = max(len(k) for k in keys)
    genes = np.zeros((len(members), width))
    for n, c in enumerate(members):
        row = [c.genes[g] for g in keys[_SPECIES_AT[id(c.species)]]]
        genes[n, :len(row)] = row
    return members, slot, fl, it, genes


def _recent(members, slot, attribute):
    """(who, when, count) of each acting member's recent attackers dict."""
    dicts = [getattr(c, attribute) for c in members]
    width = max(4, max(len(d) for d in dicts) + 4)
    who = np.full((len(members), width), -1, dtype=np.int64)
    when = np.zeros((len(members), width), dtype=np.int64)
    count = np.zeros(len(members), dtype=np.int64)
    for j, d in enumerate(dicts):
        for e, (other, tick) in enumerate(d.items()):
            who[j, e] = slot[id(other)]
            when[j, e] = tick
        count[j] = len(d)
    return who, when, count


def _write_recent(members, attribute, who, when, count, changed):
    for j in np.flatnonzero(changed).tolist():
        d = getattr(members[j], attribute)
        d.clear()
        for e in range(count[j]):
            d[members[who[j, e]]] = int(when[j, e])


class _Random:
    """Moves the state of Python's `random` into Numba's generator and back."""

    def __init__(self):
        self._ptr = _helperlib.rnd_get_py_state_ptr()

    def load(self):
        version, state, gauss = random.getstate()
        _helperlib.rnd_set_state(self._ptr, (state[-1], list(state[:-1])))
        return version, gauss

    def store(self, version, gauss):
        index, words = _helperlib.rnd_get_state(self._ptr)
        random.setstate((version, tuple(words) + (index,), gauss))


_random = None


def act_all(world, creatures, outputs):
    """The act loop of Simulation._think_and_act for `creatures` (world.creatures()) with the
    brain `outputs` rows, compiled. The world must be `supported`."""
    global _random
    if _random is None:
        _random = _Random()
    species = _species_table()
    members, slot, fl, it, genes = _gather(world, creatures)
    keys = [list(s.gene_bounds.values()) for s in ALL_SPECIES]
    width = genes.shape[1]
    gene_span = np.ones((len(ALL_SPECIES), width))
    for n, bounds in enumerate(keys):
        gene_span[n, :len(bounds)] = [hi - lo for lo, hi in bounds]
    gene_count = np.array([len(b) for b in keys], dtype=np.int64)
    rb = _recent(members, slot, "recent_biters")
    rt = _recent(members, slot, "recent_tramplers")
    rb_changed = np.zeros(len(members), dtype=np.bool_)
    rt_changed = np.zeros(len(members), dtype=np.bool_)

    food = world.food
    food_xy = world.index.food_xy(food)
    if food_xy is None:
        food_xy = np.array([(f.pos.x, f.pos.y) for f in food], dtype=np.float64).reshape(-1, 2)
    food_life = np.array([f.lifetime for f in food], dtype=np.float64)
    carcasses = np.array([(k.pos.x, k.pos.y, k.radius, k.meat, k.age) for k in world.carcasses],
                         dtype=np.float64).reshape(-1, 5)
    grid = world.index.food
    terrain = _terrain.get(world)
    t = world.terrain
    w, h = world.bounds

    version, gauss = _random.load()
    (eaten, abandoned, abandon_order, tally, draws, rb_who, rb_when, rt_who, rt_when) = _act_all(
        np.ascontiguousarray(outputs, dtype=np.float64), _OUT_COLUMNS, fl, it, genes, gene_span,
        gene_count, len(creatures), *species, world.tick, float(world.season), float(w),
        float(h), world.geometry.kind == "pygame", float(grid.cell_size), grid.cols, grid.rows,
        np.ascontiguousarray(food_xy, dtype=np.float64), food_life, carcasses,
        float(t.cell_size), t.cols, t.rows, float(t.reach), terrain.obs_start, terrain.obs,
        terrain.lake_start, terrain.lake, terrain.rects, terrain.lakes, terrain.waves,
        rb[0], rb[1], rb[2], rt[0], rt[1], rt[2], rb_changed, rt_changed, _tunables())
    if draws:
        _random.store(version, gauss)

    _write_back(members, len(creatures), fl, it, world.carcasses)
    _write_recent(members, "recent_biters", rb_who, rb_when, rb[2], rb_changed)
    _write_recent(members, "recent_tramplers", rt_who, rt_when, rt[2], rt_changed)
    for f in np.flatnonzero(eaten).tolist():
        food[f].consumed = True
    for k, meat in zip(world.carcasses, carcasses[:, 3].tolist()):
        k.meat = meat
    tallied = world.tally
    tallied.bites += int(tally[T_BITES])
    tallied.bites_confused += int(tally[T_CONFUSED])
    tallied.pack_bites += int(tally[T_PACK])
    tallied.tramples += int(tally[T_TRAMPLES])
    tallied.mob_tramples += int(tally[T_MOB])
    tallied.food_eaten += int(tally[T_FOOD])
    meals = tallied.meals_abandoned
    if not isinstance(meals, Counter):
        meals = tallied.meals_abandoned = Counter(meals)
    for s in abandon_order.tolist():
        meals[ALL_SPECIES[s].name] += int(abandoned[s])


def _write_back(members, acting, fl, it, carcasses):
    fls, its = fl.tolist(), it.tolist()
    for n in range(acting):
        c, f, i = members[n], fls[n], its[n]
        c.pos.x, c.pos.y = f[X], f[Y]
        c.angle, c.turn, c.health = f[ANGLE], f[TURN], f[HEALTH]
        c.fat = int(f[FAT]) if i[FAT_INT] else f[FAT]
        c.stamina = int(f[STAMINA]) if i[STAMINA_INT] else f[STAMINA]
        c.score = int(f[SCORE]) if i[SCORE_INT] else f[SCORE]
        c.size, c.max_speed, c.current_speed = f[SIZE], f[MAX_SPEED], f[SPEED]
        c.hurt, c.meal_health = f[HURT], f[MEAL_HEALTH]
        c.age, c.thirst, c.hunger = i[AGE], i[THIRST], i[HUNGER]
        c.eating_ticks, c.stunned_ticks = i[EATING], i[STUNNED]
        c.respawn_immunity_ticks = i[IMMUNE]
        c.is_adult, c.winded, c.in_water = bool(i[ADULT]), bool(i[WINDED]), bool(i[IN_WATER])
        c.willing, c.mate_override = bool(i[WILLING]), bool(i[MATE_OVERRIDE])
        c.last_bite_tick = None if i[LAST_BITE] == NONE else i[LAST_BITE]
        c.last_hit_tick = None if i[LAST_HIT] == NONE else i[LAST_HIT]
        c.last_trample_tick = None if i[LAST_TRAMPLE] == NONE else i[LAST_TRAMPLE]
        c.last_trampled_tick = None if i[LAST_TRAMPLED] == NONE else i[LAST_TRAMPLED]
        meal = i[MEAL_CARCASS]
        if meal == -1:
            c.meal_carcass = None
        elif meal >= 0:
            c.meal_carcass = carcasses[meal]
    for n in range(acting, len(members)):   # only their score can change
        c, f, i = members[n], fls[n], its[n]
        c.score = int(f[SCORE]) if i[SCORE_INT] else f[SCORE]
