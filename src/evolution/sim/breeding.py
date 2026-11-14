"""Mate seeking and reproduction."""
import math
import random
import time

import numpy as np
from numba import njit

from ..brain.layout import OUTPUTS as OUT
from ..config import (
    HUNGER_MAX,
    LITTER_SPREAD,
    MATE_ATTRACTION_RADIUS,
    MATE_SCORE_WEIGHT,
    MATE_SEEK_DANGER_THRESHOLD,
    MATE_SEEK_HUNGER_THRESHOLD,
    MATE_SEEK_MIN_SPEED,
    MATE_SELECTION_RADIUS,
    PARENTAL_INVESTMENT,
    REPRODUCTION_HEALTH_THRESHOLD,
    REPRODUCTION_POP_CAP_FACTOR,
)
from ..genetics import combine_brains, combine_genes, compatible, mutate_brain
from ..species import ALL_SPECIES
from ..world import creature as creature_module
from ..world import creature_table as T
from ..world.creature import Creature


def _ready_to_seek(c):
    return (c.health >= c.max_health * REPRODUCTION_HEALTH_THRESHOLD
            and c.hunger < HUNGER_MAX * MATE_SEEK_HUNGER_THRESHOLD)


def _compatible(a, b):
    hit = a.compatibility.get(b.id)
    if hit is not None and hit[0] is a.genes and hit[1] is b.genes:
        return hit[2]
    result = compatible(a.genes, b.genes, a.species)
    a.compatibility[b.id] = (a.genes, b.genes, result)
    b.compatibility[a.id] = (b.genes, a.genes, result)
    return result


def _scan_for_mate(c, kin_grid):
    """(mate, distance): the nearest compatible, ready adult among the kin that
    kin_grid.query(c.pos, MATE_ATTRACTION_RADIUS) returns, the first of equally near ones;
    (None, inf) when there is none."""
    pos = c.pos
    nearest, nearest_dist = None, math.inf
    for m in kin_grid.query(pos, MATE_ATTRACTION_RADIUS):
        if m is not c and m.is_adult and _ready_to_seek(m) and _compatible(c, m):
            d = pos.distance_to(m.pos)
            if d < nearest_dist:  # strict: the first of equally near mates wins, like min()
                nearest, nearest_dist = m, d
    return nearest, nearest_dist


@njit(cache=True, error_model="numpy")
def _closest_ready(x, y, ready, cell_x, cell_y, skip, px, py, x0, x1, y0, y1):
    """Index of the kin with the smallest squared distance to (px, py) among those that are
    ready, not skipped and in the cells x0..x1, y0..y1 (the first of equal ones), and that
    squared distance; (-1, inf) when there is none."""
    best, best_d2 = -1, math.inf
    for i in range(x.shape[0]):
        if (not ready[i] or skip[i] or cell_x[i] < x0 or cell_x[i] > x1
                or cell_y[i] < y0 or cell_y[i] > y1):
            continue
        dx = x[i] - px
        dy = y[i] - py
        d2 = dx * dx + dy * dy
        if d2 < best_d2:
            best, best_d2 = i, d2
    return best, best_d2


@njit(cache=True, error_model="numpy")
def _ready_within(x, y, ready, cell_x, cell_y, skip, px, py, x0, x1, y0, y1, limit):
    """Indices of the kin _closest_ready considers whose squared distance is <= limit."""
    found = []
    for i in range(x.shape[0]):
        if (not ready[i] or skip[i] or cell_x[i] < x0 or cell_x[i] > x1
                or cell_y[i] < y0 or cell_y[i] > y1):
            continue
        dx = x[i] - px
        dy = y[i] - py
        if dx * dx + dy * dy <= limit:
            found.append(i)
    return found


class MateRadar:
    """Answers nearest(c) as _scan_for_mate(c, kin_grid) would, without visiting every
    neighbour, for the creatures in `kin_grid` while they act one after another.

    acted(c) must follow every Creature.act. Between two of its own acts a creature may stop
    being a ready adult, but not become one."""

    def __init__(self, kin_grid):
        self._grid = kin_grid
        self._kin = None

    def _build(self):
        cells = [(cell, c) for cell, group in self._grid.grid.items() for c in group]
        self._kin = kin = [c for _, c in cells]
        self._cell_x = np.array([cell[0] for cell, _ in cells], dtype=np.int64)
        self._cell_y = np.array([cell[1] for cell, _ in cells], dtype=np.int64)
        self._slot = {c.id: i for i, c in enumerate(kin)}
        # Query order: by cell, then by insertion, which is how `cells` were just read.
        order = sorted(range(len(kin)), key=lambda i: cells[i][0])
        self._rank = [0] * len(kin)
        for rank, i in enumerate(order):
            self._rank[i] = rank
        self._x = np.array([c.pos.x for c in kin], dtype=np.float64)
        self._y = np.array([c.pos.y for c in kin], dtype=np.float64)
        self._ready = np.array([c.is_adult and _ready_to_seek(c) for c in kin], dtype=np.bool_)
        self._skip = np.zeros(len(kin), dtype=np.bool_)

    def acted(self, c):
        if self._kin is None:
            return
        i = self._slot[c.id]
        self._x[i], self._y[i] = c.pos.x, c.pos.y
        self._ready[i] = c.is_adult and _ready_to_seek(c)

    def nearest(self, c):
        if self._kin is None:
            self._build()
        pos, kin = c.pos, self._kin
        x0, x1, y0, y1 = self._grid.cell_range(pos, MATE_ATTRACTION_RADIUS)
        # Ready (as of its last act) and in range are checked in the kernels; the rest, and
        # readiness lost since, here. A creature that fails is skipped for the rest of the call.
        skip = self._skip
        skip[:] = False
        me = self._slot.get(c.id)
        if me is not None:
            skip[me] = True
        args = (self._x, self._y, self._ready, self._cell_x, self._cell_y, skip, pos.x, pos.y,
                x0, x1, y0, y1)
        while True:
            i, squared = _closest_ready(*args)
            if i < 0:
                return None, math.inf
            skip[i] = True
            m = kin[i]
            if m.is_adult and _ready_to_seek(m) and _compatible(c, m):
                break
        nearest, nearest_dist = m, pos.distance_to(m.pos)
        # Squared distances this close may round to the same distance: the scan order decides.
        ties = [j for j in _ready_within(*args, squared * (1 + 1e-9))
                if kin[j].is_adult and _ready_to_seek(kin[j]) and _compatible(c, kin[j])]
        if ties:
            nearest_dist = math.inf
            for j in sorted(ties + [i], key=self._rank.__getitem__):
                d = pos.distance_to(kin[j].pos)
                if d < nearest_dist:
                    nearest, nearest_dist = kin[j], d
        return nearest, nearest_dist


def mate_seeking_drive(c, outputs, kin_grid, radar=None):
    """outputs with steering blended toward the nearest compatible, ready adult of c's species
    (within MATE_ATTRACTION_RADIUS) while c is a fed, healthy, willing adult out of danger. The
    blend is c's mate_instinct gene; at 0 the brain steers alone. `radar` is a MateRadar of
    `kin_grid`."""
    blend = c.genes["mate_instinct"]
    if blend <= 0 or not c.willing:
        return outputs
    if not c.is_adult or c.hunger >= HUNGER_MAX * MATE_SEEK_HUNGER_THRESHOLD:
        return outputs
    if not c.species.is_predator and c.nearest_smell_strength >= MATE_SEEK_DANGER_THRESHOLD:
        return outputs
    if c.health < c.max_health * REPRODUCTION_HEALTH_THRESHOLD:
        return outputs
    nearest, nearest_dist = (radar.nearest(c) if radar is not None
                             else _scan_for_mate(c, kin_grid))
    if nearest is None or nearest_dist < 1:
        return outputs
    vec = nearest.pos - c.pos
    angle_diff = (math.atan2(vec.y, vec.x) - c.angle + math.pi) % (2 * math.pi) - math.pi
    turn = outputs[OUT.rotation]
    outputs[OUT.rotation] = turn * (1 - blend) + angle_diff / math.pi * blend
    outputs[OUT.speed] = max(outputs[OUT.speed], MATE_SEEK_MIN_SPEED * blend)
    return outputs


def stochastic_round(x):
    """floor(x), plus one with probability frac(x): 1.3 gives 2 young 30% of the time."""
    n = int(x)
    return n + (random.random() < x - n)


def mate_preference(mates, species):
    top_score = max((m.score for m in mates), default=0) or 1.0
    return lambda m: (m.display() / species.default_genes["attractiveness"]
                      + MATE_SCORE_WEIGHT * m.score / top_score)


def _ready(world, group):
    """(ready, xy): the willing, fertile members of `group` and their (N, 2) positions."""
    table = world.table
    rows = T.rows_in(table, group)
    if rows is None:
        ready = [p for p in group if p.willing and p.fertile(world.tick)]
        return ready, np.array([(p.pos.x, p.pos.y) for p in ready]).reshape(-1, 2)
    fl, it = table.fl[rows], table.it[rows]
    max_health = np.array([s.max_health for s in ALL_SPECIES], dtype=np.float64)
    mask = (it[:, T.WILLING] != 0) & T.fertile(
        fl, it, world.tick, max_health, creature_module.REPRODUCTION_HEALTH_THRESHOLD,
        creature_module.MAX_REPRODUCTIONS, creature_module.REPRODUCTION_COOLDOWN_TICKS)
    picked = np.flatnonzero(mask)
    return [group[i] for i in picked.tolist()], fl[np.ix_(picked, (T.X, T.Y))]


def breed(world):
    """Pair up nearby ready adults of each species and return their young (not yet added).
    Each parent gives PARENTAL_INVESTMENT of its health, and the litter splits the sum.

    The population cap counts every young born this tick, of either species."""
    offspring = []
    paired = set()
    now = time.time()
    for species in ALL_SPECIES:
        group = world.population(species)
        pop_cap = int(species.start_count * REPRODUCTION_POP_CAP_FACTOR)
        # Readiness only changes for creatures that pair this tick, which `paired` excludes.
        ready, xy = _ready(world, group)
        if len(ready) < 2:
            continue
        close = np.hypot(xy[:, None, 0] - xy[None, :, 0],
                         xy[:, None, 1] - xy[None, :, 1]) < MATE_SELECTION_RADIUS
        np.fill_diagonal(close, False)
        # Each row's neighbours in column order, as np.flatnonzero(close[i]) gives them.
        rows, cols = np.nonzero(close)
        bounds = np.searchsorted(rows, np.arange(len(ready) + 1)).tolist()
        cols = cols.tolist()
        for i, p1 in enumerate(ready):
            if p1.id in paired or len(group) + len(offspring) >= pop_cap:
                continue
            mates = [ready[j] for j in cols[bounds[i]:bounds[i + 1]]
                     if ready[j].id not in paired and _compatible(p1, ready[j])]
            if not mates:
                continue
            p2 = max(mates, key=mate_preference(mates, species))
            litter = min(stochastic_round((p1.genes["litter_size"] + p2.genes["litter_size"]) / 2),
                         pop_cap - len(group) - len(offspring))
            invested = 0.0
            for parent in (p1, p2):
                gift = parent.health * PARENTAL_INVESTMENT
                parent.health -= gift
                invested += gift
                parent.reproductions += 1
                parent.last_reproduction_tick = world.tick
            for _ in range(litter):
                offspring.append(_child(p1, p2, litter, invested / litter, now))
            paired.update((p1.id, p2.id))
    return offspring


def _child(p1, p2, litter, health, birth_time):
    genes = combine_genes(p1.genes, p2.genes, p1.species)
    brain = combine_brains(p1.brain, p2.brain)
    mutate_brain(brain, genes)
    child = Creature(p1.species, p1.world_bounds, brain=brain, genes=genes, geometry=p1.geometry)
    child.pos = (p1.pos + p2.pos) / 2
    if litter > 1:
        a = random.uniform(0, 2 * math.pi)
        child.pos += child.geometry.vec2(math.cos(a), math.sin(a)) * LITTER_SPREAD
    child.health = min(child.max_health, health)
    child.lineage = p1.lineage
    child.parent_ids = (p1.id, p2.id)
    child.descent = max(p1.descent, p2.descent) + 1
    child.generation = max(p1.generation, p2.generation) + 1
    child.birth_time = birth_time
    return child
