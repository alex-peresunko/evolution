"""Mate seeking and reproduction."""
import math
import random
import time

import numpy as np

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
        self._cells = [cell for cell, _ in cells]
        self._slot = {c.id: i for i, c in enumerate(kin)}
        # Query order: by cell, then by insertion, which is how `cells` were just read.
        order = sorted(range(len(kin)), key=lambda i: self._cells[i])
        self._rank = [0] * len(kin)
        for rank, i in enumerate(order):
            self._rank[i] = rank
        self._x = np.array([c.pos.x for c in kin], dtype=np.float64)
        self._y = np.array([c.pos.y for c in kin], dtype=np.float64)
        self._ready = [c.is_adult and _ready_to_seek(c) for c in kin]
        self._bar = np.where(self._ready, 0.0, np.inf)
        self._dx, self._dy = np.empty(len(kin)), np.empty(len(kin))

    def acted(self, c):
        if self._kin is None:
            return
        i = self._slot[c.id]
        self._x[i], self._y[i] = c.pos.x, c.pos.y
        ready = c.is_adult and _ready_to_seek(c)
        if ready != self._ready[i]:
            self._ready[i] = ready
            self._bar[i] = 0.0 if ready else np.inf

    def nearest(self, c):
        if self._kin is None:
            self._build()
        pos, kin, cells = c.pos, self._kin, self._cells
        min_x, max_x, min_y, max_y = self._grid.cell_range(pos, MATE_ATTRACTION_RADIUS)
        dx, dy = self._dx, self._dy
        np.subtract(self._x, pos.x, out=dx)
        np.subtract(self._y, pos.y, out=dy)
        np.multiply(dx, dx, out=dx)
        np.multiply(dy, dy, out=dy)
        np.add(dx, dy, out=dx)
        np.add(dx, self._bar, out=dx)
        dx[self._slot[c.id]] = np.inf

        def eligible(i):
            x, y = cells[i]
            m = kin[i]
            return (min_x <= x <= max_x and min_y <= y <= max_y
                    and m.is_adult and _ready_to_seek(m) and _compatible(c, m))

        while True:
            i = int(dx.argmin())
            squared = dx[i]
            if squared == np.inf:
                return None, math.inf
            dx[i] = np.inf
            if eligible(i):
                break
        nearest, nearest_dist = kin[i], pos.distance_to(kin[i].pos)
        # Squared distances this close may round to the same distance: the scan order decides.
        ties = [j for j in np.flatnonzero(dx <= squared * (1 + 1e-9)).tolist() if eligible(j)]
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
        ready = [p for p in group if p.willing and p.fertile(world.tick)]
        if len(ready) < 2:
            continue
        xy = np.array([(p.pos.x, p.pos.y) for p in ready])
        close = np.hypot(xy[:, None, 0] - xy[None, :, 0],
                         xy[:, None, 1] - xy[None, :, 1]) < MATE_SELECTION_RADIUS
        np.fill_diagonal(close, False)
        for i, p1 in enumerate(ready):
            if p1.id in paired or len(group) + len(offspring) >= pop_cap:
                continue
            mates = [ready[j] for j in np.flatnonzero(close[i])
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
