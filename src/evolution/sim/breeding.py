"""Mate seeking and reproduction."""
import math
import random
import time

import numpy as np

from ..brain.layout import OUTPUTS as OUT
from ..config import (
    HUNGER_MAX,
    LITTER_ENDOWMENT,
    LITTER_SPREAD,
    MATE_ATTRACTION_RADIUS,
    MATE_SCORE_WEIGHT,
    MATE_SEEK_BLEND,
    MATE_SEEK_DANGER_THRESHOLD,
    MATE_SEEK_HUNGER_THRESHOLD,
    MATE_SEEK_MIN_SPEED,
    MATE_SELECTION_RADIUS,
    MAX_REPRODUCTIONS,
    REPRODUCTION_COOLDOWN_TICKS,
    REPRODUCTION_HEALTH_THRESHOLD,
    REPRODUCTION_POP_CAP_FACTOR,
)
from ..genetics import combine_brains, combine_genes, compatible, mutate_brain
from ..species import ALL_SPECIES
from ..world.creature import Creature


def _ready_to_seek(c):
    return (c.health >= c.max_health * REPRODUCTION_HEALTH_THRESHOLD
            and c.hunger < HUNGER_MAX * MATE_SEEK_HUNGER_THRESHOLD)


def mate_seeking_drive(c, outputs, kin_grid):
    """outputs with steering blended toward the nearest compatible, ready adult of c's species
    (within MATE_ATTRACTION_RADIUS) while c is a fed, healthy adult out of danger."""
    if not c.is_adult or c.hunger >= HUNGER_MAX * MATE_SEEK_HUNGER_THRESHOLD:
        return outputs
    if not c.species.is_predator and c.nearest_smell_strength >= MATE_SEEK_DANGER_THRESHOLD:
        return outputs
    if c.health < c.max_health * REPRODUCTION_HEALTH_THRESHOLD:
        return outputs
    eligible = [
        m for m in kin_grid.query(c.pos, MATE_ATTRACTION_RADIUS)
        if m.id != c.id and m.is_adult and _ready_to_seek(m)
        and compatible(c.genes, m.genes, c.species)
    ]
    if not eligible:
        return outputs
    nearest = min(eligible, key=lambda m: c.pos.distance_to(m.pos))
    if c.pos.distance_to(nearest.pos) < 1:
        return outputs
    vec = nearest.pos - c.pos
    angle_diff = (math.atan2(vec.y, vec.x) - c.angle + math.pi) % (2 * math.pi) - math.pi
    turn = outputs[OUT.rotation]
    outputs[OUT.rotation] = turn * (1 - MATE_SEEK_BLEND) + angle_diff / math.pi * MATE_SEEK_BLEND
    outputs[OUT.speed] = max(outputs[OUT.speed], MATE_SEEK_MIN_SPEED)
    return outputs


def stochastic_round(x):
    """floor(x), plus one with probability frac(x): 1.3 gives 2 young 30% of the time."""
    n = int(x)
    return n + (random.random() < x - n)


def mate_preference(mates, species):
    top_score = max((m.score for m in mates), default=0) or 1.0
    return lambda m: (m.attractiveness / species.default_genes["attractiveness"]
                      + MATE_SCORE_WEIGHT * m.score / top_score)


def _can_breed(p, species, tick):
    return (p.is_adult
            and p.health > species.max_health * REPRODUCTION_HEALTH_THRESHOLD
            and p.reproductions < MAX_REPRODUCTIONS
            and (p.last_reproduction_tick is None
                 or tick - p.last_reproduction_tick > REPRODUCTION_COOLDOWN_TICKS))


def breed(world):
    """Pair up nearby ready adults of each species and return their young (not yet added).

    The population cap counts every young born this tick, of either species."""
    offspring = []
    paired = set()
    now = time.time()
    for species in ALL_SPECIES:
        group = world.population(species)
        pop_cap = int(species.start_count * REPRODUCTION_POP_CAP_FACTOR)
        # Readiness only changes for creatures that pair this tick, which `paired` excludes.
        ready = [p for p in group if _can_breed(p, species, world.tick)]
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
                     if ready[j].id not in paired and compatible(p1.genes, ready[j].genes, species)]
            if not mates:
                continue
            p2 = max(mates, key=mate_preference(mates, species))
            litter = min(stochastic_round((p1.genes["litter_size"] + p2.genes["litter_size"]) / 2),
                         pop_cap - len(group) - len(offspring))
            for _ in range(litter):
                offspring.append(_child(p1, p2, litter, now))
            for parent in (p1, p2):
                parent.health *= 0.7
                parent.reproductions += 1
                parent.last_reproduction_tick = world.tick
            paired.update((p1.id, p2.id))
    return offspring


def _child(p1, p2, litter, birth_time):
    genes = combine_genes(p1.genes, p2.genes, p1.species)
    brain = combine_brains(p1.brain, p2.brain)
    mutate_brain(brain, genes)
    child = Creature(p1.species, p1.world_bounds, brain=brain, genes=genes, geometry=p1.geometry)
    child.pos = (p1.pos + p2.pos) / 2
    if litter > 1:
        a = random.uniform(0, 2 * math.pi)
        child.pos += child.geometry.vec2(math.cos(a), math.sin(a)) * LITTER_SPREAD
    child.health = child.max_health * min(1.0, LITTER_ENDOWMENT / litter)
    child.lineage = p1.lineage
    child.parent_ids = (p1.id, p2.id)
    child.descent = max(p1.descent, p2.descent) + 1
    child.generation = max(p1.generation, p2.generation) + 1
    child.birth_time = birth_time
    return child
