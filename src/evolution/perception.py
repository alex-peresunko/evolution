"""Vectorized perception: the brain inputs for every creature in one numpy pass.

Row layout matches config.BRAIN_TOPOLOGY[0] (see CLAUDE.md, "Neural network").
Side effects, read by act(), the mate-seeking drive and the renderer:
c.whiskers, c._nearest_target_dist, c._nearest_smell_strength.
"""
import math

import numpy as np

from .config import (
    GENE_MIN_MAX,
    HUNGER_MAX,
    MAX_NORMALIZED_SIGHT_DISTANCE,
    NUM_WHISKERS,
    SMELL_DISTANCE,
)

N_INPUTS = NUM_WHISKERS + 2 + 2 + 2 + 3 + 6
_MAX_SIZE = GENE_MIN_MAX["size"][1]


def _xy(items):
    if not items:
        return np.zeros((0, 2))
    return np.array([(o.pos.x, o.pos.y) for o in items], dtype=np.float64)


def _ids(items):
    return np.array([o.id for o in items], dtype=np.int64)


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def _relative(xy, angle, ids, targets):
    """Distances and heading-relative angles (wrapped to [-pi, pi)) from each observer to
    each target, plus a mask that excludes the observer itself."""
    txy = _xy(targets)
    dx = txy[None, :, 0] - xy[:, None, 0]
    dy = txy[None, :, 1] - xy[:, None, 1]
    dist = np.hypot(dx, dy)
    rel = _wrap(np.arctan2(dy, dx) - angle[:, None])
    not_self = ids[:, None] != _ids(targets)[None, :]
    return dist, rel, not_self


def _nearest(dist, mask):
    """Index of the nearest target where mask holds, and whether one exists, per observer."""
    if dist.shape[1] == 0:
        return np.zeros(dist.shape[0], dtype=np.int64), np.zeros(dist.shape[0], dtype=bool)
    masked = np.where(mask, dist, np.inf)
    idx = masked.argmin(axis=1)
    return idx, np.isfinite(masked[np.arange(len(idx)), idx])


def _target_info(dist, rel, idx, found, sight, half_angle):
    """Normalized (angle, distance) to the chosen target, (0, 1) when there is none."""
    rows = np.arange(len(idx))
    if dist.shape[1] == 0:
        return np.zeros(len(idx)), np.ones(len(idx))
    angle = np.where(found, rel[rows, idx] / half_angle, 0.0)
    d = np.where(found, dist[rows, idx] / sight, 1.0)
    return angle, d


def _vision(rel_info, sight, half_angle):
    dist, rel, not_self = rel_info
    visible = not_self & (dist > 0) & (dist <= sight[:, None]) & (np.abs(rel) <= half_angle[:, None])
    idx, found = _nearest(dist, visible)
    return idx, found, _target_info(dist, rel, idx, found, sight, half_angle)


def _smell(rel_info):
    dist, rel, not_self = rel_info
    idx, found = _nearest(dist, not_self)
    rows = np.arange(len(idx))
    if dist.shape[1]:
        d = dist[rows, idx]
        found &= d < SMELL_DISTANCE
        return np.where(found, rel[rows, idx] / math.pi, 0.0), np.where(found, 1.0 - d / SMELL_DISTANCE, 0.0)
    return np.zeros(len(idx)), np.zeros(len(idx))


def _whiskers(xy, angle, sight, whisker_angles, obstacles):
    """Distance along each whisker to the first obstacle (Liang-Barsky clip), capped at the
    sight distance. A whisker starting inside an obstacle is touching it (distance 0)."""
    dist = np.repeat(sight[:, None], whisker_angles.shape[1], axis=1)
    if not obstacles:
        return dist
    a = angle[:, None] + whisker_angles
    dx = (np.cos(a) * sight[:, None])[..., None, None]
    dy = (np.sin(a) * sight[:, None])[..., None, None]
    x = xy[:, 0][:, None, None, None]
    y = xy[:, 1][:, None, None, None]
    rects = np.array([(o.rect.left, o.rect.right, o.rect.top, o.rect.bottom) for o in obstacles],
                     dtype=np.float64)
    left, right, top, bottom = (rects[:, i][None, None, :, None] for i in range(4))
    p = np.concatenate(np.broadcast_arrays(-dx, dx, -dy, dy), axis=-1)
    q = np.concatenate(np.broadcast_arrays(x - left, right - x, y - top, bottom - y), axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = q / p
    t0 = np.maximum(0.0, np.where(p < 0, r, -np.inf).max(axis=-1))
    t1 = np.minimum(1.0, np.where(p > 0, r, np.inf).min(axis=-1))
    hit = ~((p == 0) & (q < 0)).any(axis=-1) & (t0 < t1)
    hit_dist = np.where(hit, t0 * sight[:, None, None], np.inf).min(axis=-1)
    return np.minimum(dist, hit_dist)


def perceive_all(creatures, food_items, herbivores, carnivores, obstacles):
    """Brain inputs for `creatures` (any mix of species) as an (N, N_INPUTS) array.

    Carnivores look at and smell `herbivores`; herbivores look at `food_items`, look at
    and smell `carnivores`.
    """
    n = len(creatures)
    out = np.zeros((n, N_INPUTS))
    if n == 0:
        return out

    s = np.array([
        (c.angle, c.sight_distance, c.sight_angle, c.health, c.max_health, c.size,
         c.current_speed, c.genetic_max_speed, c.stamina, c.max_stamina, c.hunger,
         c.is_carnivore)
        for c in creatures
    ], dtype=np.float64)
    angle, sight, sight_angle = s[:, 0], s[:, 1], s[:, 2]
    half_angle = sight_angle / 2
    xy = _xy(creatures)
    ids = _ids(creatures)
    whisker_angles = np.array([c.whisker_angles for c in creatures], dtype=np.float64)

    whisker_dist = _whiskers(xy, angle, sight, whisker_angles, obstacles)
    out[:, :NUM_WHISKERS] = 1.0 - whisker_dist / sight[:, None]
    col = NUM_WHISKERS

    target_dist = np.ones(n)
    target_found = np.zeros(n, dtype=bool)
    smell_strength_self = np.zeros(n)
    carn = s[:, 11].astype(bool)

    ci = np.flatnonzero(carn)
    if len(ci):
        to_prey = _relative(xy[ci], angle[ci], ids[ci], herbivores)
        idx, found, (t_angle, t_dist) = _vision(to_prey, sight[ci], half_angle[ci])
        out[ci, col], out[ci, col + 1] = t_angle, t_dist
        target_dist[ci], target_found[ci] = t_dist, found
        sm_angle, sm_strength = _smell(to_prey)
        out[ci, col + 2], out[ci, col + 3] = sm_angle, sm_strength
        smell_strength_self[ci] = sm_strength
        if herbivores:
            prey = np.array([(h.angle, h.current_speed / h.max_speed if h.max_speed > 0 else 0.0)
                             for h in herbivores], dtype=np.float64)[idx]
            out[ci, col + 4] = np.where(found, np.cos(prey[:, 0]) * prey[:, 1], 0.0)
            out[ci, col + 5] = np.where(found, np.sin(prey[:, 0]) * prey[:, 1], 0.0)

    hi = np.flatnonzero(~carn)
    if len(hi):
        observer = (xy[hi], angle[hi], ids[hi])
        to_preds = _relative(*observer, carnivores)
        _, food_found, (food_angle, food_dist) = _vision(_relative(*observer, food_items),
                                                         sight[hi], half_angle[hi])
        _, _, (pred_angle, pred_dist) = _vision(to_preds, sight[hi], half_angle[hi])
        out[hi, col] = np.clip(food_angle - pred_angle * (1 - pred_dist) ** 2, -1, 1)
        out[hi, col + 1] = food_dist
        target_dist[hi], target_found[hi] = food_dist, food_found
        sm_angle, sm_strength = _smell(to_preds)
        out[hi, col + 2], out[hi, col + 3] = sm_angle, sm_strength
        smell_strength_self[hi] = sm_strength  # read by the mate-seeking danger check

    col += 6
    out[:, col] = s[:, 3] / s[:, 4]
    out[:, col + 1] = np.sin(angle)
    out[:, col + 2] = np.cos(angle)
    out[:, col + 3] = s[:, 5] / _MAX_SIZE
    out[:, col + 4] = s[:, 6] / s[:, 7]
    out[:, col + 5] = sight / MAX_NORMALIZED_SIGHT_DISTANCE
    out[:, col + 6] = sight_angle / (2 * math.pi)
    out[:, col + 7] = s[:, 8] / s[:, 9]
    out[:, col + 8] = np.minimum(1.0, s[:, 10] / HUNGER_MAX)

    wl = whisker_dist.tolist()
    td = target_dist.tolist()
    tf = target_found.tolist()
    ss = smell_strength_self.tolist()
    for i, c in enumerate(creatures):
        c.whiskers = wl[i]
        c._nearest_target_dist = td[i] if tf[i] else None
        c._nearest_smell_strength = ss[i]
    return out
