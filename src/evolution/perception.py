"""Vectorized perception: the brain inputs for every creature in one numpy pass.

Row layout matches config.BRAIN_TOPOLOGY[0] (see CLAUDE.md, "Neural network").
Side effects, read by act(), the mate-seeking drive and the renderer:
c.whiskers, c.target_whiskers, c._nearest_target_dist, c._nearest_smell_strength, c.effective_sight.
"""
import math

import numpy as np

from .config import (
    BRAIN_INPUTS,
    GENE_MIN_MAX,
    HUNGER_MAX,
    KIN_DENSITY_NORM,
    KIN_SENSE_RADIUS,
    MAX_NORMALIZED_SIGHT_DISTANCE,
    NIGHT_SIGHT_FACTOR,
    NUM_MEMORY,
    NUM_WHISKERS,
    SIGNAL_RADIUS,
    SMELL_DISTANCE,
    WHISKER_TARGET_WIDTH,
)

N_INPUTS = BRAIN_INPUTS
_MAX_SIZE = GENE_MIN_MAX["size"][1]


def _xy(items):
    if not items:
        return np.zeros((0, 2))
    return np.array([(o.pos.x, o.pos.y) for o in items], dtype=np.float64)


def _ids(items):
    return np.array([o.id for o in items], dtype=np.int64)


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class _Relative:
    """Pairwise observer→target geometry. Distances are computed for every pair; the
    heading-relative angle (wrapped to [-pi, pi)) only for the targets that get picked."""
    __slots__ = ("dist", "dx", "dy", "angle", "not_self")

    def __init__(self, xy, angle, ids, targets):
        txy = _xy(targets)
        self.dx = txy[None, :, 0] - xy[:, None, 0]
        self.dy = txy[None, :, 1] - xy[:, None, 1]
        self.dist = np.hypot(self.dx, self.dy)
        self.angle = angle
        self.not_self = ids[:, None] != _ids(targets)[None, :]

    def angle_to(self, idx):
        rows = np.arange(len(idx))
        return _wrap(np.arctan2(self.dy[rows, idx], self.dx[rows, idx]) - self.angle)

    def in_cone(self, half_angle):
        """|relative angle| <= half_angle, via the cosine so no arctan2 per pair."""
        with np.errstate(divide="ignore", invalid="ignore"):
            cos_rel = (self.dx * np.cos(self.angle)[:, None]
                       + self.dy * np.sin(self.angle)[:, None]) / self.dist
        return (half_angle[:, None] >= math.pi) | (cos_rel >= np.cos(half_angle)[:, None])


def _nearest(dist, mask):
    """Index of the nearest target where mask holds, and whether one exists, per observer."""
    if dist.shape[1] == 0:
        return np.zeros(dist.shape[0], dtype=np.int64), np.zeros(dist.shape[0], dtype=bool)
    masked = np.where(mask, dist, np.inf)
    idx = masked.argmin(axis=1)
    return idx, np.isfinite(masked[np.arange(len(idx)), idx])


def _vision(rel, sight, half_angle):
    """Nearest target inside the sight cone: (index, found, (angle, distance)) with angle
    and distance normalized by the cone, (0, 1) when there is none."""
    n = rel.dist.shape[0]
    if rel.dist.shape[1] == 0:
        return np.zeros(n, dtype=np.int64), np.zeros(n, dtype=bool), (np.zeros(n), np.ones(n))
    visible = (rel.not_self & (rel.dist > 0) & (rel.dist <= sight[:, None])
               & rel.in_cone(half_angle))
    idx, found = _nearest(rel.dist, visible)
    angle = np.where(found, rel.angle_to(idx) / half_angle, 0.0)
    d = np.where(found, rel.dist[np.arange(n), idx] / sight, 1.0)
    return idx, found, (angle, d)


def _smell(rel):
    n = rel.dist.shape[0]
    if rel.dist.shape[1] == 0:
        return np.zeros(n), np.zeros(n)
    idx, found = _nearest(rel.dist, rel.not_self)
    d = rel.dist[np.arange(n), idx]
    found &= d < SMELL_DISTANCE
    return (np.where(found, rel.angle_to(idx) / math.pi, 0.0),
            np.where(found, 1.0 - d / SMELL_DISTANCE, 0.0))


def _social(rel, signals):
    """Kin (angle, distance, density) and the strongest kin signal (angle, strength)."""
    dist, not_self = rel.dist, rel.not_self
    n = dist.shape[0]
    if dist.shape[1] == 0:
        return np.zeros(n), np.ones(n), np.zeros(n), np.zeros(n), np.zeros(n)
    rows = np.arange(n)
    near = not_self & (dist < KIN_SENSE_RADIUS)
    idx, found = _nearest(dist, near)
    kin_angle = np.where(found, rel.angle_to(idx) / math.pi, 0.0)
    kin_dist = np.where(found, dist[rows, idx] / KIN_SENSE_RADIUS, 1.0)
    density = np.minimum(1.0, near.sum(axis=1) / KIN_DENSITY_NORM)
    heard = np.where(not_self & (dist < SIGNAL_RADIUS),
                     signals[None, :] * (1.0 - dist / SIGNAL_RADIUS), 0.0)
    loudest = heard.argmax(axis=1)
    strength = heard[rows, loudest]
    sig_angle = np.where(strength > 0, rel.angle_to(loudest) / math.pi, 0.0)
    return kin_angle, kin_dist, density, sig_angle, strength


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


def _target_whiskers(xy, angle, length, whisker_angles, targets_xy, occlusion):
    """Distance along each whisker ray to the nearest target within WHISKER_TARGET_WIDTH of
    it, no farther than `length` and closer than the ray's obstacle hit; inf when none."""
    n, w = whisker_angles.shape
    if len(targets_xy) == 0:
        return np.full((n, w), np.inf)
    a = (angle[:, None] + whisker_angles)[..., None]
    dx = targets_xy[None, None, :, 0] - xy[:, None, None, 0]
    dy = targets_xy[None, None, :, 1] - xy[:, None, None, 1]
    t = dx * np.cos(a) + dy * np.sin(a)
    perp = np.abs(dy * np.cos(a) - dx * np.sin(a))
    hit = ((t > 0) & (t <= length[:, None, None]) & (t < occlusion[..., None])
           & (perp <= WHISKER_TARGET_WIDTH))
    return np.where(hit, t, np.inf).min(axis=-1)


def perceive_all(creatures, food_items, herbivores, carnivores, obstacles, carcasses=(),
                 light=1.0, season=0.0):
    """Brain inputs for `creatures` (any mix of species) as an (N, N_INPUTS) array.

    Carnivores look at and smell `herbivores` and look at `carcasses`; herbivores look at
    `food_items`, look at and smell `carnivores`. Both sense their own species (kin and
    signals). `light` (0 = midnight, 1 = noon) shrinks the vision cones, not whiskers or smell.
    `season` (-1 = midwinter, 1 = midsummer) is passed through as the last input.
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
    eye = sight * (NIGHT_SIGHT_FACTOR + (1 - NIGHT_SIGHT_FACTOR) * light)
    xy = _xy(creatures)
    ids = _ids(creatures)
    whisker_angles = np.array([c.whisker_angles for c in creatures], dtype=np.float64)

    whisker_dist = _whiskers(xy, angle, sight, whisker_angles, obstacles)
    out[:, :NUM_WHISKERS] = 1.0 - whisker_dist / sight[:, None]
    col = NUM_WHISKERS

    target_whisker_dist = np.full((n, NUM_WHISKERS), np.inf)
    target_dist = np.ones(n)
    target_found = np.zeros(n, dtype=bool)
    smell_strength_self = np.zeros(n)
    carn = s[:, 11].astype(bool)
    social = col + 15
    out[:, social + 1] = 1.0  # kin distance when there is none
    out[:, social + 6] = 1.0  # carcass distance when there is none (and for herbivores)

    ci = np.flatnonzero(carn)
    if len(ci):
        to_prey = _Relative(xy[ci], angle[ci], ids[ci], herbivores)
        idx, found, (t_angle, t_dist) = _vision(to_prey, eye[ci], half_angle[ci])
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
        to_preds = _Relative(*observer, carnivores)
        _, food_found, (food_angle, food_dist) = _vision(_Relative(*observer, food_items),
                                                         eye[hi], half_angle[hi])
        _, _, (pred_angle, pred_dist) = _vision(to_preds, eye[hi], half_angle[hi])
        out[hi, col] = np.clip(food_angle - pred_angle * (1 - pred_dist) ** 2, -1, 1)
        out[hi, col + 1] = food_dist
        target_dist[hi], target_found[hi] = food_dist, food_found
        sm_angle, sm_strength = _smell(to_preds)
        out[hi, col + 2], out[hi, col + 3] = sm_angle, sm_strength
        smell_strength_self[hi] = sm_strength  # read by the mate-seeking danger check

    for gi, kin in ((ci, carnivores), (hi, herbivores)):
        if len(gi):
            signals = np.array([k.signal for k in kin], dtype=np.float64)
            social_cols = _social(_Relative(xy[gi], angle[gi], ids[gi], kin), signals)
            for j, values in enumerate(social_cols):
                out[gi, social + j] = values
    if len(ci) and carcasses:
        _, _, (cc_angle, cc_dist) = _vision(_Relative(xy[ci], angle[ci], ids[ci], carcasses),
                                            eye[ci], half_angle[ci])
        out[ci, social + 5], out[ci, social + 6] = cc_angle, cc_dist
    out[:, social + 7] = light
    out[:, social + 8:social + 8 + NUM_MEMORY] = [c.memory for c in creatures]
    for gi, targets in ((ci, list(herbivores) + list(carcasses)), (hi, food_items)):
        if len(gi):
            target_whisker_dist[gi] = _target_whiskers(xy[gi], angle[gi], eye[gi],
                                                       whisker_angles[gi], _xy(targets),
                                                       whisker_dist[gi])
    tw = social + 8 + NUM_MEMORY
    out[:, tw:tw + NUM_WHISKERS] = np.where(np.isfinite(target_whisker_dist),
                                            1.0 - target_whisker_dist / eye[:, None], 0.0)
    out[:, tw + NUM_WHISKERS] = season

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
    twl = [[d if math.isfinite(d) else None for d in row] for row in target_whisker_dist.tolist()]
    td = target_dist.tolist()
    tf = target_found.tolist()
    ss = smell_strength_self.tolist()
    es = eye.tolist()
    for i, c in enumerate(creatures):
        c.effective_sight = es[i]
        c.whiskers = wl[i]
        c.target_whiskers = twl[i]
        c._nearest_target_dist = td[i] if tf[i] else None
        c._nearest_smell_strength = ss[i]
    return out
