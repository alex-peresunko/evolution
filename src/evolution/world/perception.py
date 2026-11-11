"""Vectorized perception: the brain inputs for every creature in one numpy pass.

Columns are named by brain.layout.INPUTS.
Side effects, read by act(), the mate-seeking drive and the renderer:
c.whiskers, c.target_whiskers, c.nearest_target_dist, c.nearest_smell_strength, c.effective_sight.
"""
import math

import numpy as np

from ..brain.layout import INPUTS as IN
from ..config import (
    BODY_RADIUS_FACTOR,
    GENE_MIN_MAX,
    HUNGER_MAX,
    KIN_DENSITY_NORM,
    KIN_SENSE_RADIUS,
    MATE_ATTRACTION_RADIUS,
    MAX_NORMALIZED_SIGHT_DISTANCE,
    NIGHT_SIGHT_FACTOR,
    NUM_WHISKERS,
    SIGNAL_RADIUS,
    SMELL_DISTANCE,
    THIRST_MAX,
    TOUCH_TOLERANCE,
    WATER_SENSE_RADIUS,
    WHISKER_TARGET_WIDTH,
)

_MAX_SIZE = GENE_MIN_MAX["size"][1]


def _xy(items):
    if not items:
        return np.zeros((0, 2))
    return np.array([(o.pos.x, o.pos.y) for o in items], dtype=np.float64)


def _ids(items):
    return np.array([o.id for o in items], dtype=np.int64)


def _positions(items):
    return _xy(items), _ids(items)


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class _Relative:
    """Pairwise observer→target geometry. Distances are computed for every pair; the
    heading-relative angle (wrapped to [-pi, pi)) only for the targets that get picked.
    `others` is `dist` with inf where the target is the observer itself."""
    __slots__ = ("dist", "dx", "dy", "angle", "others")

    def __init__(self, xy, angle, ids, targets):
        """targets: (xy, ids) arrays, as from _positions."""
        txy, tids = targets
        self.dx = txy[None, :, 0] - xy[:, None, 0]
        self.dy = txy[None, :, 1] - xy[:, None, 1]
        self.dist = np.hypot(self.dx, self.dy)
        self.angle = angle
        self.others = np.where(ids[:, None] != tids[None, :], self.dist, np.inf)

    def angle_to(self, idx):
        rows = np.arange(len(idx))
        return _wrap(np.arctan2(self.dy[rows, idx], self.dx[rows, idx]) - self.angle)

    def nearest_other(self):
        """Per observer: the index of the nearest target that is not itself and its distance,
        inf when there is none. `others` must have a column."""
        idx = self.others.argmin(axis=1)
        return idx, self.others[np.arange(len(idx)), idx]

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


def _nonzero(mask):
    """np.nonzero of a 2-D mask, which is several times slower on one."""
    return np.divmod(np.flatnonzero(mask), mask.shape[1])


def _near_pairs(xy, radius, txy):
    """(rows, cols) of observer-target pairs: every pair no farther apart than the observer's
    `radius`, and some farther ones. Rows ascend; the columns of a row are in no order.
    `txy` must not be empty."""
    n = len(xy)
    cell = max(float(radius.mean()), 16.0)
    tcx = np.floor(txy[:, 0] / cell).astype(np.int64)
    tcy = np.floor(txy[:, 1] / cell).astype(np.int64)
    left, top = tcx.min(), tcy.min()
    tcx -= left
    tcy -= top
    width, height = int(tcx.max()) + 1, int(tcy.max()) + 1
    key = tcy * width + tcx
    order = np.argsort(key)
    # first[k]: where cell k starts among the targets sorted by cell.
    first = np.concatenate(([0], np.cumsum(np.bincount(key, minlength=width * height))))
    reach = radius + 1.0  # absorbs rounding
    x0 = np.maximum(np.floor((xy[:, 0] - reach) / cell).astype(np.int64) - left, 0)
    x1 = np.minimum(np.floor((xy[:, 0] + reach) / cell).astype(np.int64) - left, width - 1)
    y0 = np.floor((xy[:, 1] - reach) / cell).astype(np.int64) - top
    y1 = np.floor((xy[:, 1] + reach) / cell).astype(np.int64) - top
    span = int((y1 - y0).max()) + 1
    ys = y0[:, None] + np.arange(span)[None, :]
    valid = (ys <= y1[:, None]) & (ys >= 0) & (ys < height) & (x0 <= x1)[:, None]
    line = np.clip(ys, 0, height - 1) * width
    lo = first[line + np.minimum(x0, width - 1)[:, None]]
    hi = first[line + np.maximum(x1, 0)[:, None] + 1]
    counts = np.where(valid, hi - lo, 0).ravel()
    ends = np.cumsum(counts)
    at = np.arange(ends[-1]) - np.repeat(ends - counts - lo.ravel(), counts)
    return np.repeat(np.repeat(np.arange(n), span), counts), order[at]


def _group_starts(rows):
    """Index of the first element of each run of equal values in `rows` (not empty)."""
    return np.flatnonzero(np.concatenate(([True], rows[1:] != rows[:-1])))


def _crowns(obstacles):
    """(T, 3) x, y, radius of the tree crowns among `obstacles`."""
    return np.array([(o.x, o.y, o.crown_radius) for o in obstacles if hasattr(o, "crown_radius")],
                    dtype=np.float64).reshape(-1, 3)


def _crown_hides(ox, oy, sx, sy, crowns):
    """Per pair (observer at ox, oy, target at ox + sx, oy + sy): whether a tree crown hides the
    target. A crown hides what is under or behind it, except from an observer under that same
    crown."""
    hidden = np.zeros(len(ox), dtype=bool)
    cx, cy, r = crowns[:, 0], crowns[:, 1], crowns[:, 2]
    # A crown can only hide the target of an observer within the sight line's length plus the
    # crown radius of its centre; the +1 absorbs rounding.
    reach = np.sqrt(sx * sx + sy * sy)[:, None] + r + 1.0
    close = np.flatnonzero((((cx - ox[:, None]) ** 2 + (cy - oy[:, None]) ** 2)
                            < reach * reach).any(axis=1))
    if len(close) == 0:
        return hidden
    r2 = r ** 2
    ox, oy, sx, sy = ox[close, None], oy[close, None], sx[close, None], sy[close, None]
    tx, ty = ox + sx, oy + sy
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.nan_to_num(np.clip(((cx - ox) * sx + (cy - oy) * sy) / (sx * sx + sy * sy), 0, 1))
    crosses = (ox + t * sx - cx) ** 2 + (oy + t * sy - cy) ** 2 < r2
    shared = ((ox - cx) ** 2 + (oy - cy) ** 2 < r2) & ((tx - cx) ** 2 + (ty - cy) ** 2 < r2)
    hidden[close] = (crosses & ~shared).any(axis=1)
    return hidden


def _crown_occlusion(xy, angle, whisker_angles, crowns):
    """(N, W) distance along each whisker ray to where crowns hide what lies beyond: the
    entry of a crown ahead, or the exit of a crown the observer stands under; inf when none."""
    n, w = whisker_angles.shape
    if len(crowns) == 0:
        return np.full((n, w), np.inf)
    a = (angle[:, None] + whisker_angles)[..., None]
    dx = crowns[None, None, :, 0] - xy[:, None, None, 0]
    dy = crowns[None, None, :, 1] - xy[:, None, None, 1]
    r2 = crowns[None, None, :, 2] ** 2
    along = dx * np.cos(a) + dy * np.sin(a)
    d2 = dx * dx + dy * dy
    half = np.sqrt(np.maximum(r2 - (d2 - along * along), 0.0))
    hit = d2 - along * along < r2
    occ = np.where(d2 < r2, along + half,
                   np.where(hit & (along - half > 0), along - half, np.inf))
    return occ.min(axis=-1)


def _vision(rel, sight, half_angle, xy=None, crowns=()):
    """Nearest target inside the sight cone and not hidden by a crown (observers at `xy`):
    (index, found, (angle, distance)) with angle and distance normalized by the cone, (0, 1)
    when there is none."""
    n = rel.dist.shape[0]
    if rel.dist.shape[1] == 0:
        return np.zeros(n, dtype=np.int64), np.zeros(n, dtype=bool), (np.zeros(n), np.ones(n))
    visible = (rel.others > 0) & (rel.others <= sight[:, None]) & rel.in_cone(half_angle)
    if len(crowns):
        rows, cols = _nonzero(visible)
        if len(rows):
            hidden = _crown_hides(xy[rows, 0], xy[rows, 1], rel.dx[rows, cols],
                                  rel.dy[rows, cols], crowns)
            visible[rows[hidden], cols[hidden]] = False
    idx, found = _nearest(rel.dist, visible)
    angle = np.where(found, rel.angle_to(idx) / half_angle, 0.0)
    d = np.where(found, rel.dist[np.arange(n), idx] / sight, 1.0)
    return idx, found, (angle, d)


def _smell(rel):
    n = rel.dist.shape[0]
    if rel.dist.shape[1] == 0:
        return np.zeros(n), np.zeros(n)
    idx, d = rel.nearest_other()
    found = d < SMELL_DISTANCE
    return (np.where(found, rel.angle_to(idx) / math.pi, 0.0),
            np.where(found, 1.0 - d / SMELL_DISTANCE, 0.0))


def _social(rel, signals):
    """Kin (angle, distance, density) and the strongest kin signal (angle, strength)."""
    dist, others = rel.dist, rel.others
    n = dist.shape[0]
    if dist.shape[1] == 0:
        return np.zeros(n), np.ones(n), np.zeros(n), np.zeros(n), np.zeros(n)
    rows = np.arange(n)
    idx, nearest = rel.nearest_other()
    found = nearest < KIN_SENSE_RADIUS
    kin_angle = np.where(found, rel.angle_to(idx) / math.pi, 0.0)
    kin_dist = np.where(found, nearest / KIN_SENSE_RADIUS, 1.0)
    density = np.minimum(1.0, (others < KIN_SENSE_RADIUS).sum(axis=1) / KIN_DENSITY_NORM)
    heard = np.where(others < SIGNAL_RADIUS, signals[None, :] * (1.0 - dist / SIGNAL_RADIUS), 0.0)
    loudest = heard.argmax(axis=1)
    strength = heard[rows, loudest]
    sig_angle = np.where(strength > 0, rel.angle_to(loudest) / math.pi, 0.0)
    return kin_angle, kin_dist, density, sig_angle, strength


def _mate(rel, fertile):
    """Angle and distance of the nearest fertile kin within MATE_ATTRACTION_RADIUS; (0, 1) when
    there is none."""
    n = rel.dist.shape[0]
    if rel.dist.shape[1] == 0:
        return np.zeros(n), np.ones(n)
    # inf for the kin that are not fertile, so the nearest of the rest is the minimum.
    fertile_dist = rel.others + np.where(fertile, 0.0, np.inf)[None, :]
    idx = fertile_dist.argmin(axis=1)
    nearest = fertile_dist[np.arange(n), idx]
    found = nearest < MATE_ATTRACTION_RADIUS
    return (np.where(found, rel.angle_to(idx) / math.pi, 0.0),
            np.where(found, nearest / MATE_ATTRACTION_RADIUS, 1.0))


def _touch(rel, size, targets):
    """Strength and angle of the body pressing hardest. With gap = distance minus the body radii,
    strength is 0.5 at contact and grows linearly to 1 at TOUCH_TOLERANCE deep, from 0 at
    TOUCH_TOLERANCE apart."""
    n = rel.dist.shape[0]
    if rel.dist.shape[1] == 0:
        return np.zeros(n), np.zeros(n)
    target_size = np.array([t.size for t in targets], dtype=np.float64)
    strength, angle = np.zeros(n), np.zeros(n)
    # Bodies farther apart than this do not press; the +1 absorbs rounding.
    reach = (size + target_size.max()) * BODY_RADIUS_FACTOR + TOUCH_TOLERANCE + 1.0
    rows, cols = _nonzero(rel.others < reach[:, None])
    if len(rows) == 0:
        return strength, angle
    gap = rel.dist[rows, cols] - (size[rows] + target_size[cols]) * BODY_RADIUS_FACTOR
    press = np.clip((TOUCH_TOLERANCE - gap) / (2 * TOUCH_TOLERANCE), 0, 1)
    starts = _group_starts(rows)
    hardest = np.repeat(np.maximum.reduceat(press, starts), np.diff(np.append(starts, len(rows))))
    # _nonzero lists a row's columns in order, so the first of equal presses wins.
    tied = np.flatnonzero((press == hardest) & (press > 0))
    if len(tied) == 0:
        return strength, angle
    pick = tied[_group_starts(rows[tied])]
    who, idx = rows[pick], cols[pick]
    strength[who] = press[pick]
    angle[who] = _wrap(np.arctan2(rel.dy[who, idx], rel.dx[who, idx]) - rel.angle[who]) / math.pi
    return strength, angle


def _edge_distance(xy, a, world_bounds):
    """Distance along rays at angles a (N, W) from xy to the world edge."""
    w, h = world_bounds
    cos, sin = np.cos(a), np.sin(a)
    x, y = xy[:, 0][:, None], xy[:, 1][:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        tx = np.where(cos > 0, (w - x) / cos, np.where(cos < 0, -x / cos, np.inf))
        ty = np.where(sin > 0, (h - y) / sin, np.where(sin < 0, -y / sin, np.inf))
    return np.maximum(0.0, np.minimum(tx, ty))


def _whiskers(xy, angle, sight, whisker_angles, obstacles, world_bounds):
    """Distance along each whisker to the first obstacle (Liang-Barsky clip) or world edge,
    capped at the sight distance. A whisker starting inside an obstacle is touching it
    (distance 0)."""
    a = angle[:, None] + whisker_angles
    dist = np.minimum(sight[:, None], _edge_distance(xy, a, world_bounds))
    if not obstacles:
        return dist
    rects = np.array([(o.rect.left, o.rect.right, o.rect.top, o.rect.bottom) for o in obstacles],
                     dtype=np.float64)
    x, y = xy[:, 0][:, None], xy[:, 1][:, None]
    gap_x = np.maximum(np.maximum(rects[:, 0] - x, x - rects[:, 1]), 0.0)
    gap_y = np.maximum(np.maximum(rects[:, 2] - y, y - rects[:, 3]), 0.0)
    # A whisker cannot reach a rect farther away than the sight; the +1 absorbs rounding.
    rows, cols = _nonzero(gap_x * gap_x + gap_y * gap_y <= ((sight + 1.0) ** 2)[:, None])
    if len(rows) == 0:
        return dist
    reach = sight[rows, None]
    dx = (np.cos(a) * sight[:, None])[rows, :, None]
    dy = (np.sin(a) * sight[:, None])[rows, :, None]
    x, y = xy[rows, 0][:, None, None], xy[rows, 1][:, None, None]
    left, right, top, bottom = (rects[cols, i][:, None, None] for i in range(4))
    p = np.concatenate(np.broadcast_arrays(-dx, dx, -dy, dy), axis=-1)
    q = np.concatenate(np.broadcast_arrays(x - left, right - x, y - top, bottom - y), axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = q / p
    t0 = np.maximum(0.0, np.where(p < 0, r, -np.inf).max(axis=-1))
    t1 = np.minimum(1.0, np.where(p > 0, r, np.inf).min(axis=-1))
    hit = ~((p == 0) & (q < 0)).any(axis=-1) & (t0 < t1)
    starts = _group_starts(rows)
    hit_dist = np.full(dist.shape, np.inf)
    hit_dist[rows[starts]] = np.minimum.reduceat(np.where(hit, t0 * reach, np.inf), starts, axis=0)
    return np.minimum(dist, hit_dist)


def _shore_gaps(xy, lakes):
    """(L, N) Lake.shore_gap of every position for every lake blob, where it is below
    WATER_SENSE_RADIUS; inf elsewhere."""
    harmonics = max(len(lake.harmonics) for lake in lakes)
    # Blobs with fewer harmonics are padded with zero amplitudes.
    waves = np.zeros((len(lakes), harmonics, 3))
    for i, lake in enumerate(lakes):
        if lake.harmonics:
            waves[i, :len(lake.harmonics)] = lake.harmonics
    shape = np.array([(lake.x, lake.y, lake.radius, lake.max_radius) for lake in lakes],
                     dtype=np.float64)
    dx = xy[None, :, 0] - shape[:, None, 0]
    dy = xy[None, :, 1] - shape[:, None, 1]
    dist = np.hypot(dx, dy)
    gaps = np.full(dist.shape, np.inf)
    # The shore is at most max_radius from the centre; the +1 absorbs rounding.
    blob, who = _nonzero(dist < (shape[:, 3] + WATER_SENSE_RADIUS + 1.0)[:, None])
    if len(blob):
        theta = np.arctan2(dy[blob, who], dx[blob, who])
        ripple = 0
        for h in range(harmonics):
            k, a, phase = waves[blob, h, 0], waves[blob, h, 1], waves[blob, h, 2]
            ripple = ripple + a * np.cos(k * theta + phase)
        gaps[blob, who] = dist[blob, who] - shape[blob, 2] * (1 + ripple)
    return gaps


def _target_whiskers(angle, length, whisker_angles, occlusion, rows, dx, dy):
    """Distance along each whisker ray to the nearest target within WHISKER_TARGET_WIDTH of
    it, no farther than `length` and closer than `occlusion`; inf when none.
    The candidate pairs are observer `rows` (ascending) with the target at offset (dx, dy);
    they must include every pair within length + WHISKER_TARGET_WIDTH of each other."""
    n, w = whisker_angles.shape
    result = np.full((n, w), np.inf)
    if len(rows) == 0:
        return result
    a = (angle[:, None] + whisker_angles)[..., None]
    cos, sin = np.cos(a)[rows, :, 0], np.sin(a)[rows, :, 0]
    dx, dy = dx[:, None], dy[:, None]
    t = dx * cos + dy * sin
    perp = np.abs(dy * cos - dx * sin)
    hit = ((t > 0) & (t <= length[rows, None]) & (t < occlusion[rows])
           & (perp <= WHISKER_TARGET_WIDTH))
    starts = _group_starts(rows)
    result[rows[starts]] = np.minimum.reduceat(np.where(hit, t, np.inf), starts, axis=0)
    return result


def _dense_target_whiskers(xy, angle, length, whisker_angles, targets_xy, occlusion, dist):
    """_target_whiskers from the (N, M) observer-target distance matrix `dist`."""
    if len(targets_xy) == 0:
        return np.full(whisker_angles.shape, np.inf)
    # A hit lies within sqrt(length² + width²) <= length + width; the +1 absorbs rounding.
    rows, cols = _nonzero(dist <= (length + WHISKER_TARGET_WIDTH + 1.0)[:, None])
    return _target_whiskers(angle, length, whisker_angles, occlusion, rows,
                            targets_xy[cols, 0] - xy[rows, 0], targets_xy[cols, 1] - xy[rows, 1])


def _food_sight(xy, angle, eye, half_angle, whisker_angles, occlusion, food_xy, crowns):
    """What _vision and _target_whiskers give for food, computed only for the pairs in range:
    (found, (angle, distance), whisker distances)."""
    n = len(xy)
    found, v_angle, v_dist = np.zeros(n, dtype=bool), np.zeros(n), np.ones(n)
    if len(food_xy) == 0:
        return found, (v_angle, v_dist), np.full(whisker_angles.shape, np.inf)
    rows, cols = _near_pairs(xy, eye + WHISKER_TARGET_WIDTH + 1.0, food_xy)
    dx = food_xy[cols, 0] - xy[rows, 0]
    dy = food_xy[cols, 1] - xy[rows, 1]
    dist = np.hypot(dx, dy)
    near = np.flatnonzero(dist <= (eye + WHISKER_TARGET_WIDTH + 1.0)[rows])
    rows, cols, dx, dy, dist = rows[near], cols[near], dx[near], dy[near], dist[near]
    whisker_dist = _target_whiskers(angle, eye, whisker_angles, occlusion, rows, dx, dy)

    seen = np.flatnonzero((dist > 0) & (dist <= eye[rows]))
    rows, cols, dx, dy, dist = rows[seen], cols[seen], dx[seen], dy[seen], dist[seen]
    cos_rel = (dx * np.cos(angle)[rows] + dy * np.sin(angle)[rows]) / dist
    seen = (half_angle[rows] >= math.pi) | (cos_rel >= np.cos(half_angle)[rows])
    if len(crowns) and seen.any():
        seen[seen] = ~_crown_hides(xy[rows[seen], 0], xy[rows[seen], 1], dx[seen], dy[seen],
                                   crowns)
    rows, cols, dx, dy, dist = rows[seen], cols[seen], dx[seen], dy[seen], dist[seen]
    if len(rows) == 0:
        return found, (v_angle, v_dist), whisker_dist
    starts = _group_starts(rows)
    nearest = np.full(n, np.inf)
    nearest[rows[starts]] = np.minimum.reduceat(dist, starts)
    # Of equally near items the first in `food_xy` wins, as argmin picks it.
    tied = np.flatnonzero(dist == nearest[rows])
    tied = tied[np.lexsort((cols[tied], rows[tied]))]
    pick = tied[_group_starts(rows[tied])]
    who = rows[pick]
    found[who] = True
    v_angle[who] = _wrap(np.arctan2(dy[pick], dx[pick]) - angle[who]) / half_angle[who]
    v_dist[who] = dist[pick] / eye[who]
    return found, (v_angle, v_dist), whisker_dist


def perceive_all(creatures, food_items, herbivores, carnivores, obstacles, carcasses=(),
                 light=1.0, season=0.0, lakes=(), tick=0, food_xy=None):
    """Brain inputs for `creatures` (any mix of species) as an (N, INPUTS.width) array.

    Carnivores look at and smell `herbivores` and `carcasses`; herbivores look at
    `food_items`, look at and smell `carnivores`. Both sense their own species (kin and
    signals). `light` (0 = midnight, 1 = noon) shrinks the vision cones, not whiskers or smell.
    Tree crowns among `obstacles` block sight (cones and target whiskers), not obstacle
    whiskers, smell or the kin/signal senses.
    `season` (-1 = midwinter, 1 = midsummer) is passed through as an input.
    `tick` decides who is fertile (Creature.fertile).
    Water is the lake blob in `lakes` whose shore is nearest.
    `food_xy`: the (F, 2) positions of `food_items`, when the caller already has them.
    """
    n = len(creatures)
    out = np.zeros((n, IN.width))
    if n == 0:
        return out

    s = np.array([
        (c.angle, c.sight_distance, c.sight_angle, c.health, c.max_health, c.size,
         c.current_speed, c.genetic_max_speed, c.stamina, c.max_stamina, c.hunger,
         c.species.is_predator)
        for c in creatures
    ], dtype=np.float64)
    angle, sight, sight_angle = s[:, 0], s[:, 1], s[:, 2]
    half_angle = sight_angle / 2
    eye = sight * (NIGHT_SIGHT_FACTOR + (1 - NIGHT_SIGHT_FACTOR) * light)
    xy = _xy(creatures)
    ids = _ids(creatures)
    whisker_angles = np.array([c.whisker_angles for c in creatures], dtype=np.float64)

    whisker_dist = _whiskers(xy, angle, sight, whisker_angles, obstacles,
                             creatures[0].world_bounds)
    crowns = _crowns(obstacles)
    out[:, IN.obstacle_whiskers] = 1.0 - whisker_dist / sight[:, None]

    target_whisker_dist = np.full((n, NUM_WHISKERS), np.inf)
    target_dist = np.ones(n)
    target_found = np.zeros(n, dtype=bool)
    smell_strength_self = np.zeros(n)
    carn = s[:, 11].astype(bool)
    out[:, IN.kin_dist] = 1.0      # when there is none
    out[:, IN.carcass_dist] = 1.0  # when there is none (and for herbivores)
    out[:, IN.mate_dist] = 1.0     # when there is none

    at_herbivores, at_carnivores = _positions(herbivores), _positions(carnivores)
    at_carcasses = _positions(carcasses)
    if food_xy is None:
        food_xy = _xy(food_items)
    to_prey = to_preds = None
    ci = np.flatnonzero(carn)
    if len(ci):
        to_prey = _Relative(xy[ci], angle[ci], ids[ci], at_herbivores)
        idx, found, (t_angle, t_dist) = _vision(to_prey, eye[ci], half_angle[ci], xy[ci], crowns)
        out[ci, IN.target_angle], out[ci, IN.target_dist] = t_angle, t_dist
        target_dist[ci], target_found[ci] = t_dist, found
        sm_angle, sm_strength = _smell(to_prey)
        out[ci, IN.smell_angle], out[ci, IN.smell_strength] = sm_angle, sm_strength
        smell_strength_self[ci] = sm_strength
        if herbivores:
            prey = np.array([(h.angle, h.current_speed / h.max_speed if h.max_speed > 0 else 0.0)
                             for h in herbivores], dtype=np.float64)[idx]
            out[ci, IN.prey_vx] = np.where(found, np.cos(prey[:, 0]) * prey[:, 1], 0.0)
            out[ci, IN.prey_vy] = np.where(found, np.sin(prey[:, 0]) * prey[:, 1], 0.0)

    hi = np.flatnonzero(~carn)
    if len(hi):
        to_preds = _Relative(xy[hi], angle[hi], ids[hi], at_carnivores)
        _, _, (pred_angle, pred_dist) = _vision(to_preds, eye[hi], half_angle[hi], xy[hi], crowns)
        sm_angle, sm_strength = _smell(to_preds)
        out[hi, IN.smell_angle], out[hi, IN.smell_strength] = sm_angle, sm_strength
        smell_strength_self[hi] = sm_strength  # read by the mate-seeking danger check

    kin_rels = {}
    for gi, kin, at_kin in ((ci, carnivores, at_carnivores), (hi, herbivores, at_herbivores)):
        if len(gi):
            signals = np.array([k.signal for k in kin], dtype=np.float64)
            kin_rels[id(kin)] = rel = _Relative(xy[gi], angle[gi], ids[gi], at_kin)
            social = _social(rel, signals)
            for column, values in zip((IN.kin_angle, IN.kin_dist, IN.kin_density,
                                       IN.signal_angle, IN.signal_strength), social):
                out[gi, column] = values
            out[gi, IN.mate_angle], out[gi, IN.mate_dist] = _mate(
                rel, np.array([k.fertile(tick) for k in kin], dtype=bool))
    for gi, kin, other, to_other in ((ci, carnivores, herbivores, to_prey),
                                     (hi, herbivores, carnivores, to_preds)):
        if len(gi):
            for rel, targets, strength_col, angle_col in (
                    (kin_rels[id(kin)], kin, IN.touch_kin, IN.touch_kin_angle),
                    (to_other, other, IN.touch_other, IN.touch_other_angle)):
                out[gi, strength_col], out[gi, angle_col] = _touch(rel, s[gi, 5], targets)
    crown_occ = _crown_occlusion(xy, angle, whisker_angles, crowns)
    if len(ci) and carcasses:
        to_carcasses = _Relative(xy[ci], angle[ci], ids[ci], at_carcasses)
        _, _, (cc_angle, cc_dist) = _vision(to_carcasses, eye[ci], half_angle[ci], xy[ci], crowns)
        out[ci, IN.carcass_angle], out[ci, IN.carcass_dist] = cc_angle, cc_dist
        cc_smell = _smell(to_carcasses)
        out[ci, IN.carcass_smell_angle], out[ci, IN.carcass_smell_strength] = cc_smell
    out[:, IN.light] = light
    out[:, IN.memory] = [c.memory for c in creatures]
    if len(ci):
        carcass_dist = to_carcasses.dist if len(carcasses) else np.zeros((len(ci), 0))
        target_whisker_dist[ci] = _dense_target_whiskers(
            xy[ci], angle[ci], eye[ci], whisker_angles[ci],
            np.concatenate((at_herbivores[0], at_carcasses[0])),
            np.minimum(whisker_dist[ci], crown_occ[ci]),
            np.concatenate((to_prey.dist, carcass_dist), axis=1))
    if len(hi):
        food_found, (food_angle, food_dist), target_whisker_dist[hi] = _food_sight(
            xy[hi], angle[hi], eye[hi], half_angle[hi], whisker_angles[hi],
            np.minimum(whisker_dist[hi], crown_occ[hi]), food_xy, crowns)
        out[hi, IN.target_angle] = np.clip(food_angle - pred_angle * (1 - pred_dist) ** 2, -1, 1)
        out[hi, IN.target_dist] = food_dist
        target_dist[hi], target_found[hi] = food_dist, food_found
    out[:, IN.target_whiskers] = np.where(np.isfinite(target_whisker_dist),
                                            1.0 - target_whisker_dist / eye[:, None], 0.0)
    out[:, IN.season] = season
    out[:, IN.water_dist] = 1.0    # when none is sensed
    if lakes:
        gaps = _shore_gaps(xy, lakes)
        nearest = gaps.argmin(axis=0)
        gap = gaps[nearest, np.arange(n)]
        out[gap < 0, IN.in_water] = 1.0
        centre = np.array([(lake.x, lake.y) for lake in lakes], dtype=np.float64)[nearest]
        sensed = gap < WATER_SENSE_RADIUS
        to_water = _wrap(np.arctan2(centre[:, 1] - xy[:, 1], centre[:, 0] - xy[:, 0]) - angle)
        out[:, IN.water_angle] = np.where(sensed, to_water / math.pi, 0.0)
        out[:, IN.water_dist] = np.where(sensed, np.maximum(gap, 0.0) / WATER_SENSE_RADIUS, 1.0)
    out[:, IN.thirst] = [min(1.0, c.thirst / THIRST_MAX) for c in creatures]
    out[:, IN.eating] = [c.eating_ticks / c.species.eat_ticks if c.eating_ticks else 0.0
                         for c in creatures]
    out[:, IN.hurt] = [c.hurt for c in creatures]
    out[:, IN.fertile] = [c.fertile(tick) for c in creatures]

    out[:, IN.health] = s[:, 3] / s[:, 4]
    out[:, IN.sin_angle] = np.sin(angle)
    out[:, IN.cos_angle] = np.cos(angle)
    out[:, IN.size] = s[:, 5] / _MAX_SIZE
    out[:, IN.speed] = s[:, 6] / s[:, 7]
    out[:, IN.sight] = sight / MAX_NORMALIZED_SIGHT_DISTANCE
    out[:, IN.sight_angle] = sight_angle / (2 * math.pi)
    out[:, IN.stamina] = s[:, 8] / s[:, 9]
    out[:, IN.hunger] = np.minimum(1.0, s[:, 10] / HUNGER_MAX)

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
        c.nearest_target_dist = td[i] if tf[i] else None
        c.nearest_smell_strength = ss[i]
    return out
