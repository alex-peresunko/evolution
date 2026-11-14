"""Perception: the brain inputs for every creature in one compiled pass.

Columns are named by brain.layout.INPUTS.
Side effects, read by act(), the mate-seeking drive and the renderer:
c.whiskers, c.target_whiskers, c.nearest_target_dist, c.nearest_smell_strength, c.effective_sight.

The work happens in `_kernel`, a Numba function that loops over observers and targets. Each
sense evaluates the same floating-point expressions, in the same order, as the vectorized numpy
version it replaced, and breaks ties the same way (the first of equally near targets wins), so
the results are bit-identical to it. Keep that property when changing a formula: write it the
way a numpy expression over arrays would evaluate it, without fusing or reordering operations.
"""
import math

import numpy as np
from numba import njit

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
_PI = math.pi
_TWO_PI = 2 * math.pi
_INF = math.inf

# Column indices, as plain ints the kernel can use.
_OBST_W = IN.obstacle_whiskers.start
_TARGET_W = IN.target_whiskers.start
_C_TARGET_ANGLE = IN.target_angle
_C_TARGET_DIST = IN.target_dist
_C_SMELL_ANGLE = IN.smell_angle
_C_SMELL_STRENGTH = IN.smell_strength
_C_PREY_VX = IN.prey_vx
_C_PREY_VY = IN.prey_vy
_C_HEALTH = IN.health
_C_SIN = IN.sin_angle
_C_COS = IN.cos_angle
_C_SIZE = IN.size
_C_SPEED = IN.speed
_C_SIGHT = IN.sight
_C_SIGHT_ANGLE = IN.sight_angle
_C_STAMINA = IN.stamina
_C_HUNGER = IN.hunger
_C_KIN_ANGLE = IN.kin_angle
_C_KIN_DIST = IN.kin_dist
_C_KIN_DENSITY = IN.kin_density
_C_SIGNAL_ANGLE = IN.signal_angle
_C_SIGNAL_STRENGTH = IN.signal_strength
_C_CARCASS_ANGLE = IN.carcass_angle
_C_CARCASS_DIST = IN.carcass_dist
_C_LIGHT = IN.light
_C_SEASON = IN.season
_C_CARCASS_SMELL_ANGLE = IN.carcass_smell_angle
_C_CARCASS_SMELL_STRENGTH = IN.carcass_smell_strength
_C_IN_WATER = IN.in_water
_C_EATING = IN.eating
_C_HURT = IN.hurt
_C_TOUCH_KIN = IN.touch_kin
_C_TOUCH_KIN_ANGLE = IN.touch_kin_angle
_C_TOUCH_OTHER = IN.touch_other
_C_TOUCH_OTHER_ANGLE = IN.touch_other_angle
_C_MATE_ANGLE = IN.mate_angle
_C_MATE_DIST = IN.mate_dist
_C_FERTILE = IN.fertile
_C_THIRST = IN.thirst
_C_WATER_ANGLE = IN.water_angle
_C_WATER_DIST = IN.water_dist

# Observer columns of the (N, _OBS_COLS) array _gather builds.
(_X, _Y, _ANGLE, _SIGHT_D, _SIGHT_A, _HEALTH, _MAX_HEALTH, _SIZE, _CUR_SPEED, _GEN_MAX_SPEED,
 _STAMINA, _MAX_STAMINA, _HUNGER, _PREDATOR, _ID, _SIGNAL, _FERTILE, _THIRST, _EATING, _HURT,
 _MAX_SPEED) = range(21)
_OBS_COLS = 21
# Target columns of the (M, _TGT_COLS) arrays of each species.
_T_X, _T_Y, _T_ID, _T_ANGLE, _T_SPEED_RATIO, _T_SIZE, _T_SIGNAL, _T_FERTILE = range(8)
_TGT_COLS = 8

_JIT = {"cache": True, "error_model": "numpy"}


@njit(**_JIT)
def _wrap(a):
    """(a + pi) % (2 pi) - pi; Numba's float % is CPython's, which equals np.remainder."""
    return (a + _PI) % _TWO_PI - _PI


@njit(**_JIT)
def _crown_hides(ox, oy, sx, sy, crowns):
    """Whether a tree crown hides the target at (ox + sx, oy + sy) from an observer at (ox, oy).
    A crown hides what is under or behind it, except from an observer under that same crown."""
    t_count = crowns.shape[0]
    length = math.sqrt(sx * sx + sy * sy)
    close = False
    for t in range(t_count):
        # Only crowns within the sight line's length plus their radius can hide the target; the
        # +1 absorbs rounding. With one close, every crown is tested, as the numpy version did.
        reach = length + crowns[t, 2] + 1.0
        ddx = crowns[t, 0] - ox
        ddy = crowns[t, 1] - oy
        if ddx * ddx + ddy * ddy < reach * reach:
            close = True
            break
    if not close:
        return False
    denom = sx * sx + sy * sy
    tx = ox + sx
    ty = oy + sy
    for t in range(t_count):
        cx, cy, r = crowns[t, 0], crowns[t, 1], crowns[t, 2]
        r2 = r * r
        u = ((cx - ox) * sx + (cy - oy) * sy) / denom
        if u != u:
            u = 0.0
        elif u < 0.0:
            u = 0.0
        elif u > 1.0:
            u = 1.0
        px = ox + u * sx - cx
        py = oy + u * sy - cy
        if px * px + py * py < r2:
            ax, ay = ox - cx, oy - cy
            bx, by = tx - cx, ty - cy
            if not (ax * ax + ay * ay < r2 and bx * bx + by * by < r2):
                return True
    return False


@njit(**_JIT)
def _in_cone(dx, dy, dist, cos_a, sin_a, half_angle, cos_half):
    if half_angle >= _PI:
        return True
    return (dx * cos_a + dy * sin_a) / dist >= cos_half


@njit(**_JIT)
def _angle_to(x, y, angle, tgt, j):
    return _wrap(math.atan2(tgt[j, _T_Y] - y, tgt[j, _T_X] - x) - angle)


@njit(**_JIT)
def _max_size(tgt):
    """The largest body among the targets; -inf when there are none."""
    max_size = -_INF
    for j in range(tgt.shape[0]):
        max_size = max(max_size, tgt[j, _T_SIZE])
    return max_size


@njit(**_JIT)
def _touch_reach(size, max_size):
    """Bodies farther apart than this do not press (the +1 absorbs rounding); -1 when there are
    no targets (max_size -inf)."""
    if max_size == -_INF:
        return -1.0
    return (size + max_size) * BODY_RADIUS_FACTOR + TOUCH_TOLERANCE + 1.0


@njit(**_JIT)
def _press(d, size, tgt_size):
    """Touch strength: with gap = distance minus the body radii, 0.5 at contact, growing
    linearly to 1 at TOUCH_TOLERANCE deep, from 0 at TOUCH_TOLERANCE apart."""
    gap = d - (size + tgt_size) * BODY_RADIUS_FACTOR
    return min(max((TOUCH_TOLERANCE - gap) / (2 * TOUCH_TOLERANCE), 0.0), 1.0)


@njit(**_JIT)
def _scan(x, y, own_id, cos_a, sin_a, eye, half_angle, cos_half, size, tgt, crowns,
          touch_reach, whisker_reach, cos_w, sin_w, occlusion, tw):
    """One pass over the targets (rows of tgt: x, y, id, ...) for every sense of them.
    Returns (seen, near, near_d, touched, press):
      seen: the nearest target inside the sight cone (up to `eye`) not hidden by a crown;
      near, near_d: the nearest target (smell), only guaranteed within SMELL_DISTANCE;
      touched, press: the first of the bodies pressing hardest (touch_reach < 0: none).
    Indices are -1 when there is none. The first of equally near targets wins every sense.
    Targets within `whisker_reach` also shorten the target whiskers `tw` (whisker_reach < 0:
    none)."""
    seen, seen_d = -1, _INF
    near, near_d = -1, _INF
    touched, best_press = -1, 0.0
    # Nothing farther than `reach` matters; squared distances above it skip the exact hypot.
    reach = max(max(eye, SMELL_DISTANCE), max(touch_reach, whisker_reach)) + 1.0
    reach2 = reach * reach
    for j in range(tgt.shape[0]):
        dx = tgt[j, _T_X] - x
        dy = tgt[j, _T_Y] - y
        if dx * dx + dy * dy > reach2:
            continue
        d = math.hypot(dx, dy)
        if d <= whisker_reach:
            _target_whisker_hits(dx, dy, cos_w, sin_w, eye, occlusion, tw)
        if tgt[j, _T_ID] == own_id:
            continue
        if d < near_d:
            near, near_d = j, d
        if d < touch_reach:
            press = _press(d, size, tgt[j, _T_SIZE])
            if press > best_press:
                touched, best_press = j, press
        if (d > 0.0 and d <= eye and d < seen_d
                and _in_cone(dx, dy, d, cos_a, sin_a, half_angle, cos_half)
                and not (crowns.shape[0] and _crown_hides(x, y, dx, dy, crowns))):
            seen, seen_d = j, d
    return seen, near, near_d, touched, best_press


@njit(**_JIT)
def _scan_kin(x, y, own_id, size, kin, touch_reach):
    """One pass over the own species for the kin senses. Returns (near, near_d, dense, loudest,
    heard, mate, mate_d, touched, press): the nearest kin (only guaranteed within
    KIN_SENSE_RADIUS), the count within KIN_SENSE_RADIUS, the loudest signal (-1 when none is
    heard), the nearest fertile kin (only guaranteed within MATE_ATTRACTION_RADIUS) and the
    body pressing hardest. The first of equal ones wins."""
    near, near_d = -1, _INF
    dense = 0
    loudest, loudest_heard = -1, 0.0
    mate, mate_d = -1, _INF
    touched, best_press = -1, 0.0
    reach = max(max(KIN_SENSE_RADIUS, SIGNAL_RADIUS),
                max(MATE_ATTRACTION_RADIUS, touch_reach)) + 1.0
    reach2 = reach * reach
    for j in range(kin.shape[0]):
        if kin[j, _T_ID] == own_id:
            continue
        dx = kin[j, _T_X] - x
        dy = kin[j, _T_Y] - y
        if dx * dx + dy * dy > reach2:
            continue
        d = math.hypot(dx, dy)
        if d < near_d:
            near, near_d = j, d
        if d < KIN_SENSE_RADIUS:
            dense += 1
        if d < SIGNAL_RADIUS:
            heard = kin[j, _T_SIGNAL] * (1.0 - d / SIGNAL_RADIUS)
            if heard > loudest_heard:
                loudest, loudest_heard = j, heard
        if kin[j, _T_FERTILE] != 0.0 and d < mate_d:
            mate, mate_d = j, d
        if d < touch_reach:
            press = _press(d, size, kin[j, _T_SIZE])
            if press > best_press:
                touched, best_press = j, press
    return near, near_d, dense, loudest, loudest_heard, mate, mate_d, touched, best_press


@njit(**_JIT)
def _target_whisker_hits(dx, dy, cos_w, sin_w, length, occlusion, result):
    """Shorten result[k] to the distance along whisker k at which the target at offset (dx, dy)
    lies within WHISKER_TARGET_WIDTH of the ray, if it is no farther than `length` and closer
    than occlusion[k]."""
    for k in range(cos_w.shape[0]):
        t = dx * cos_w[k] + dy * sin_w[k]
        if t > 0.0 and t <= length and t < occlusion[k]:
            if abs(dy * cos_w[k] - dx * sin_w[k]) <= WHISKER_TARGET_WIDTH and t < result[k]:
                result[k] = t


@njit(**_JIT)
def _kernel(obs, whisker_angles, herbs, carns, carcasses, food_xy, rects, crowns,
            lake_shape, lake_waves, world_w, world_h, light, season, light_factor, out,
            whisker_dist, target_whisker_dist, target_dist, target_found, smell_self, eye_out):
    n = obs.shape[0]
    n_w = whisker_angles.shape[1]
    herb_max_size, carn_max_size = _max_size(herbs), _max_size(carns)

    # The food grid: food indices sorted by cell (ascending within a cell), cell starts.
    max_eye = 0.0
    for i in range(n):
        if obs[i, _PREDATOR] == 0.0:
            max_eye = max(max_eye, obs[i, _SIGHT_D] * light_factor)
    reach_max = max_eye + WHISKER_TARGET_WIDTH + 2.0
    cell = max(reach_max, 16.0)
    n_food = food_xy.shape[0]
    cols = max(1, int(world_w / cell) + 1)
    rows = max(1, int(world_h / cell) + 1)
    food_cell = np.empty(n_food, dtype=np.int64)
    counts = np.zeros(cols * rows + 1, dtype=np.int64)
    for f in range(n_food):
        cx = min(max(int(math.floor(food_xy[f, 0] / cell)), 0), cols - 1)
        cy = min(max(int(math.floor(food_xy[f, 1] / cell)), 0), rows - 1)
        food_cell[f] = cy * cols + cx
        counts[food_cell[f] + 1] += 1
    for k in range(cols * rows):
        counts[k + 1] += counts[k]
    fill = counts[:-1].copy()
    food_order = np.empty(n_food, dtype=np.int64)
    for f in range(n_food):
        food_order[fill[food_cell[f]]] = f
        fill[food_cell[f]] += 1

    cos_w = np.empty(n_w)
    sin_w = np.empty(n_w)
    occlusion = np.empty(n_w)
    tw = np.empty(n_w)

    for i in range(n):
        x, y, angle = obs[i, _X], obs[i, _Y], obs[i, _ANGLE]
        sight = obs[i, _SIGHT_D]
        sight_angle = obs[i, _SIGHT_A]
        half_angle = sight_angle / 2
        cos_half = math.cos(half_angle)
        eye = sight * light_factor
        eye_out[i] = eye
        own_id = obs[i, _ID]
        size = obs[i, _SIZE]
        predator = obs[i, _PREDATOR] != 0.0
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        if predator:
            kin, other, kin_max_size, other_max_size = carns, herbs, carn_max_size, herb_max_size
        else:
            kin, other, kin_max_size, other_max_size = herbs, carns, herb_max_size, carn_max_size

        # --- obstacle whiskers: the world edge or the first rect, capped at the sight
        for k in range(n_w):
            a = angle + whisker_angles[i, k]
            c, s = math.cos(a), math.sin(a)
            cos_w[k], sin_w[k] = c, s
            if c > 0.0:
                tx = (world_w - x) / c
            elif c < 0.0:
                tx = -x / c
            else:
                tx = _INF
            if s > 0.0:
                ty = (world_h - y) / s
            elif s < 0.0:
                ty = -y / s
            else:
                ty = _INF
            whisker_dist[i, k] = min(sight, max(0.0, min(tx, ty)))
        reach2 = (sight + 1.0) * (sight + 1.0)
        for o in range(rects.shape[0]):
            left, right, top, bottom = rects[o, 0], rects[o, 1], rects[o, 2], rects[o, 3]
            gap_x = max(max(left - x, x - right), 0.0)
            gap_y = max(max(top - y, y - bottom), 0.0)
            if not gap_x * gap_x + gap_y * gap_y <= reach2:
                continue
            for k in range(n_w):
                dx = cos_w[k] * sight
                dy = sin_w[k] * sight
                t0, t1 = -_INF, _INF
                blocked = False
                for side in range(4):
                    if side == 0:
                        p, q = -dx, x - left
                    elif side == 1:
                        p, q = dx, right - x
                    elif side == 2:
                        p, q = -dy, y - top
                    else:
                        p, q = dy, bottom - y
                    if p < 0.0:
                        t0 = max(t0, q / p)
                    elif p > 0.0:
                        t1 = min(t1, q / p)
                    elif q < 0.0:
                        blocked = True
                t0 = max(0.0, t0)
                t1 = min(1.0, t1)
                if not blocked and t0 < t1:
                    whisker_dist[i, k] = min(whisker_dist[i, k], t0 * sight)
            # (min is exact, so folding each rect in at once matches the reduction over rects)
        for k in range(n_w):
            out[i, _OBST_W + k] = 1.0 - whisker_dist[i, k] / sight

        # --- crown occlusion along each whisker: the entry of a crown ahead, or the exit of a
        # crown the observer stands under
        for k in range(n_w):
            occ = _INF
            for t in range(crowns.shape[0]):
                dx = crowns[t, 0] - x
                dy = crowns[t, 1] - y
                r2 = crowns[t, 2] * crowns[t, 2]
                along = dx * cos_w[k] + dy * sin_w[k]
                d2 = dx * dx + dy * dy
                half = math.sqrt(max(r2 - (d2 - along * along), 0.0))
                if d2 < r2:
                    v = along + half
                elif d2 - along * along < r2 and along - half > 0.0:
                    v = along - half
                else:
                    v = _INF
                occ = min(occ, v)
            occlusion[k] = min(whisker_dist[i, k], occ)

        # --- the other species: sight (carnivore: prey target; herbivore: predator), smell,
        # touch, and the carnivore's target whiskers on prey
        for k in range(n_w):
            tw[k] = _INF
        target_reach = eye + WHISKER_TARGET_WIDTH + 1.0
        seen, near, near_d, touched, press = _scan(
            x, y, own_id, cos_a, sin_a, eye, half_angle, cos_half, size, other, crowns,
            _touch_reach(size, other_max_size), target_reach if predator else -1.0,
            cos_w, sin_w, occlusion, tw)
        if seen >= 0:
            seen_angle = _angle_to(x, y, angle, other, seen) / half_angle
            seen_dist = math.hypot(other[seen, _T_X] - x, other[seen, _T_Y] - y) / eye
        else:
            seen_angle, seen_dist = 0.0, 1.0
        if near >= 0 and near_d < SMELL_DISTANCE:
            out[i, _C_SMELL_ANGLE] = _angle_to(x, y, angle, other, near) / _PI
            smell = 1.0 - near_d / SMELL_DISTANCE
            out[i, _C_SMELL_STRENGTH] = smell
            smell_self[i] = smell
        else:
            smell_self[i] = 0.0
        if touched >= 0:
            out[i, _C_TOUCH_OTHER] = press
            out[i, _C_TOUCH_OTHER_ANGLE] = _angle_to(x, y, angle, other, touched) / _PI

        # --- kin: nearest, density, the loudest signal, the nearest fertile mate, touch
        near, near_d, dense, loudest, heard, mate, mate_d, touched, press = _scan_kin(
            x, y, own_id, size, kin, _touch_reach(size, kin_max_size))
        if near >= 0 and near_d < KIN_SENSE_RADIUS:
            out[i, _C_KIN_ANGLE] = _angle_to(x, y, angle, kin, near) / _PI
            out[i, _C_KIN_DIST] = near_d / KIN_SENSE_RADIUS
        else:
            out[i, _C_KIN_DIST] = 1.0
        if kin.shape[0]:
            out[i, _C_KIN_DENSITY] = min(1.0, dense / KIN_DENSITY_NORM)
        if loudest >= 0:
            out[i, _C_SIGNAL_ANGLE] = _angle_to(x, y, angle, kin, loudest) / _PI
            out[i, _C_SIGNAL_STRENGTH] = heard
        if mate >= 0 and mate_d < MATE_ATTRACTION_RADIUS:
            out[i, _C_MATE_ANGLE] = _angle_to(x, y, angle, kin, mate) / _PI
            out[i, _C_MATE_DIST] = mate_d / MATE_ATTRACTION_RADIUS
        else:
            out[i, _C_MATE_DIST] = 1.0
        if touched >= 0:
            out[i, _C_TOUCH_KIN] = press
            out[i, _C_TOUCH_KIN_ANGLE] = _angle_to(x, y, angle, kin, touched) / _PI

        if predator:
            out[i, _C_TARGET_ANGLE], out[i, _C_TARGET_DIST] = seen_angle, seen_dist
            target_dist[i], target_found[i] = seen_dist, seen >= 0
            if seen >= 0:
                ratio = other[seen, _T_SPEED_RATIO]
                out[i, _C_PREY_VX] = math.cos(other[seen, _T_ANGLE]) * ratio
                out[i, _C_PREY_VY] = math.sin(other[seen, _T_ANGLE]) * ratio
            out[i, _C_CARCASS_DIST] = 1.0
            # carcasses: sight, smell and target whiskers
            seen, near, near_d, _, _ = _scan(
                x, y, own_id, cos_a, sin_a, eye, half_angle, cos_half, size, carcasses, crowns,
                -1.0, target_reach, cos_w, sin_w, occlusion, tw)
            if seen >= 0:
                out[i, _C_CARCASS_ANGLE] = _angle_to(x, y, angle, carcasses, seen) / half_angle
                out[i, _C_CARCASS_DIST] = math.hypot(carcasses[seen, _T_X] - x,
                                                     carcasses[seen, _T_Y] - y) / eye
            if near >= 0 and near_d < SMELL_DISTANCE:
                out[i, _C_CARCASS_SMELL_ANGLE] = _angle_to(x, y, angle, carcasses, near) / _PI
                out[i, _C_CARCASS_SMELL_STRENGTH] = 1.0 - near_d / SMELL_DISTANCE
        else:
            out[i, _C_CARCASS_DIST] = 1.0
            # food: target whiskers, then the nearest item in the cone (the first of equally
            # near ones in food_xy order)
            food_best, food_best_d = -1, _INF
            if n_food:
                # Cells within target_reach, plus 1 px so rounding never loses an item.
                pad = target_reach + 1.0
                gx0 = min(max(int(math.floor((x - pad) / cell)), 0), cols - 1)
                gx1 = min(max(int(math.floor((x + pad) / cell)), 0), cols - 1)
                gy0 = min(max(int(math.floor((y - pad) / cell)), 0), rows - 1)
                gy1 = min(max(int(math.floor((y + pad) / cell)), 0), rows - 1)
                for gy in range(gy0, gy1 + 1):
                    for gx in range(gx0, gx1 + 1):
                        g = gy * cols + gx
                        for at in range(counts[g], counts[g + 1]):
                            f = food_order[at]
                            dx = food_xy[f, 0] - x
                            dy = food_xy[f, 1] - y
                            if dx * dx + dy * dy > pad * pad:
                                continue
                            d = math.hypot(dx, dy)
                            if not d <= target_reach:
                                continue
                            _target_whisker_hits(dx, dy, cos_w, sin_w, eye, occlusion, tw)
                            if not (d > 0.0 and d <= eye):
                                continue
                            if d > food_best_d or (d == food_best_d and f > food_best):
                                continue
                            if not _in_cone(dx, dy, d, cos_a, sin_a, half_angle, cos_half):
                                continue
                            if crowns.shape[0] and _crown_hides(x, y, dx, dy, crowns):
                                continue
                            food_best, food_best_d = f, d
            if food_best >= 0:
                food_angle = _wrap(math.atan2(food_xy[food_best, 1] - y,
                                              food_xy[food_best, 0] - x) - angle) / half_angle
                food_dist = food_best_d / eye
            else:
                food_angle, food_dist = 0.0, 1.0
            fade = 1 - seen_dist
            blended = food_angle - seen_angle * (fade * fade)
            out[i, _C_TARGET_ANGLE] = min(max(blended, -1.0), 1.0)
            out[i, _C_TARGET_DIST] = food_dist
            target_dist[i], target_found[i] = food_dist, food_best >= 0
        for k in range(n_w):
            target_whisker_dist[i, k] = tw[k]
            out[i, _TARGET_W + k] = 1.0 - tw[k] / eye if tw[k] < _INF else 0.0

        # --- water: the lake blob whose shore is nearest
        out[i, _C_WATER_DIST] = 1.0
        n_lakes = lake_shape.shape[0]
        if n_lakes:
            best, best_gap = 0, _INF
            for lk in range(n_lakes):
                dx = x - lake_shape[lk, 0]
                dy = y - lake_shape[lk, 1]
                # The shore is at most max_radius from the centre; the +1 absorbs rounding.
                limit = lake_shape[lk, 3] + WATER_SENSE_RADIUS + 1.0
                if dx * dx + dy * dy > (limit + 1.0) * (limit + 1.0):
                    continue
                d = math.hypot(dx, dy)
                # Skip blobs whose shore cannot come closer than the best so far (the shore
                # lies within max_radius; the 1 px covers rounding).
                if not d < limit or d - lake_shape[lk, 3] - 1.0 > best_gap:
                    continue
                theta = math.atan2(dy, dx)
                ripple = 0.0
                for h in range(lake_waves.shape[1]):
                    ripple = ripple + lake_waves[lk, h, 1] * math.cos(
                        lake_waves[lk, h, 0] * theta + lake_waves[lk, h, 2])
                gap = d - lake_shape[lk, 2] * (1 + ripple)
                if gap < best_gap:
                    best, best_gap = lk, gap
            if best_gap < 0.0:
                out[i, _C_IN_WATER] = 1.0
            if best_gap < WATER_SENSE_RADIUS:
                to_water = _wrap(math.atan2(lake_shape[best, 1] - y, lake_shape[best, 0] - x)
                                 - angle)
                out[i, _C_WATER_ANGLE] = to_water / _PI
                out[i, _C_WATER_DIST] = max(best_gap, 0.0) / WATER_SENSE_RADIUS

        # --- the creature itself
        out[i, _C_LIGHT] = light
        out[i, _C_SEASON] = season
        out[i, _C_THIRST] = obs[i, _THIRST]
        out[i, _C_EATING] = obs[i, _EATING]
        out[i, _C_HURT] = obs[i, _HURT]
        out[i, _C_FERTILE] = obs[i, _FERTILE]
        out[i, _C_HEALTH] = obs[i, _HEALTH] / obs[i, _MAX_HEALTH]
        out[i, _C_SIN] = sin_a
        out[i, _C_COS] = cos_a
        out[i, _C_SIZE] = size / _MAX_SIZE
        out[i, _C_SPEED] = obs[i, _CUR_SPEED] / obs[i, _GEN_MAX_SPEED]
        out[i, _C_SIGHT] = sight / MAX_NORMALIZED_SIGHT_DISTANCE
        out[i, _C_SIGHT_ANGLE] = sight_angle / _TWO_PI
        out[i, _C_STAMINA] = obs[i, _STAMINA] / obs[i, _MAX_STAMINA]
        out[i, _C_HUNGER] = min(1.0, obs[i, _HUNGER] / HUNGER_MAX)


def _gather(creatures, tick):
    """(N, _OBS_COLS) array of what perception reads from each creature."""
    return np.array([
        (c.pos.x, c.pos.y, c.angle, c.sight_distance, c.sight_angle, c.health, c.max_health,
         c.size, c.current_speed, c.genetic_max_speed, c.stamina, c.max_stamina, c.hunger,
         c.species.is_predator, c.id, c.signal, c.fertile(tick),
         min(1.0, c.thirst / THIRST_MAX),
         c.eating_ticks / c.species.eat_ticks if c.eating_ticks else 0.0,
         c.hurt, c.max_speed)
        for c in creatures
    ], dtype=np.float64).reshape(-1, _OBS_COLS)


def _targets(obs):
    """(M, _TGT_COLS) target rows from observer rows."""
    t = np.empty((len(obs), _TGT_COLS))
    t[:, _T_X] = obs[:, _X]
    t[:, _T_Y] = obs[:, _Y]
    t[:, _T_ID] = obs[:, _ID]
    t[:, _T_ANGLE] = obs[:, _ANGLE]
    max_speed = obs[:, _MAX_SPEED]
    t[:, _T_SPEED_RATIO] = np.divide(obs[:, _CUR_SPEED], max_speed,
                                     out=np.zeros(len(obs)), where=max_speed > 0)
    t[:, _T_SIZE] = obs[:, _SIZE]
    t[:, _T_SIGNAL] = obs[:, _SIGNAL]
    t[:, _T_FERTILE] = obs[:, _FERTILE]
    return t


def _carcass_rows(carcasses):
    t = np.zeros((len(carcasses), _TGT_COLS))
    if carcasses:
        t[:, :3] = [(k.pos.x, k.pos.y, k.id) for k in carcasses]
    return t


class _Static:
    """Arrays of the obstacles and lakes, rebuilt only when those lists change."""

    def __init__(self):
        self._key = None

    def get(self, obstacles, lakes):
        key = (id(obstacles), len(obstacles), id(lakes), len(lakes))
        if key != self._key:
            self._key = key
            self._obstacles, self._lakes = obstacles, lakes  # keep the ids alive
            self.rects = np.array(
                [(o.rect.left, o.rect.right, o.rect.top, o.rect.bottom) for o in obstacles],
                dtype=np.float64).reshape(-1, 4)
            self.crowns = np.array(
                [(o.x, o.y, o.crown_radius) for o in obstacles if hasattr(o, "crown_radius")],
                dtype=np.float64).reshape(-1, 3)
            harmonics = max((len(lake.harmonics) for lake in lakes), default=0)
            # Blobs with fewer harmonics are padded with zero amplitudes.
            self.waves = np.zeros((len(lakes), harmonics, 3))
            for i, lake in enumerate(lakes):
                if lake.harmonics:
                    self.waves[i, :len(lake.harmonics)] = lake.harmonics
            self.lakes = np.array([(lake.x, lake.y, lake.radius, lake.max_radius)
                                   for lake in lakes], dtype=np.float64).reshape(-1, 4)
        return self


_static = _Static()


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

    obs = _gather(creatures, tick)
    n_herbs = len(herbivores)
    # The world passes creatures as herbivores + carnivores: their rows are the targets' rows.
    if (len(carnivores) + n_herbs == n and creatures[:n_herbs] == herbivores
            and creatures[n_herbs:] == carnivores):
        herb_rows, carn_rows = _targets(obs[:n_herbs]), _targets(obs[n_herbs:])
    else:
        herb_rows = _targets(_gather(herbivores, tick))
        carn_rows = _targets(_gather(carnivores, tick))
    if food_xy is None:
        food_xy = np.array([(o.pos.x, o.pos.y) for o in food_items],
                           dtype=np.float64).reshape(-1, 2)
    whisker_angles = np.array([c.whisker_angles for c in creatures], dtype=np.float64)
    static = _static.get(obstacles, lakes)
    world_w, world_h = creatures[0].world_bounds

    whisker_dist = np.empty((n, NUM_WHISKERS))
    target_whisker_dist = np.empty((n, NUM_WHISKERS))
    target_dist = np.empty(n)
    target_found = np.zeros(n, dtype=np.bool_)
    smell_self = np.empty(n)
    eye = np.empty(n)
    _kernel(obs, whisker_angles, herb_rows, carn_rows, _carcass_rows(carcasses),
            np.ascontiguousarray(food_xy, dtype=np.float64), static.rects, static.crowns,
            static.lakes, static.waves, float(world_w), float(world_h), float(light),
            float(season), NIGHT_SIGHT_FACTOR + (1 - NIGHT_SIGHT_FACTOR) * light, out,
            whisker_dist, target_whisker_dist, target_dist, target_found, smell_self, eye)
    out[:, IN.memory] = [c.memory for c in creatures]

    wl = whisker_dist.tolist()
    twl = target_whisker_dist.tolist()
    td = target_dist.tolist()
    tf = target_found.tolist()
    ss = smell_self.tolist()
    es = eye.tolist()
    inf = math.inf
    for i, c in enumerate(creatures):
        c.effective_sight = es[i]
        c.whiskers = wl[i]
        c.target_whiskers = [None if d == inf else d for d in twl[i]]
        c.nearest_target_dist = td[i] if tf[i] else None
        c.nearest_smell_strength = ss[i]
    return out
