"""Solid bodies: collision impulses, knockback, and pushing overlapping bodies apart."""
import math

import numpy as np
from numba import njit

from ..config import (
    BODY_RADIUS_FACTOR,
    COLLISION_PASSES,
    IMPACT_DAMAGE,
    IMPACT_DAMAGE_THRESHOLD,
    KNOCK_FRICTION,
    KNOCK_MIN,
    RESTITUTION,
)
from . import creature_table as T
from .creature_table import CreatureTable

_JIT = {"cache": True, "error_model": "numpy"}

# Table columns and config values reach the kernel as arguments: Numba would freeze module
# globals, and its disk cache does not notice changes to other modules.
(_X, _Y, _HEALTH, _SIZE, _SPEED, _ANGLE, _KNOCK_X, _KNOCK_Y, _HEALTH_INT,
 _LAST_IMPACT) = range(10)
_COLUMNS = np.array([T.X, T.Y, T.HEALTH, T.SIZE, T.SPEED, T.ANGLE, T.KNOCK_X, T.KNOCK_Y,
                     T.HEALTH_INT, T.LAST_IMPACT], dtype=np.int64)
(_K_RADIUS, _K_PASSES, _K_DAMAGE, _K_THRESHOLD, _K_FRICTION, _K_KNOCK_MIN,
 _K_BOUNCE) = range(7)


def _tunables():
    return np.array([BODY_RADIUS_FACTOR, COLLISION_PASSES, IMPACT_DAMAGE,
                     IMPACT_DAMAGE_THRESHOLD, KNOCK_FRICTION, KNOCK_MIN, 1 + RESTITUTION],
                    dtype=np.float64)


@njit(**_JIT)
def _overlapping(axy, ar, bxy, br, distinct=False):
    """(i, j) index arrays of the pairs with hypot(b[j] - a[i]) < ar[i] + br[j], ordered by
    i then j, as np.nonzero of the full (A, B) matrix would give them; distinct=True keeps only
    i < j (a and b being the same bodies)."""
    na, nb = axy.shape[0], bxy.shape[0]
    size = 4 * na + 16
    pi = np.empty(size, dtype=np.intp)
    pj = np.empty(size, dtype=np.intp)
    count = 0
    if na == 0 or nb == 0:
        return pi[:0], pj[:0]
    order = np.argsort(bxy[:, 0])
    bx = bxy[order, 0]
    widest = br.max()
    for i in range(na):
        x, y = axy[i, 0], axy[i, 1]
        reach = ar[i] + widest
        first = count
        at = np.searchsorted(bx, x - reach)
        while at < nb and bx[at] <= x + reach:
            j = order[at]
            at += 1
            if distinct and j <= i:
                continue
            if math.hypot(bxy[j, 0] - x, bxy[j, 1] - y) < ar[i] + br[j]:
                if count == size:
                    size *= 2
                    grown_i = np.empty(size, dtype=np.intp)
                    grown_j = np.empty(size, dtype=np.intp)
                    grown_i[:count] = pi[:count]
                    grown_j[:count] = pj[:count]
                    pi, pj = grown_i, grown_j
                pi[count] = i
                pj[count] = j
                count += 1
        for e in range(first + 1, count):      # this i's partners, ascending
            j = pj[e]
            d = e
            while d > first and pj[d - 1] > j:
                pj[d] = pj[d - 1]
                d -= 1
            pj[d] = j
    return pi[:count], pj[:count]


@njit(**_JIT)
def _hits_rect(rect, x, y, size, pygame):
    """rect.colliderect(the body's bounding rect); pygame.Rect truncates to ints."""
    bx, by, bw, bh = x - size, y - size, size * 2, size * 2
    if pygame:
        bx, by, bw, bh = float(int(bx)), float(int(by)), float(int(bw)), float(int(bh))
        if bw == 0 or bh == 0 or rect[2] == 0 or rect[3] == 0:
            return False
    return (rect[0] < bx + bw and rect[0] + rect[2] > bx
            and rect[1] < by + bh and rect[1] + rect[3] > by)


@njit(**_JIT)
def _blocked(x, y, size, pygame, rects, cell_start, cell_rects, cell, cols, rows, reach):
    """Whether a body at x, y touches an obstacle: those of its terrain cell, or every one
    when there are no cells or the body is larger than they cover."""
    if cols == 0 or size > reach:
        for o in range(rects.shape[0]):
            if _hits_rect(rects[o], x, y, size, pygame):
                return True
        return False
    at = (min(cols - 1, max(0, int(x // cell))) * rows
          + min(rows - 1, max(0, int(y // cell))))
    for e in range(cell_start[at], cell_start[at + 1]):
        if _hits_rect(rects[cell_rects[e]], x, y, size, pygame):
            return True
    return False


@njit(**_JIT)
def _separate(fl, it, col, k, tick, cxy, cr, world_w, world_h, pygame, rects, cell_start,
              cell_rects, cell, cols, rows, reach):
    live = np.flatnonzero(fl[:, col[_HEALTH]] > 0)
    n = live.shape[0]
    if n == 0:
        return
    xy0 = np.empty((n, 2))
    knock = np.empty((n, 2))
    r = np.empty(n)
    moving = False
    for m in range(n):
        row = live[m]
        xy0[m, 0], xy0[m, 1] = fl[row, col[_X]], fl[row, col[_Y]]
        knock[m, 0], knock[m, 1] = fl[row, col[_KNOCK_X]], fl[row, col[_KNOCK_Y]]
        r[m] = fl[row, col[_SIZE]] * k[_K_RADIUS]
        moving = moving or knock[m, 0] != 0 or knock[m, 1] != 0
    xy = xy0.copy()
    pi, pj = _overlapping(xy, r, xy, r, True)
    ci, ck = _overlapping(xy, r, cxy, cr, False)
    pairs, touching = pi.shape[0], ci.shape[0]
    if pairs == 0 and touching == 0 and not moving:
        return
    mass = r * r
    ux = np.empty(max(pairs, touching))
    uy = np.empty(max(pairs, touching))
    push = np.empty(max(pairs, touching))

    if pairs or touching:
        v = np.empty((n, 2))
        for m in range(n):
            speed, angle = fl[live[m], col[_SPEED]], fl[live[m], col[_ANGLE]]
            v[m, 0] = math.cos(angle) * speed + knock[m, 0]
            v[m, 1] = math.sin(angle) * speed + knock[m, 1]
        dv = np.zeros((n, 2))
        # Sums run over the pairs' first bodies, then over their second ones.
        for e in range(pairs):
            i, j = pi[e], pj[e]
            dx, dy = xy[j, 0] - xy[i, 0], xy[j, 1] - xy[i, 1]
            dist = math.hypot(dx, dy)
            ux[e], uy[e] = (dx / dist, dy / dist) if dist > 0 else (1.0, 0.0)
            vn = (v[j, 0] - v[i, 0]) * ux[e] + (v[j, 1] - v[i, 1]) * uy[e]
            push[e] = -k[_K_BOUNCE] * vn / (1 / mass[i] + 1 / mass[j]) if vn < 0 else 0.0
            dv[i, 0] += -push[e] / mass[i] * ux[e]
            dv[i, 1] += -push[e] / mass[i] * uy[e]
        for e in range(pairs):
            j = pj[e]
            dv[j, 0] += push[e] / mass[j] * ux[e]
            dv[j, 1] += push[e] / mass[j] * uy[e]
        for e in range(touching):      # carcasses have infinite mass
            i, c = ci[e], ck[e]
            dx, dy = xy[i, 0] - cxy[c, 0], xy[i, 1] - cxy[c, 1]
            dist = math.hypot(dx, dy)
            nx, ny = (dx / dist, dy / dist) if dist > 0 else (1.0, 0.0)
            vn = v[i, 0] * nx + v[i, 1] * ny
            bounce = -k[_K_BOUNCE] * vn if vn < 0 else 0.0
            dv[i, 0] += bounce * nx
            dv[i, 1] += bounce * ny
        for m in range(n):
            knock[m, 0] += dv[m, 0]
            knock[m, 1] += dv[m, 1]
            excess = math.hypot(dv[m, 0], dv[m, 1]) - k[_K_THRESHOLD]
            if excess > 0:
                row = live[m]
                fl[row, col[_HEALTH]] -= k[_K_DAMAGE] * excess
                it[row, col[_HEALTH_INT]] = 0
                it[row, col[_LAST_IMPACT]] = tick
    for m in range(n):
        xy[m, 0] += knock[m, 0]
        xy[m, 1] += knock[m, 1]

    shift = np.empty((n, 2))
    for _ in range(int(k[_K_PASSES])):
        shift[:] = 0.0
        for e in range(pairs):
            i, j = pi[e], pj[e]
            dx, dy = xy[j, 0] - xy[i, 0], xy[j, 1] - xy[i, 1]
            dist = math.hypot(dx, dy)
            nx, ny = (dx / dist, dy / dist) if dist > 0 else (1.0, 0.0)
            overlap = max(0.0, r[i] + r[j] - dist)
            ux[e], uy[e] = nx * overlap, ny * overlap
            share = mass[j] / (mass[i] + mass[j])
            push[e] = 1 - share
            shift[i, 0] += -ux[e] * share
            shift[i, 1] += -uy[e] * share
        for e in range(pairs):
            j = pj[e]
            shift[j, 0] += ux[e] * push[e]
            shift[j, 1] += uy[e] * push[e]
        for e in range(touching):
            i, c = ci[e], ck[e]
            dx, dy = xy[i, 0] - cxy[c, 0], xy[i, 1] - cxy[c, 1]
            dist = math.hypot(dx, dy)
            nx, ny = (dx / dist, dy / dist) if dist > 0 else (1.0, 0.0)
            overlap = max(0.0, r[i] + cr[c] - dist)
            shift[i, 0] += nx * overlap
            shift[i, 1] += ny * overlap
        for m in range(n):
            xy[m, 0] += shift[m, 0]
            xy[m, 1] += shift[m, 1]

    for m in range(n):
        row = live[m]
        knock[m, 0] *= k[_K_FRICTION]
        knock[m, 1] *= k[_K_FRICTION]
        if math.hypot(knock[m, 0], knock[m, 1]) < k[_K_KNOCK_MIN]:
            knock[m, 0] = knock[m, 1] = 0.0
        if xy[m, 0] != xy0[m, 0] or xy[m, 1] != xy0[m, 1]:
            # Creature.clamp_to_walls, then the obstacle check, on the wanted position.
            size = fl[row, col[_SIZE]]
            x = min(max(xy[m, 0], size), world_w - size)
            y = min(max(xy[m, 1], size), world_h - size)
            if x != xy[m, 0]:
                knock[m, 0] = 0.0
            if y != xy[m, 1]:
                knock[m, 1] = 0.0
            if _blocked(x, y, size, pygame, rects, cell_start, cell_rects, cell, cols, rows,
                        reach):      # stay where it was
                x, y = xy0[m, 0], xy0[m, 1]
                knock[m, 0] = knock[m, 1] = 0.0
            fl[row, col[_X]], fl[row, col[_Y]] = x, y
        fl[row, col[_KNOCK_X]], fl[row, col[_KNOCK_Y]] = knock[m, 0], knock[m, 1]


_NO_CELLS = np.zeros(1, dtype=np.int64)


def _rects(obstacles):
    return np.array([(o.rect.x, o.rect.y, o.rect.w, o.rect.h) for o in obstacles],
                    dtype=np.float64).reshape(-1, 4)


def separate(creatures, carcasses, obstacles, tick, terrain=None):
    """Collide live creatures with each other and with carcasses, which never move.

    Approaching bodies exchange an impulse (mass size², RESTITUTION) that is added to their
    `knock` velocity; a velocity change above IMPACT_DAMAGE_THRESHOLD costs health and sets
    `last_impact_tick`. Every creature then moves by its knock, overlaps are pushed apart split
    by mass, and the knock decays by KNOCK_FRICTION. A creature moved into an obstacle keeps
    its old position and loses its knock; a wall stops the knock along that axis. `terrain`
    (a TerrainGrid of `obstacles`) narrows that obstacle check."""
    if BODY_RADIUS_FACTOR <= 0 or not creatures:
        return
    table = creatures[0]._t
    if not table.holds(creatures):
        table = CreatureTable()
        table.arrange(creatures)
    solid = [(c.pos.x, c.pos.y, c.radius * BODY_RADIUS_FACTOR) for c in carcasses if c.meat > 0]
    bodies = np.array(solid, dtype=np.float64).reshape(-1, 3)
    if terrain is None:
        rects, cell_start, cell_rects, cell, cols, rows, reach = (
            _rects(obstacles), _NO_CELLS, _NO_CELLS, 1.0, 0, 0, 0.0)
    else:
        cell_start, cell_rects, rects = terrain.obstacle_arrays()
        cell, cols, rows, reach = (float(terrain.cell_size), terrain.cols, terrain.rows,
                                   float(terrain.reach))
    first = creatures[0]
    w, h = first.world_bounds
    _separate(table.fl, table.it, _COLUMNS, _tunables(), tick,
              np.ascontiguousarray(bodies[:, :2]), np.ascontiguousarray(bodies[:, 2]),
              float(w), float(h), first.geometry.kind == "pygame", rects, cell_start,
              cell_rects, cell, cols, rows, reach)
