"""Solid bodies: collision impulses, knockback, and pushing overlapping bodies apart."""
import numpy as np

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


def _unit(dx, dy, dist):
    """Unit direction from the first body to the second; coincident centres get (1, 0)."""
    safe = np.where(dist > 0, dist, 1.0)
    return np.where(dist > 0, dx / safe, 1.0), np.where(dist > 0, dy / safe, 0.0)


def _overlapping(axy, ar, bxy, br, distinct=False):
    """(i, j) index arrays of the pairs with hypot(b[j] - a[i]) < ar[i] + br[j], ordered by
    i then j, as np.nonzero of the full (A, B) matrix would give them; distinct=True keeps only
    i < j (a and b being the same bodies)."""
    empty = np.empty(0, dtype=np.intp)
    if len(axy) == 0 or len(bxy) == 0:
        return empty, empty
    order = np.argsort(bxy[:, 0], kind="stable")
    bx = bxy[order, 0]
    reach = ar + br.max() + 1.0   # +1: never lose a pair to rounding at the window edge
    lo = np.searchsorted(bx, axy[:, 0] - reach, "left")
    counts = np.searchsorted(bx, axy[:, 0] + reach, "right") - lo
    total = int(counts.sum())
    if total == 0:
        return empty, empty
    i = np.repeat(np.arange(len(axy)), counts)
    starts = np.repeat(np.cumsum(counts) - counts, counts)
    j = order[np.repeat(lo, counts) + np.arange(total) - starts]
    if distinct:
        keep = i < j
        i, j = i[keep], j[keep]
    near = np.hypot(bxy[j, 0] - axy[i, 0], bxy[j, 1] - axy[i, 1]) < ar[i] + br[j]
    i, j = i[near], j[near]
    by_pair = np.lexsort((j, i))
    return i[by_pair], j[by_pair]


def _impulses(xy, v, mass, i, j, cxy, ci, ck):
    """Velocity change (N, 2) of each creature from the collisions of approaching pairs."""
    dv = np.zeros_like(v)
    if len(i):
        nx, ny = _unit(xy[j, 0] - xy[i, 0], xy[j, 1] - xy[i, 1],
                       np.hypot(xy[j, 0] - xy[i, 0], xy[j, 1] - xy[i, 1]))
        vn = (v[j, 0] - v[i, 0]) * nx + (v[j, 1] - v[i, 1]) * ny
        impulse = np.where(vn < 0, -(1 + RESTITUTION) * vn / (1 / mass[i] + 1 / mass[j]), 0.0)
        np.add.at(dv[:, 0], i, -impulse / mass[i] * nx)
        np.add.at(dv[:, 1], i, -impulse / mass[i] * ny)
        np.add.at(dv[:, 0], j, impulse / mass[j] * nx)
        np.add.at(dv[:, 1], j, impulse / mass[j] * ny)
    if len(ci):
        dx, dy = xy[ci, 0] - cxy[ck, 0], xy[ci, 1] - cxy[ck, 1]
        nx, ny = _unit(dx, dy, np.hypot(dx, dy))
        vn = v[ci, 0] * nx + v[ci, 1] * ny
        bounce = np.where(vn < 0, -(1 + RESTITUTION) * vn, 0.0)
        np.add.at(dv[:, 0], ci, bounce * nx)
        np.add.at(dv[:, 1], ci, bounce * ny)
    return dv


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
    fl, it = table.fl, table.it
    live = np.flatnonzero(fl[:, T.HEALTH] > 0)
    n = len(live)
    if n == 0:
        return
    xy0 = fl[np.ix_(live, (T.X, T.Y))]
    r = fl[live, T.SIZE] * BODY_RADIUS_FACTOR
    knock0 = fl[np.ix_(live, (T.KNOCK_X, T.KNOCK_Y))]
    knock = knock0.copy()
    xy = xy0.copy()

    i, j = _overlapping(xy, r, xy, r, distinct=True)
    mass = r * r
    share_i = mass[j] / (mass[i] + mass[j])

    solid = [k for k in carcasses if k.meat > 0]
    if solid:
        cxy = np.array([(k.pos.x, k.pos.y) for k in solid])
        cr = np.array([k.radius for k in solid]) * BODY_RADIUS_FACTOR
        ci, ck = _overlapping(xy, r, cxy, cr)
    else:
        cxy = np.empty((0, 2))
        ci = ck = np.empty(0, dtype=int)
    if len(i) == 0 and len(ci) == 0 and not knock0.any():
        return

    if len(i) or len(ci):
        speed = fl[live, T.SPEED]
        angle = fl[live, T.ANGLE]
        v = np.column_stack((np.cos(angle) * speed, np.sin(angle) * speed)) + knock
        dv = _impulses(xy, v, mass, i, j, cxy, ci, ck)
        knock += dv
        excess = np.hypot(dv[:, 0], dv[:, 1]) - IMPACT_DAMAGE_THRESHOLD
        hit = excess > 0
        if hit.any():
            rows = live[hit]
            fl[rows, T.HEALTH] -= IMPACT_DAMAGE * excess[hit]
            it[rows, T.HEALTH_INT] = 0
            it[rows, T.LAST_IMPACT] = tick
    xy += knock

    for _ in range(COLLISION_PASSES):
        shift = np.zeros_like(xy)
        if len(i):
            dx, dy = xy[j, 0] - xy[i, 0], xy[j, 1] - xy[i, 1]
            dist = np.hypot(dx, dy)
            ux, uy = _unit(dx, dy, dist)
            overlap = np.maximum(0.0, r[i] + r[j] - dist)
            px, py = ux * overlap, uy * overlap
            np.add.at(shift[:, 0], i, -px * share_i)
            np.add.at(shift[:, 1], i, -py * share_i)
            np.add.at(shift[:, 0], j, px * (1 - share_i))
            np.add.at(shift[:, 1], j, py * (1 - share_i))
        if len(ci):
            dx, dy = xy[ci, 0] - cxy[ck, 0], xy[ci, 1] - cxy[ck, 1]
            dist = np.hypot(dx, dy)
            ux, uy = _unit(dx, dy, dist)
            overlap = np.maximum(0.0, r[ci] + cr[ck] - dist)
            np.add.at(shift[:, 0], ci, ux * overlap)
            np.add.at(shift[:, 1], ci, uy * overlap)
        xy += shift

    knock *= KNOCK_FRICTION
    knock[np.hypot(knock[:, 0], knock[:, 1]) < KNOCK_MIN] = 0.0
    moved = np.flatnonzero(np.any(xy != xy0, axis=1))
    if len(moved):
        # Creature.clamp_to_walls, then the obstacle check, on the wanted positions.
        rows = live[moved]
        size = fl[rows, T.SIZE]
        want = xy[moved]
        w, h = table.members[rows[0]].world_bounds
        x = np.minimum(np.maximum(want[:, 0], size), w - size)
        y = np.minimum(np.maximum(want[:, 1], size), h - size)
        knock[moved[x != want[:, 0]], 0] = 0.0
        knock[moved[y != want[:, 1]], 1] = 0.0
        members = table.members
        for k, (m, row, cx, cy, s) in enumerate(zip(moved.tolist(), rows.tolist(), x.tolist(),
                                                    y.tolist(), size.tolist())):
            body = members[row].geometry.rect(cx - s, cy - s, s * 2, s * 2)
            nearby = (terrain.near(cx, cy)[0]
                      if terrain is not None and s <= terrain.reach else obstacles)
            if any(obs.rect.colliderect(body) for obs in nearby):   # stay where it was
                x[k], y[k] = xy0[m]
                knock[m] = 0.0
        fl[rows, T.X] = x
        fl[rows, T.Y] = y
    fl[live, T.KNOCK_X] = knock[:, 0]
    fl[live, T.KNOCK_Y] = knock[:, 1]
