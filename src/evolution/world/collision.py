"""Solid bodies: push overlapping creatures apart and out of carcasses."""
import numpy as np

from ..config import BODY_RADIUS_FACTOR, COLLISION_PASSES


def _push(dx, dy, dist, overlap):
    """Unit direction from the first body to the second, scaled by overlap; coincident centres
    get a fixed direction."""
    safe = np.where(dist > 0, dist, 1.0)
    ux = np.where(dist > 0, dx / safe, 1.0)
    uy = np.where(dist > 0, dy / safe, 0.0)
    return ux * overlap, uy * overlap


def separate(creatures, carcasses, obstacles):
    """Move live creatures so that none overlaps another creature or a carcass. Creature pairs
    split the push by mass (size²); carcasses never move. A creature pushed into an obstacle
    keeps its old position."""
    if BODY_RADIUS_FACTOR <= 0:
        return
    live = [c for c in creatures if c.health > 0]
    n = len(live)
    if n == 0:
        return
    xy0 = np.array([(c.pos.x, c.pos.y) for c in live])
    r = np.array([c.size for c in live]) * BODY_RADIUS_FACTOR
    xy = xy0.copy()

    i, j = np.triu_indices(n, 1)
    near = np.hypot(xy[j, 0] - xy[i, 0], xy[j, 1] - xy[i, 1]) < r[i] + r[j]
    i, j = i[near], j[near]
    mass = r * r
    share_i = mass[j] / (mass[i] + mass[j])

    solid = [k for k in carcasses if k.meat > 0]
    if solid:
        cxy = np.array([(k.pos.x, k.pos.y) for k in solid])
        cr = np.array([k.radius for k in solid]) * BODY_RADIUS_FACTOR
        ci, ck = np.nonzero(np.hypot(xy[:, None, 0] - cxy[None, :, 0],
                                     xy[:, None, 1] - cxy[None, :, 1]) < r[:, None] + cr[None, :])
    else:
        ci = ck = np.empty(0, dtype=int)
    if len(i) == 0 and len(ci) == 0:
        return

    for _ in range(COLLISION_PASSES):
        shift = np.zeros_like(xy)
        if len(i):
            dx, dy = xy[j, 0] - xy[i, 0], xy[j, 1] - xy[i, 1]
            dist = np.hypot(dx, dy)
            px, py = _push(dx, dy, dist, np.maximum(0.0, r[i] + r[j] - dist))
            np.add.at(shift[:, 0], i, -px * share_i)
            np.add.at(shift[:, 1], i, -py * share_i)
            np.add.at(shift[:, 0], j, px * (1 - share_i))
            np.add.at(shift[:, 1], j, py * (1 - share_i))
        if len(ci):
            dx, dy = xy[ci, 0] - cxy[ck, 0], xy[ci, 1] - cxy[ck, 1]
            dist = np.hypot(dx, dy)
            px, py = _push(dx, dy, dist, np.maximum(0.0, r[ci] + cr[ck] - dist))
            np.add.at(shift[:, 0], ci, px)
            np.add.at(shift[:, 1], ci, py)
        xy += shift

    moved = np.flatnonzero(np.any(xy != xy0, axis=1))
    for m in moved.tolist():
        c = live[m]
        old_x, old_y = c.pos.x, c.pos.y
        c.pos.x, c.pos.y = float(xy[m, 0]), float(xy[m, 1])
        c.clamp_to_walls()
        body = c.geometry.rect(c.pos.x - c.size, c.pos.y - c.size, c.size * 2, c.size * 2)
        if any(obs.rect.colliderect(body) for obs in obstacles):
            c.pos.x, c.pos.y = old_x, old_y
