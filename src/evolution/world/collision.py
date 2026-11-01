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


def _unit(dx, dy, dist):
    """Unit direction from the first body to the second; coincident centres get (1, 0)."""
    safe = np.where(dist > 0, dist, 1.0)
    return np.where(dist > 0, dx / safe, 1.0), np.where(dist > 0, dy / safe, 0.0)


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


def separate(creatures, carcasses, obstacles, tick):
    """Collide live creatures with each other and with carcasses, which never move.

    Approaching bodies exchange an impulse (mass size², RESTITUTION) that is added to their
    `knock` velocity; a velocity change above IMPACT_DAMAGE_THRESHOLD costs health and sets
    `last_impact_tick`. Every creature then moves by its knock, overlaps are pushed apart split
    by mass, and the knock decays by KNOCK_FRICTION. A creature moved into an obstacle keeps
    its old position and loses its knock; a wall stops the knock along that axis."""
    if BODY_RADIUS_FACTOR <= 0:
        return
    live = [c for c in creatures if c.health > 0]
    n = len(live)
    if n == 0:
        return
    xy0 = np.array([(c.pos.x, c.pos.y) for c in live])
    r = np.array([c.size for c in live]) * BODY_RADIUS_FACTOR
    knock0 = np.array([(c.knock.x, c.knock.y) for c in live])
    knock = knock0.copy()
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
        cxy = np.empty((0, 2))
        ci = ck = np.empty(0, dtype=int)
    if len(i) == 0 and len(ci) == 0 and not knock0.any():
        return

    if len(i) or len(ci):
        speed = np.array([c.current_speed for c in live])
        angle = np.array([c.angle for c in live])
        v = np.column_stack((np.cos(angle) * speed, np.sin(angle) * speed)) + knock
        dv = _impulses(xy, v, mass, i, j, cxy, ci, ck)
        knock += dv
        excess = np.hypot(dv[:, 0], dv[:, 1]) - IMPACT_DAMAGE_THRESHOLD
        for m in np.flatnonzero(excess > 0).tolist():
            live[m].health -= IMPACT_DAMAGE * float(excess[m])
            live[m].last_impact_tick = tick
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
    moved = np.any(xy != xy0, axis=1)
    for m in np.flatnonzero(moved | np.any(knock != knock0, axis=1)).tolist():
        c = live[m]
        kx, ky = float(knock[m, 0]), float(knock[m, 1])
        if moved[m]:
            old_x, old_y = c.pos.x, c.pos.y
            want_x, want_y = float(xy[m, 0]), float(xy[m, 1])
            c.pos.x, c.pos.y = want_x, want_y
            c.clamp_to_walls()
            if c.pos.x != want_x:
                kx = 0.0
            if c.pos.y != want_y:
                ky = 0.0
            body = c.geometry.rect(c.pos.x - c.size, c.pos.y - c.size, c.size * 2, c.size * 2)
            if any(obs.rect.colliderect(body) for obs in obstacles):
                c.pos.x, c.pos.y = old_x, old_y
                kx = ky = 0.0
        c.knock.x, c.knock.y = kx, ky
