"""Slow-motion smoothing: below one tick per frame, creatures are drawn part way through the last
tick instead of jumping once per tick."""
import math


class MotionSmoother:
    def __init__(self):
        self._before = {}      # creature id -> (x, y, angle) before the last tick
        self._after_tick = None

    def before_tick(self, world):
        self._before = {c.id: (c.pos.x, c.pos.y, c.angle) for c in world.creatures()}
        self._after_tick = world.tick + 1

    def poses(self, world, progress):
        """{creature id: (x, y, angle)} `progress` (0..1) of the way through the last tick, or
        None when no snapshot was taken just before it. Creatures born in it are left out."""
        if world.tick != self._after_tick:
            return None
        poses = {}
        for c in world.creatures():
            before = self._before.get(c.id)
            if before is None:
                continue
            x0, y0, a0 = before
            turn = (c.angle - a0 + math.pi) % (2 * math.pi) - math.pi
            poses[c.id] = (x0 + (c.pos.x - x0) * progress, y0 + (c.pos.y - y0) * progress,
                           a0 + turn * progress)
        return poses
