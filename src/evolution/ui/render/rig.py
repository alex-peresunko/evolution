"""Body rig for the sprite art. Lengths are in u (body radii) from the creature's origin; a
"station" is a distance along the straight spine, and a lateral offset is positive on the +y
side of a creature facing +x."""
import math
from bisect import bisect_right
from functools import cache, wraps

import pygame

from .shapes import ellipse_pts


def smooth_profile(ctrl):
    """Callable x -> value through `ctrl` ((x, value), x ascending), without overshoot. Outside
    the control range it returns the end values."""
    xs = [c[0] for c in ctrl]
    ys = [c[1] for c in ctrl]
    n = len(xs)
    secants = [(ys[i + 1] - ys[i]) / (xs[i + 1] - xs[i]) for i in range(n - 1)]
    slopes = [secants[0]]
    for a, b in zip(secants, secants[1:], strict=False):
        slopes.append(0.0 if a * b <= 0 else 2 * a * b / (a + b))
    slopes.append(secants[-1])

    @cache
    def value(x):
        if x <= xs[0]:
            return ys[0]
        if x >= xs[-1]:
            return ys[-1]
        i = bisect_right(xs, x) - 1
        h = xs[i + 1] - xs[i]
        t = (x - xs[i]) / h
        t2, t3 = t * t, t * t * t
        return ((2 * t3 - 3 * t2 + 1) * ys[i] + (t3 - 2 * t2 + t) * h * slopes[i]
                + (3 * t2 - 2 * t3) * ys[i + 1] + (t3 - t2) * h * slopes[i + 1])

    return value


def stations(x0, x1, step, extra=()):
    n = max(1, round((x1 - x0) / step))
    return tuple(sorted({round(x0 + (x1 - x0) * i / n, 4) for i in range(n + 1)} | set(extra)))


def _per_spine(shape):
    """Keep each outline with its spine: poses are cached and every sprite of a pose asks for
    the same shapes."""
    @wraps(shape)
    def cached(spine, *args, **kwargs):
        key = (shape, args, tuple(kwargs.items()))
        pts = spine.shapes.get(key)
        if pts is None:
            pts = spine.shapes[key] = shape(spine, *args, **kwargs)
        return pts
    return cached


class Spine:
    """The curve a pose gives the spine: maps (station, lateral offset) to a point. Treat it
    as immutable; `xs` must be a tuple."""

    def __init__(self, xs, pts):
        self.xs, self.pts = xs, pts
        self.shapes = {}
        self.angles = []
        last = len(xs) - 1
        for i in range(len(xs)):
            (x0, y0), (x1, y1) = pts[max(i - 1, 0)], pts[min(i + 1, last)]
            a = math.atan2(y1 - y0, x1 - x0)
            if self.angles:
                a += 2 * math.pi * round((self.angles[-1] - a) / (2 * math.pi))
            self.angles.append(a)

    @classmethod
    def swaying(cls, xs, sway):
        """A straight spine pushed sideways by sway(station)."""
        return cls(xs, [(x, sway(x)) for x in xs])

    @classmethod
    def bent(cls, xs, heading, origin=0.0):
        """A spine of unchanged length that points along heading(station), with the station
        nearest `origin` left in place."""
        i0 = min(range(len(xs)), key=lambda i: abs(xs[i] - origin))
        pts = [None] * len(xs)
        pts[i0] = (xs[i0], 0.0)
        for i in list(range(i0 + 1, len(xs))) + list(range(i0 - 1, -1, -1)):
            j = i - 1 if i > i0 else i + 1
            a, d = heading((xs[i] + xs[j]) / 2), xs[i] - xs[j]
            pts[i] = (pts[j][0] + math.cos(a) * d, pts[j][1] + math.sin(a) * d)
        return cls(xs, pts)

    def at(self, x):
        """(px, py, heading) at a station; stations past either end extrapolate straight on."""
        xs = self.xs
        i = min(max(bisect_right(xs, x) - 1, 0), len(xs) - 2)
        t = (x - xs[i]) / (xs[i + 1] - xs[i])
        (x0, y0), (x1, y1) = self.pts[i], self.pts[i + 1]
        tc = min(max(t, 0.0), 1.0)
        return (x0 + (x1 - x0) * t, y0 + (y1 - y0) * t,
                self.angles[i] + (self.angles[i + 1] - self.angles[i]) * tc)

    def place(self, x, lat=0.0):
        key = (x, lat)
        pt = self.shapes.get(key)
        if pt is None:
            px, py, a = self.at(x)
            pt = self.shapes[key] = (px - math.sin(a) * lat, py + math.cos(a) * lat)
        return pt


@_per_spine
def ribbon(spine, width, xs, k=1.0, grow=0.0):
    """Outline of the body between stations `xs`: width(station) * k + grow to each side."""
    left = [spine.place(x, width(x) * k + grow) for x in xs]
    right = [spine.place(x, -(width(x) * k + grow)) for x in xs]
    return left + right[::-1]


@_per_spine
def patch(spine, width, x, f, rx, rf, n=12):
    """Oval of skin centred at station x and at fraction f of the half-width; rx long, and rf
    (a fraction of the half-width) wide."""
    pts = []
    for i in range(n):
        a = 2 * math.pi * i / n
        sx = x + rx * math.cos(a)
        pts.append(spine.place(sx, (f + rf * math.sin(a)) * width(sx)))
    return pts


@_per_spine
def band(spine, width, x, half, reach=0.85, sweep=0.0, n=5):
    """Stripe across the back at station x: `half` thick at the spine, thinning toward the
    flanks (to fraction `reach` of the half-width) and swept back by `sweep`."""
    front, back = [], []
    for i in range(-n, n + 1):
        q = (i / n) ** 2
        xc, th = x - sweep * q, half * (1 - 0.85 * q)
        front.append(spine.place(xc + th, reach * i / n * width(xc + th)))
        back.append(spine.place(xc - th, reach * i / n * width(xc - th)))
    return front + back[::-1]


def knee(hip, foot, upper, lower, prefer):
    """The middle joint of a two-segment limb, on the side `prefer` (a direction) points to."""
    dx, dy = foot[0] - hip[0], foot[1] - hip[1]
    dist = math.hypot(dx, dy) or 1e-6
    d = min(max(dist, abs(upper - lower) + 1e-6), upper + lower)
    along = (upper * upper - lower * lower + d * d) / (2 * d)
    out = math.sqrt(max(0.0, upper * upper - along * along))
    ux, uy = dx / dist, dy / dist
    nx, ny = -uy, ux
    if nx * prefer[0] + ny * prefer[1] < 0:
        nx, ny = -nx, -ny
    return hip[0] + ux * along + nx * out, hip[1] + uy * along + ny * out


def turned(origin, angle, pts):
    """Local points (x forward) rotated by `angle` about `origin`."""
    ca, sa = math.cos(angle), math.sin(angle)
    return [(origin[0] + x * ca - y * sa, origin[1] + x * sa + y * ca) for x, y in pts]


class Pen:
    """Draws in u onto a supersampled surface."""

    def __init__(self, surf, cx, cy, u):
        self.surf, self.cx, self.cy, self.u = surf, cx, cy, u

    def _px(self, pts):
        cx, cy, u = self.cx, self.cy, self.u
        return [(cx + x * u, cy + y * u) for x, y in pts]

    def poly(self, color, pts):
        pygame.draw.polygon(self.surf, color, self._px(pts))

    def oval(self, color, pt, rx, ry, rot=0.0, n=20):
        self.poly(color, ellipse_pts(pt[0], pt[1], rx, ry, rot, n))

    def line(self, color, pts, width):
        pygame.draw.lines(self.surf, color, False, self._px(pts), max(1, int(width * self.u)))

    def capsule(self, color, p0, p1, r0, r1):
        """A limb segment: radius r0 at p0, tapering to r1 at p1."""
        dx, dy = p1[0] - p0[0], p1[1] - p0[1]
        d = math.hypot(dx, dy) or 1e-6
        nx, ny = -dy / d, dx / d
        self.poly(color, [(p0[0] + nx * r0, p0[1] + ny * r0), (p1[0] + nx * r1, p1[1] + ny * r1),
                          (p1[0] - nx * r1, p1[1] - ny * r1), (p0[0] - nx * r0, p0[1] - ny * r0)])
        self.oval(color, p0, r0, r0, n=14)
        self.oval(color, p1, r1, r1, n=14)

    def layer(self):
        """A transparent sheet of the same size, for translucent paint; lay it with `stamp`."""
        return Pen(pygame.Surface(self.surf.get_size(), pygame.SRCALPHA), self.cx, self.cy, self.u)

    def stamp(self, layer, mask=None):
        """Blend `layer` over this surface, keeping only what lies inside `mask`'s paint."""
        if mask is not None:
            layer.surf.blit(mask.surf, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
        self.surf.blit(layer.surf, (0, 0))
