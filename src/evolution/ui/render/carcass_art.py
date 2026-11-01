"""Carcass sprites: the dead creature's skeleton seen from above, facing +x, with the flesh
still on it; u = body radius in supersampled pixels."""
import math
import random

import pygame

from .creature_art import head_outline
from .shapes import blur, ellipse, ellipse_pts
from .style import (
    CARCASS_LEVELS,
    CARNIVORE_PALETTE,
    HERBIVORE_PALETTE,
    SS,
    faded_palette,
    mix,
    tinted_palette,
)

BONE = (230, 222, 200)
BONE_EDGE = (110, 98, 80)
SOCKET = (78, 64, 50)
TOOTH = (250, 248, 236)

# Spine control points from the tail tip to the base of the skull: (x, y, vertebra half-width).
# The neck is thrown back, as in most fossil skeletons.
_HERBIVORE_SPINE = ((-1.9, 0.45, 0.02), (-1.5, 0.2, 0.035), (-1.1, 0.05, 0.055),
                    (-0.75, 0.0, 0.07), (0.5, 0.0, 0.075), (0.9, -0.1, 0.06),
                    (1.3, -0.35, 0.05), (1.55, -0.7, 0.045))
_CARNIVORE_SPINE = ((-2.3, 0.5, 0.02), (-1.7, 0.25, 0.04), (-1.0, 0.05, 0.065),
                    (-0.5, 0.0, 0.08), (0.45, 0.0, 0.085), (0.68, -0.08, 0.08),
                    (0.84, -0.16, 0.075))
# Torso: (centre x, half length, half width), matching the living sprite's body.
_HERBIVORE_TORSO = (-0.12, 1.0, 0.56)
_CARNIVORE_TORSO = (-0.1, 0.88, 0.5)


def _bone(s, u, x, y, rx, ry, rot=0.0):
    e = 0.025 * u
    ellipse(s, BONE_EDGE, x, y, rx + e, ry + e, rot)
    ellipse(s, BONE, x, y, rx, ry, rot)


def _rod(s, u, pts, width, knobs=True):
    """A long bone through `pts` (in px), `width` in u, with knobbed ends."""
    w = width * u
    pygame.draw.lines(s, BONE_EDGE, False, pts, max(1, int(w + 0.05 * u)))
    if knobs:
        for x, y in (pts[0], pts[-1]):
            pygame.draw.circle(s, BONE_EDGE, (x, y), w * 0.75 + 0.025 * u)
    pygame.draw.lines(s, BONE, False, pts, max(1, int(w)))
    if knobs:
        for x, y in (pts[0], pts[-1]):
            pygame.draw.circle(s, BONE, (x, y), w * 0.75)


def _spine_samples(ctrl, step):
    """(x, y, half-width, direction) every `step` along the control polyline, in u."""
    out = []
    for (x0, y0, w0), (x1, y1, w1) in zip(ctrl, ctrl[1:], strict=False):
        seg = math.hypot(x1 - x0, y1 - y0)
        a = math.atan2(y1 - y0, x1 - x0)
        n = max(1, round(seg / step))
        for k in range(n):
            t = k / n
            out.append((x0 + (x1 - x0) * t, y0 + (y1 - y0) * t, w0 + (w1 - w0) * t, a))
    x, y, w = ctrl[-1]
    out.append((x, y, w, out[-1][3]))
    return out


def _place(points, cx, cy, u, ax, ay, a):
    """Local skull points (x forward from the neck, in u) to sprite pixels."""
    ca, sa = math.cos(a), math.sin(a)
    return [(cx + (ax + x * ca - y * sa) * u, cy + (ay + x * sa + y * ca) * u) for x, y in points]


def _carnivore_skull(s, cx, cy, u, ax, ay, a):
    outline = [((px - cx) / u - 0.84, (py - cy) / u) for px, py in head_outline(cx, cy, u)]
    inner = [((px - cx) / u - 0.84, (py - cy) / u) for px, py in head_outline(cx, cy, u, 0.9)]
    pygame.draw.polygon(s, BONE_EDGE, _place(outline, cx, cy, u, ax, ay, a))
    pygame.draw.polygon(s, BONE, _place(inner, cx, cy, u, ax, ay, a))
    for side in (1, -1):
        for x, y in ((0.41, 0.39), (0.54, 0.36), (0.66, 0.32), (0.78, 0.275), (0.9, 0.22)):
            pygame.draw.polygon(s, TOOTH, _place(
                [(x - 0.04, side * (y - 0.03)), (x + 0.04, side * (y - 0.03)),
                 (x + 0.01, side * (y + 0.05))], cx, cy, u, ax, ay, a))
        for (x, y), (rx, ry), rot in (((0.26, 0.24), (0.075, 0.055), 0.3),    # orbit
                                      ((0.55, 0.19), (0.11, 0.04), 0.15),     # antorbital opening
                                      ((0.06, 0.18), (0.06, 0.04), -0.4),     # temporal opening
                                      ((0.93, 0.07), (0.035, 0.02), 0.0)):    # nostril
            pts = ellipse_pts(x, side * y, rx, ry, side * rot, n=16)
            pygame.draw.polygon(s, SOCKET, _place(pts, cx, cy, u, ax, ay, a))


def _herbivore_skull(s, cx, cy, u, ax, ay, a):
    pygame.draw.polygon(s, BONE_EDGE, _place(ellipse_pts(0.22, 0, 0.27, 0.17), cx, cy, u, ax, ay, a))
    pygame.draw.polygon(s, BONE, _place(ellipse_pts(0.22, 0, 0.24, 0.14), cx, cy, u, ax, ay, a))
    for side in (1, -1):
        for (x, y), (rx, ry) in (((0.16, 0.085), (0.045, 0.032)), ((0.38, 0.045), (0.025, 0.016))):
            pts = ellipse_pts(x, side * y, rx, ry, n=14)
            pygame.draw.polygon(s, SOCKET, _place(pts, cx, cy, u, ax, ay, a))


def _ribs(s, cx, cy, u, torso, x_from, x_to, pairs):
    bx, rx, ry = torso
    for k in range(pairs):
        x = x_from + (x_to - x_from) * k / (pairs - 1)
        half = ry * math.sqrt(max(0.0, 1 - ((x - bx) / rx) ** 2))
        for side in (1, -1):
            pts = [(cx + (x - 0.22 * t * t) * u, cy + side * (0.06 + half * 0.92 * math.sin(
                t * math.pi / 2)) * u) for t in (0.0, 0.25, 0.5, 0.75, 1.0)]
            _rod(s, u, pts, 0.045, knobs=False)


def _limb(s, cx, cy, u, joints, width, toes, toe_len, side):
    """A splayed leg: `joints` in u (hip first, the foot last, y for side +1)."""
    pts = [(cx + x * u, cy + side * y * u) for x, y in joints]
    for p, q in zip(pts, pts[1:], strict=False):
        _rod(s, u, [p, q], width)
    fx, fy = joints[-1]
    for d in toes:
        tx, ty = fx + toe_len * math.cos(d), side * (fy + toe_len * math.sin(d))
        _rod(s, u, [(cx + fx * u, cy + side * fy * u), (cx + tx * u, cy + ty * u)],
             width * 0.45, knobs=False)
        pygame.draw.circle(s, BONE_EDGE, (cx + tx * u, cy + ty * u), width * 0.35 * u)


def _skeleton(s, cx, cy, u, carn):
    spine = _CARNIVORE_SPINE if carn else _HERBIVORE_SPINE
    torso = _CARNIVORE_TORSO if carn else _HERBIVORE_TORSO
    if carn:
        _limb(s, cx, cy, u, ((-0.2, 0.18), (-0.05, 0.6), (0.2, 0.78)), 0.09,
              (-0.45, 0.0, 0.45), 0.22, 1)
        _limb(s, cx, cy, u, ((-0.2, 0.18), (-0.05, 0.6), (0.2, 0.78)), 0.09,
              (-0.45, 0.0, 0.45), 0.22, -1)
        for side in (1, -1):
            _limb(s, cx, cy, u, ((0.5, 0.2), (0.62, 0.36), (0.74, 0.36)), 0.035,
                  (-0.3, 0.3), 0.08, side)
    else:
        for x in (0.45, -0.6):
            for side in (1, -1):
                _limb(s, cx, cy, u, ((x, 0.18), (x + 0.05, 0.5), (x + 0.12, 0.72)), 0.075,
                      (-0.6, -0.1, 0.4, 0.9), 0.13, side)
    hip = spine[3][0]
    for side in (1, -1):
        _bone(s, u, cx + (hip + 0.05) * u, cy + side * 0.17 * u, 0.22 * u, 0.09 * u, side * 0.35)
        _bone(s, u, cx + 0.32 * u, cy + side * 0.24 * u, 0.17 * u, 0.06 * u, -side * 0.6)
    _ribs(s, cx, cy, u, torso, 0.35 if carn else 0.4, -0.45 if carn else -0.55, 8)
    samples = _spine_samples(spine, 0.11)
    for x, y, w, a in samples:
        px, py = cx + x * u, cy + y * u
        _bone(s, u, px, py, 0.02 * u, w * 1.7 * u, a)
        _bone(s, u, px, py, 0.05 * u, w * 0.75 * u, a)
    x, y, _, a = samples[-1]
    (_carnivore_skull if carn else _herbivore_skull)(s, cx, cy, u, x, y, a)


def _flesh(s, cx, cy, u, carn, meat, skin):
    """What's left of the body under the bones; `meat` 1 is untouched, 0 bare bones."""
    bx, rx, ry = _CARNIVORE_TORSO if carn else _HERBIVORE_TORSO
    k = 0.5 + 0.55 * meat
    if meat > 0.5:
        ellipse(s, skin["rim"], cx + bx * u, cy, rx * k * u, ry * 1.15 * k * u)
        ellipse(s, skin["core"], cx + bx * u, cy, rx * k * 0.92 * u, ry * 1.15 * k * 0.88 * u)
    raw_rim, raw = mix((70, 24, 22), (112, 18, 22), meat), mix((128, 66, 52), (176, 38, 40), meat)
    for i in range(5):
        t = i / 4
        ellipse(s, mix(raw_rim, raw, t), cx + bx * u, cy - 0.04 * t * u,
                rx * k * (0.9 - 0.3 * t) * u, ry * k * (0.95 - 0.3 * t) * u)


def _flesh_over_ribs(s, cx, cy, u, carn, meat):
    """Meat still covering the middle of the rib cage."""
    bx, rx, ry = _CARNIVORE_TORSO if carn else _HERBIVORE_TORSO
    a = int(200 * meat ** 1.3)
    for i in range(4):
        t = i / 3
        color = mix((96, 16, 20), (184, 46, 46), t) + (a,)
        ellipse(s, color, cx + (bx - 0.05) * u, cy - 0.03 * t * u,
                rx * (0.25 + 0.4 * meat) * (1 - 0.35 * t) * u,
                ry * (0.25 + 0.35 * meat) * (1 - 0.35 * t) * u)


def build_carcass(carn, sb, level, cb):
    """Unrotated carcass sprite facing +x with its blood stain, the shadow of its bones and
    flesh, and the x offset of the sprite centre from the carcass position. `level` 0 is a
    fresh kill, CARCASS_LEVELS - 1 bare bones."""
    meat = 1 - level / (CARCASS_LEVELS - 1)
    x0, x1, y_half = (-2.45, 2.0, 1.05) if carn else (-2.05, 2.0, 1.1)
    w = math.ceil((x1 - x0) * sb) + 2
    h = math.ceil(2 * y_half * sb) + 2
    ox = 1 - x0 * sb
    cx, cy, u = ox * SS, h * SS / 2, sb * SS
    rnd = random.Random(sb * 31 + carn * 7 + level)

    stain = pygame.Surface((w * SS, h * SS), pygame.SRCALPHA)
    blood = mix((70, 52, 34), (110, 12, 14), meat) + (int(110 + 110 * meat),)
    torso = _CARNIVORE_TORSO if carn else _HERBIVORE_TORSO
    for _ in range(7):
        ellipse(stain, blood, cx + (torso[0] + rnd.uniform(-0.6, 0.5)) * u,
                cy + rnd.uniform(-0.35, 0.35) * u, rnd.uniform(0.3, 0.6) * (0.6 + 0.4 * meat) * u,
                rnd.uniform(0.2, 0.4) * (0.6 + 0.4 * meat) * u, rnd.uniform(0, math.pi))

    body = pygame.Surface((w * SS, h * SS), pygame.SRCALPHA)
    if meat > 0:
        skin = faded_palette(tinted_palette(CARNIVORE_PALETTE if carn else HERBIVORE_PALETTE, cb), 0)
        _flesh(body, cx, cy, u, carn, meat, skin)
    _skeleton(body, cx, cy, u, carn)
    if meat > 0:
        over = pygame.Surface((w * SS, h * SS), pygame.SRCALPHA)
        _flesh_over_ribs(over, cx, cy, u, carn, meat)
        body.blit(over, (0, 0))

    sprite = pygame.transform.smoothscale(blur(stain, 6), (w, h))
    body = pygame.transform.smoothscale(body, (w, h))
    sprite.blit(body, (0, 0))
    shadow = blur(body, 2)
    shadow.fill((0, 0, 0), special_flags=pygame.BLEND_RGB_MULT)
    shadow.fill((255, 255, 255, 70), special_flags=pygame.BLEND_RGBA_MULT)
    return sprite, shadow, w / 2 - ox
