"""Pygame renderer: procedurally drawn, pre-rotated, cached sprites.

Only game.py imports this module, so the simulation library stays pygame-free.
Every sprite is drawn once at SS× resolution and smoothscaled down (anti-aliasing),
then cached by quantized (on-screen size, health, walk frame, colour gene, angle); a frame is
mostly blits. Zooming scales positions and sprite sizes, so zoomed sprites are drawn natively.
"""
import colorsys
import math
import random
import time
from itertools import islice

import numpy as np
import pygame
import pygame.gfxdraw

from . import config
from .entities import Tree

SS = 4  # supersampling factor for sprite art
FOOD_DRAW_RADIUS = 8
FOOD_VARIANTS = 6
FOOD_LEVELS = 8
HEALTH_LEVELS = 4
WALK_FRAMES = 4
PULSE_LEVELS = 4
CARCASS_LEVELS = 4
HIT_FLASH_TICKS = 8
CROWN_GHOST_ALPHA = 120  # creatures under a crown show through it this strongly
BG_CACHE_MARGIN = 0.25  # share of the view cached around it on each side when zoomed
NIGHT_COLOR = (8, 12, 42)

HERBIVORE_PALETTE = {
    "rim": (36, 88, 70), "core": (132, 196, 136), "leg": (28, 70, 56), "spot": (168, 216, 156),
    "stripe": (58, 120, 92), "crest": (150, 210, 150), "claw": (226, 220, 200),
    "eye": (255, 255, 255), "pupil": (15, 15, 20), "nose": (24, 52, 42),
}
CARNIVORE_PALETTE = {
    "rim": (92, 44, 22), "core": (206, 128, 58), "leg": (70, 34, 18), "stripe": (54, 26, 14),
    "claw": (236, 226, 210), "snout": (226, 168, 106), "eye": (255, 212, 60),
    "pupil": (10, 10, 10), "fang": (250, 250, 240), "nose": (30, 10, 10),
}
_NO_FADE = {"eye", "pupil", "fang", "claw"}


def _mix(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def _ellipse_pts(cx, cy, rx, ry, rot=0.0, n=28):
    cr, sr = math.cos(rot), math.sin(rot)
    pts = []
    for i in range(n):
        a = 2 * math.pi * i / n
        x, y = rx * math.cos(a), ry * math.sin(a)
        pts.append((cx + x * cr - y * sr, cy + x * sr + y * cr))
    return pts


def _ellipse(surf, color, cx, cy, rx, ry, rot=0.0):
    pygame.draw.polygon(surf, color, _ellipse_pts(cx, cy, rx, ry, rot))


def _shaded_ellipse(surf, rim, core, cx, cy, rx, ry, rot=0.0, steps=6, hx=0.0, hy=0.0):
    """Ellipse shaded from its rim colour to a highlight core shifted by (hx, hy)."""
    for i in range(steps):
        t = i / (steps - 1)
        k = 1 - 0.55 * t
        _ellipse(surf, _mix(rim, core, t), cx + hx * t, cy + hy * t, rx * k, ry * k, rot)


def _faded_palette(palette, health_level):
    t = (HEALTH_LEVELS - 1 - health_level) * 0.2
    out = {}
    for k, c in palette.items():
        if k in _NO_FADE:
            out[k] = c
        else:
            lum = int(0.3 * c[0] + 0.59 * c[1] + 0.11 * c[2])
            out[k] = _mix(c, (lum, lum, int(lum * 0.95)), t)
    return out


def color_bucket(color_gene):
    return min(config.COLOR_BUCKETS - 1, int(color_gene * config.COLOR_BUCKETS))


def _tinted_palette(palette, bucket):
    """Hue-rotate the palette by the colour gene's bucket (the middle bucket is ~unchanged);
    eyes, pupils, fangs and claws keep their colours."""
    shift = ((bucket + 0.5) / config.COLOR_BUCKETS - 0.5) * config.COLOR_HUE_RANGE / 360
    out = {}
    for k, c in palette.items():
        if k in _NO_FADE:
            out[k] = c
            continue
        h, s, v = colorsys.rgb_to_hsv(*(x / 255 for x in c))
        out[k] = tuple(round(x * 255) for x in colorsys.hsv_to_rgb((h + shift) % 1, s, v))
    return out


# =============================================================================
# Creature art (facing +x; u = body radius in supersampled pixels)
# =============================================================================
def _draw_herbivore(s, cx, cy, u, p, frame):
    ph = frame * math.pi / 2
    sway = math.sin(ph + 1.0)
    n = 18
    for i in range(n):
        t = i / (n - 1)
        r = 0.26 * (1 - t) ** 1.2 + 0.03
        _ellipse(s, _mix(p["rim"], p["core"], 0.4 * (1 - t)),
                 cx + (-0.75 - 1.1 * t) * u, cy + sway * 0.3 * t * t * u, r * u, r * u)
    swing = math.sin(ph) * 0.16 * u
    for fx, side, sgn in ((0.45, 1, 1), (0.45, -1, -1), (-0.6, 1, -1), (-0.6, -1, 1)):
        lx, ly = cx + fx * u + sgn * swing, cy + side * 0.5 * u
        _shaded_ellipse(s, p["leg"], p["core"], lx, ly, 0.21 * u, 0.19 * u, steps=4,
                        hy=-side * 0.03 * u)
        for d in (-0.09, 0.0, 0.09):
            _ellipse(s, p["claw"], lx + 0.19 * u, ly + d * u, 0.04 * u, 0.035 * u)
    _shaded_ellipse(s, p["rim"], p["core"], cx - 0.12 * u, cy, 1.0 * u, 0.56 * u, steps=12,
                    hx=0.1 * u)
    spot = _mix(p["core"], p["spot"], 0.6)
    for sx, sy, r in ((-0.5, -0.26, 0.08), (-0.15, 0.3, 0.07), (-0.72, 0.14, 0.06),
                      (0.18, -0.26, 0.06), (0.32, 0.2, 0.05), (-0.32, 0.08, 0.05)):
        _ellipse(s, spot, cx + sx * u, cy + sy * u, r * u, r * u * 0.8)
    for i in range(7):
        _ellipse(s, p["stripe"], cx + (-0.85 + i * 0.25) * u, cy, 0.07 * u, 0.04 * u)
    neck_sway = math.sin(ph + 2.5) * 0.12
    n = 16
    for i in range(n):
        t = i / (n - 1)
        r = 0.3 - 0.17 * t ** 0.7
        _ellipse(s, _mix(p["rim"], p["core"], 0.55 + 0.25 * t),
                 cx + (0.55 + 1.3 * t) * u, cy + neck_sway * t * t * u, r * u, r * u)
    hx, hy = cx + 2.02 * u, cy + neck_sway * u
    _shaded_ellipse(s, p["rim"], p["core"], hx, hy, 0.26 * u, 0.18 * u, steps=5, hx=0.03 * u)
    _ellipse(s, p["crest"], hx - 0.04 * u, hy, 0.1 * u, 0.07 * u)
    for side in (1, -1):
        ey = hy + side * 0.13 * u
        _ellipse(s, p["eye"], hx - 0.02 * u, ey, 0.06 * u, 0.05 * u)
        _ellipse(s, p["pupil"], hx + 0.0 * u, ey, 0.035 * u, 0.035 * u)
        _ellipse(s, p["nose"], hx + 0.2 * u, hy + side * 0.05 * u, 0.025 * u, 0.02 * u)


def _head_outline(cx, cy, u, k=1.0, dx=0.0):
    """Top-down T-Rex skull outline, shrunk by k toward its centre and shifted by dx (in u)."""
    half = ((0.84, 0.32), (0.98, 0.43), (1.22, 0.41), (1.5, 0.32), (1.73, 0.23), (1.85, 0.12),
            (1.88, 0.0))
    mid = 1.36
    pts = [(mid + (x - mid) * k + dx, -y * k) for x, y in half]
    pts += [(x, -y) for x, y in reversed(pts[:-1])]
    return [(cx + x * u, cy + y * u) for x, y in pts]


def _draw_carnivore(s, cx, cy, u, p, frame):
    ph = frame * math.pi / 2
    sway = math.sin(ph + 1.0)
    n = 26
    for i in range(n):
        t = i / (n - 1)
        r = 0.38 * (1 - t) ** 1.2 + 0.03
        _ellipse(s, _mix(p["rim"], p["core"], 0.35 * (1 - t)),
                 cx + (-0.5 - 1.75 * t) * u, cy + sway * 0.38 * t * t * u, r * u, r * u)
    swing = math.sin(ph) * 0.24 * u
    for side in (1, -1):
        lx, ly = cx - 0.12 * u + side * swing, cy + side * 0.5 * u
        fx, fy = lx + 0.34 * u, ly + side * 0.1 * u
        _ellipse(s, p["leg"], fx, fy, 0.16 * u, 0.11 * u)
        for a in (-0.55, 0.0, 0.55):
            _ellipse(s, p["claw"], fx + 0.16 * u * math.cos(a), fy + 0.13 * u * math.sin(a),
                     0.055 * u, 0.035 * u, rot=a)
        _shaded_ellipse(s, p["leg"], p["core"], lx, ly, 0.4 * u, 0.22 * u, steps=5,
                        hx=0.05 * u, hy=-side * 0.04 * u)
    _shaded_ellipse(s, p["rim"], p["core"], cx - 0.1 * u, cy, 0.88 * u, 0.5 * u, steps=7,
                    hx=0.08 * u)
    _shaded_ellipse(s, p["rim"], p["core"], cx + 0.68 * u, cy, 0.4 * u, 0.34 * u, steps=5)
    for sx, sy, r in ((-0.5, 0.22, 0.07), (-0.2, -0.27, 0.06), (0.15, 0.24, 0.05),
                      (-0.75, -0.12, 0.05), (0.4, -0.18, 0.045)):
        _ellipse(s, _mix(p["rim"], p["core"], 0.4), cx + sx * u, cy + sy * u, r * u, r * u)
    for i in range(9):
        x = -1.3 + i * 0.2
        k = 1 - abs(x + 0.4) * 0.35
        _ellipse(s, _mix(p["stripe"], p["rim"], 0.3), cx + x * u, cy, 0.08 * u * k, 0.05 * u * k)
    for side in (1, -1):
        ax, ay = cx + 0.6 * u, cy + side * 0.36 * u
        _ellipse(s, p["leg"], ax, ay, 0.17 * u, 0.075 * u, rot=side * 0.5)
        _ellipse(s, p["core"], ax - 0.01 * u, ay - side * 0.01 * u, 0.12 * u, 0.04 * u,
                 rot=side * 0.5)
        for d in (-0.04, 0.04):
            _ellipse(s, p["claw"], ax + 0.15 * u, ay + (side * 0.09 + d) * u, 0.035 * u, 0.028 * u)
    pygame.draw.polygon(s, p["rim"], _head_outline(cx, cy, u))
    pygame.draw.polygon(s, _mix(p["rim"], p["core"], 0.6), _head_outline(cx, cy, u, 0.86))
    pygame.draw.polygon(s, p["core"], _head_outline(cx, cy, u, 0.64, -0.03))
    _ellipse(s, p["snout"], cx + 1.58 * u, cy, 0.24 * u, 0.15 * u)
    for side in (1, -1):
        for x, y in ((1.25, 0.39), (1.38, 0.36), (1.5, 0.32), (1.62, 0.275), (1.74, 0.22)):
            pygame.draw.polygon(s, p["fang"], [
                (cx + (x - 0.045) * u, cy + side * (y - 0.03) * u),
                (cx + (x + 0.045) * u, cy + side * (y - 0.03) * u),
                (cx + (x + 0.01) * u, cy + side * (y + 0.06) * u),
            ])
        ey = cy + side * 0.27 * u
        _ellipse(s, p["stripe"], cx + 1.1 * u, cy + side * 0.21 * u, 0.16 * u, 0.06 * u,
                 rot=side * 0.3)
        _ellipse(s, p["eye"], cx + 1.08 * u, ey, 0.1 * u, 0.06 * u, rot=side * 0.35)
        _ellipse(s, p["pupil"], cx + 1.1 * u, ey, 0.022 * u, 0.055 * u, rot=side * 0.35)
        _ellipse(s, p["nose"], cx + 1.79 * u, cy + side * 0.07 * u, 0.035 * u, 0.025 * u)


# =============================================================================
# Background
# =============================================================================
def _blur(surf, factor):
    w, h = surf.get_size()
    small = pygame.transform.smoothscale(surf, (max(1, w // factor), max(1, h // factor)))
    return pygame.transform.smoothscale(small, (w, h))


def _value_noise(rng, w, h, cell):
    """Smooth noise in [0, 1] with features roughly `cell` pixels across, shape (w, h)."""
    small = (rng.random((max(2, w // cell), max(2, h // cell))) * 255).astype(np.uint8)
    surf = pygame.transform.smoothscale(pygame.surfarray.make_surface(np.dstack([small] * 3)), (w, h))
    return pygame.surfarray.array_red(surf).astype(np.float32) / 255.0


def _draw_rock(bg, rect, rng, rnd):
    """Bevelled, lit boulder filling `rect`, from a rounded-box distance field."""
    x, y, w, h = int(rect.x), int(rect.y), int(rect.width), int(rect.height)
    m = min(w, h)
    rad = m * 0.3

    pad = 18
    shadow = pygame.Surface((w + pad * 2, h + pad * 2), pygame.SRCALPHA)
    pygame.draw.rect(shadow, (0, 0, 0, 150), (pad, pad, w, h), border_radius=int(rad))
    bg.blit(_blur(shadow, 7), (x - pad + 9, y - pad + 11))

    px = (np.arange(w, dtype=np.float32) + 0.5 - w / 2)[:, None]
    py = (np.arange(h, dtype=np.float32) + 0.5 - h / 2)[None, :]
    qx = np.abs(px) - (w / 2 - rad - 1)
    qy = np.abs(py) - (h / 2 - rad - 1)
    d = (np.sqrt(np.maximum(qx, 0) ** 2 + np.maximum(qy, 0) ** 2)
         + np.minimum(np.maximum(qx, qy), 0) - rad)
    d += (_value_noise(rng, w, h, 14) - 0.5) * m * 0.07

    bevel = m * 0.24
    height = np.clip(-d / bevel, 0, 1)
    height = 1 - (1 - height) ** 2
    mottle = _value_noise(rng, w, h, 9)
    height = height + 0.05 * mottle + 0.025 * _value_noise(rng, w, h, 3)
    gx, gy = np.gradient(height * bevel * 0.8)
    nz = 1 / np.sqrt(gx ** 2 + gy ** 2 + 1)
    light = np.array([-0.5, -0.6, 0.62], np.float32)
    light /= np.linalg.norm(light)
    lam = np.clip((-gx * light[0] - gy * light[1] + light[2]) * nz, 0, 1)

    tone = rnd.uniform(-10, 10)
    base = np.array([104 + tone, 99 + tone, 92 + tone], np.float32)
    shade = (0.28 + 0.95 * lam) * (0.82 + 0.36 * mottle)
    col = base[None, None, :] * shade[..., None]
    moss = np.clip((_value_noise(rng, w, h, 16) - 0.6) * 5, 0, 1) * (height > 0.6)
    col = col * (1 - moss[..., None]) + (np.array([62, 104, 50], np.float32)[None, None, :]
                                         * (0.45 + 0.8 * lam)[..., None] * moss[..., None])

    rock = pygame.Surface((w, h), pygame.SRCALPHA)
    pygame.surfarray.pixels3d(rock)[:] = np.clip(col, 0, 255).astype(np.uint8)
    pygame.surfarray.pixels_alpha(rock)[:] = (np.clip(-d + 0.5, 0, 1) * 255).astype(np.uint8)

    margin = rad * 0.6 + 4
    for _ in range(rnd.randint(1, 3)):
        cx, cy = rnd.uniform(margin, w - margin), rnd.uniform(margin, h - margin)
        a = rnd.uniform(0, 2 * math.pi)
        pts = [(cx, cy)]
        for _ in range(rnd.randint(3, 5)):
            a += rnd.uniform(-0.8, 0.8)
            seg = m * rnd.uniform(0.05, 0.1)
            cx = min(w - margin, max(margin, cx + math.cos(a) * seg))
            cy = min(h - margin, max(margin, cy + math.sin(a) * seg))
            pts.append((cx, cy))
        pygame.draw.aalines(rock, (150, 144, 134), False, [(qx + 1, qy + 1) for qx, qy in pts])
        pygame.draw.aalines(rock, (38, 36, 34), False, pts)
    bg.blit(rock, (x, y))


def _draw_tree_ground(bg, tree, rnd):
    """Crown shadow, roots and trunk under a tree; the crown itself is drawn over creatures."""
    r = tree.crown_radius
    pad = int(r * 0.4) + 8
    w = int(r * 2) + pad * 2
    shadow = pygame.Surface((w, w), pygame.SRCALPHA)
    pygame.draw.circle(shadow, (0, 0, 0, 120), (w / 2, w / 2), r * 0.95)
    bg.blit(_blur(shadow, 6), (tree.x - w / 2 + r * 0.22, tree.y - w / 2 + r * 0.28))

    t = tree.rect.w / 2
    bark, dark = (92, 66, 42), (44, 30, 20)
    for _ in range(rnd.randint(4, 6)):
        a = rnd.uniform(0, 2 * math.pi)
        ln = t * rnd.uniform(1.6, 2.4)
        end = (tree.x + math.cos(a) * ln, tree.y + math.sin(a) * ln)
        pygame.draw.line(bg, dark, (tree.x, tree.y), end, max(2, int(t * 0.55)))
        pygame.draw.line(bg, bark, (tree.x - 0.5, tree.y - 0.5), end, max(1, int(t * 0.3)))
    big = pygame.Surface((int(t * 2 + 4) * SS,) * 2, pygame.SRCALPHA)
    c = big.get_width() / 2
    _shaded_ellipse(big, dark, (132, 98, 64), c, c, t * SS, t * SS, steps=6,
                    hx=-0.3 * t * SS, hy=-0.35 * t * SS)
    for k in range(3):
        pygame.draw.circle(big, _mix(bark, dark, 0.5), (c, c), t * SS * (0.25 + 0.2 * k), width=SS)
    small = pygame.transform.smoothscale(big, (big.get_width() // SS,) * 2)
    bg.blit(small, (tree.x - small.get_width() / 2, tree.y - small.get_height() / 2))


def _build_crown(r, seed):
    """Leafy crown of radius `r` px, lit as a height field of overlapping leaf clumps with
    leaf-scale noise; returns (sprite, half width)."""
    rnd = random.Random(seed)
    rng = np.random.default_rng(seed)
    w = math.ceil(r * 2.3) + 4
    c = w / 2
    clumps = [(rnd.uniform(-0.08, 0.08), rnd.uniform(-0.08, 0.08), 0.5)]
    n = rnd.randint(7, 10)
    off = rnd.uniform(0, 2 * math.pi)
    for i in range(n):
        a = off + i * 2 * math.pi / n + rnd.uniform(-0.25, 0.25)
        d = rnd.uniform(0.52, 0.66)
        clumps.append((math.cos(a) * d, math.sin(a) * d, rnd.uniform(0.32, 0.44)))
    for i in range(4):
        a = off + i * math.pi / 2 + 0.7
        clumps.append((math.cos(a) * 0.3, math.sin(a) * 0.3, rnd.uniform(0.34, 0.42)))

    px = (np.arange(w, dtype=np.float32) + 0.5 - c)[:, None] / r
    py = (np.arange(w, dtype=np.float32) + 0.5 - c)[None, :] / r
    height = np.full((w, w), -1.0, np.float32)
    for x, y, cr in clumps:
        dome = cr * cr - (px - x) ** 2 - (py - y) ** 2
        lift = 0.35 * (1 - math.hypot(x, y))
        height = np.maximum(height, np.sqrt(np.maximum(dome, 0)) + lift - (dome <= 0) * 2)
    edge = height + 1.0                     # > 0 inside the crown
    leaves = _value_noise(rng, w, w, max(2, int(r * 0.09)))
    twigs = _value_noise(rng, w, w, max(2, int(r * 0.035)))
    bump = height + 0.05 * leaves + 0.022 * twigs
    gx, gy = np.gradient(bump * r * 0.9)
    nz = 1 / np.sqrt(gx ** 2 + gy ** 2 + 1)
    light = np.array([-0.5, -0.6, 0.62], np.float32)
    light /= np.linalg.norm(light)
    lam = np.clip((-gx * light[0] - gy * light[1] + light[2]) * nz, 0, 1)
    occl = np.clip(height / 0.55, 0, 1) ** 0.6

    tone = rnd.uniform(-1, 1)
    dark = np.array(_mix((16, 48, 24), (40, 56, 16), 0.5 + 0.35 * tone), np.float32)
    lit = np.array(_mix((96, 164, 66), (156, 176, 60), 0.4 + 0.35 * tone), np.float32)
    t = np.clip(lam * (0.55 + 0.45 * occl) * (0.8 + 0.4 * leaves), 0, 1)[..., None] ** 1.2
    col = dark * (1 - t) + lit * t
    sparkle = np.clip((twigs - 0.78) * 6, 0, 1) * np.clip((lam - 0.6) * 3, 0, 1)
    col += sparkle[..., None] * np.array([60, 60, 30], np.float32)

    crown = pygame.Surface((w, w), pygame.SRCALPHA)
    pygame.surfarray.pixels3d(crown)[:] = np.clip(col, 0, 255).astype(np.uint8)
    alpha = np.clip(edge * r * 0.6, 0, 1) * config.TREE_CROWN_ALPHA
    pygame.surfarray.pixels_alpha(crown)[:] = alpha.astype(np.uint8)
    return crown.convert_alpha(), c


LAKE_SHORE_WIDTH = 9
LAKE_MUD = (58, 52, 38)
LAKE_SHALLOW = (54, 104, 108)
LAKE_DEEP = (18, 46, 74)


def _blur_array(a, factor):
    """Soft blur of a [0, 1] (w, h) array by smoothscaling down `factor` times and back."""
    w, h = a.shape
    surf = pygame.surfarray.make_surface(np.dstack([(a * 255).astype(np.uint8)] * 3))
    small = pygame.transform.smoothscale(surf, (max(1, w // factor), max(1, h // factor)))
    return pygame.surfarray.array_red(pygame.transform.smoothscale(small, (w, h))).astype(np.float32) / 255


def _paint_lakes(col, lakes, ripple):
    """Blend muddy shores and depth-shaded water into the (w, h, 3) background colours."""
    w, h = col.shape[:2]
    edge = np.full((w, h), np.inf, np.float32)   # gap to the nearest shoreline, < 0 = water
    for lake in lakes:
        m = lake.max_radius + LAKE_SHORE_WIDTH + 2
        x0, x1 = max(0, int(lake.x - m)), min(w, int(lake.x + m) + 1)
        y0, y1 = max(0, int(lake.y - m)), min(h, int(lake.y + m) + 1)
        d = lake.shore_gap(np.arange(x0, x1, dtype=np.float32)[:, None] + 0.5,
                           np.arange(y0, y1, dtype=np.float32)[None, :] + 0.5)
        edge[x0:x1, y0:y1] = np.minimum(edge[x0:x1, y0:y1], d)
    water_mask = (edge < 0).astype(np.float32)
    blurred = sum(_blur_array(water_mask, f) for f in (8, 16, 32, 64)) / 4
    depth = np.clip((blurred - 0.5) * 1.6, 0, 1) * water_mask
    mud = np.clip((LAKE_SHORE_WIDTH - edge) / 4, 0, 1)[..., None]
    col[:] = col * (1 - mud) + np.array(LAKE_MUD, np.float32) * mud * (0.85 + 0.3 * ripple[..., None])
    t = (depth ** 0.8)[..., None]
    water = (np.array(LAKE_SHALLOW, np.float32) * (1 - t) + np.array(LAKE_DEEP, np.float32) * t)
    water *= 0.92 + 0.16 * ripple[..., None]
    wet = np.clip(0.5 - edge, 0, 1)[..., None]
    col[:] = col * (1 - wet) + water * wet


def _build_background(w, h, obstacles, lakes=()):
    rng = np.random.default_rng(7)
    rnd = random.Random(7)

    def blotch(cell):
        return _value_noise(rng, w, h, cell)

    shade = 0.6 * blotch(160) + 0.4 * blotch(40) - 0.5
    hue = blotch(110) - 0.5
    fine = rng.random((w, h)).astype(np.float32) - 0.5
    col = np.array([30, 44, 34], np.float32)[None, None, :] * (
        1 + 0.45 * shade[..., None] + 0.1 * fine[..., None])
    col += hue[..., None] * np.array([16, 6, -6], np.float32)
    xs = (np.arange(w, dtype=np.float32) / w - 0.5) * 2
    ys = (np.arange(h, dtype=np.float32) / h - 0.5) * 2
    d = np.sqrt(xs[:, None] ** 2 + ys[None, :] ** 2) / math.sqrt(2)
    col *= (1 - 0.5 * d ** 2.2)[..., None]
    if lakes:
        _paint_lakes(col, lakes, blotch(30))
    bg = pygame.surfarray.make_surface(np.clip(col, 0, 255).astype(np.uint8))

    def dry(x, y, margin=LAKE_SHORE_WIDTH):
        return not any(lake.shore_gap(x, y) < margin for lake in lakes)

    for _ in range(w * h // 700):
        bx, by = rnd.uniform(0, w), rnd.uniform(0, h)
        if not dry(bx, by):
            continue
        g = _mix((26, 48, 32), (70, 104, 60), rnd.random() ** 1.5)
        for _ in range(rnd.randint(2, 4)):
            a = -math.pi / 2 + rnd.uniform(-0.7, 0.7)
            ln = rnd.uniform(3, 7)
            pygame.draw.aaline(bg, g, (bx, by), (bx + math.cos(a) * ln, by + math.sin(a) * ln))
    for _ in range(w * h // 9000):
        px, py = rnd.uniform(0, w), rnd.uniform(0, h)
        r = rnd.uniform(1.5, 3.5)
        if not dry(px, py, 0):
            continue
        stone = _mix((58, 58, 54), (98, 94, 86), rnd.random())
        pygame.draw.ellipse(bg, (18, 24, 20), (px - r + 1, py - r * 0.7 + 1, r * 2, r * 1.4))
        pygame.draw.ellipse(bg, stone, (px - r, py - r * 0.7, r * 2, r * 1.4))
    for _ in range(w * h // 45000):
        px, py = rnd.uniform(0, w), rnd.uniform(0, h)
        if not dry(px, py):
            continue
        petal = rnd.choice(((214, 206, 150), (196, 180, 220), (220, 220, 220)))
        for k in range(5):
            a = k * 2 * math.pi / 5
            pygame.draw.circle(bg, _mix(petal, (40, 50, 40), 0.45),
                               (px + math.cos(a) * 1.6, py + math.sin(a) * 1.6), 1.2)
        pygame.draw.circle(bg, (230, 190, 70), (px, py), 1)

    for obs in obstacles:
        if isinstance(obs, Tree):
            _draw_tree_ground(bg, obs, rnd)
        else:
            _draw_rock(bg, obs.rect, rng, rnd)
    return bg.convert()


# =============================================================================
# Renderer
# =============================================================================
class Renderer:
    def __init__(self, world_w, world_h, obstacles, lakes=()):
        self.background = _build_background(world_w, world_h, obstacles, lakes)
        self.minimap_bg = pygame.transform.smoothscale(
            self.background, (config.MINIMAP_WIDTH, config.MINIMAP_HEIGHT))
        self.frame = 0
        self._base = {}
        self._rotated = {}
        self._glows = {}
        self._bubbles = {}
        self._phase = {}
        self._carcasses = {}
        self._night = None
        self._bg_cache = None  # (zoom, world rect, scaled background of that rect)
        self._food = {}  # draw radius -> [variant][level] sprites
        self.trees = [o for o in obstacles if isinstance(o, Tree)]
        self._crowns = {}  # (tree index, on-screen radius) -> (sprite, half width)

    # --- sprite caches -----------------------------------------------------
    def _creature_sprite(self, carn, sb, hl, frame, cb, ai):
        """Rotated sprite with its drop shadow baked in, plus the blit offset from the
        creature's position to the sprite's top-left corner."""
        key = (carn, sb, hl, frame, cb, ai)
        if len(self._rotated) >= config.SPRITE_CACHE_LIMIT:
            for k in list(islice(self._rotated, config.SPRITE_CACHE_LIMIT // 2)):
                del self._rotated[k]
            self._base.clear()  # zooming in builds large bases; they are cheap to redraw
        base_key = key[:5]
        entry = self._base.get(base_key)
        if entry is None:
            entry = self._base[base_key] = self._build_creature(*base_key)
        base, shadow_base, center_dx = entry
        theta = ai * 2 * math.pi / config.SPRITE_ANGLE_STEPS
        rot = pygame.transform.rotozoom(base, -math.degrees(theta), 1)
        shadow = pygame.transform.rotozoom(shadow_base, -math.degrees(theta), 1)
        rw, rh = rot.get_size()
        sx, sy = int(sb * 0.3) + 1, int(sb * 0.4) + 1
        comp = pygame.Surface((rw + sx, rh + sy), pygame.SRCALPHA)
        comp.blit(shadow, (sx, sy))
        comp.blit(rot, (0, 0))
        val = self._rotated[key] = (comp.convert_alpha(),
                                    center_dx * math.cos(theta) - rw / 2,
                                    center_dx * math.sin(theta) - rh / 2)
        return val

    @staticmethod
    def _build_creature(carn, sb, hl, frame, cb):
        """Unrotated sprite facing +x, its shadow silhouette, and the x offset of the
        sprite centre from the creature origin."""
        x0, x1, y_half = (-2.32, 1.92, 0.86) if carn else (-1.92, 2.32, 0.76)
        w = math.ceil((x1 - x0) * sb) + 2
        h = math.ceil(2 * y_half * sb) + 2
        ox = 1 - x0 * sb
        big = pygame.Surface((w * SS, h * SS), pygame.SRCALPHA)
        pal = _faded_palette(_tinted_palette(CARNIVORE_PALETTE if carn else HERBIVORE_PALETTE, cb), hl)
        (_draw_carnivore if carn else _draw_herbivore)(big, ox * SS, h * SS / 2, sb * SS, pal, frame)
        sprite = pygame.transform.smoothscale(big, (w, h))
        shadow = _blur(sprite, 2)
        shadow.fill((0, 0, 0), special_flags=pygame.BLEND_RGB_MULT)
        shadow.fill((255, 255, 255, 110), special_flags=pygame.BLEND_RGBA_MULT)
        return sprite, shadow, w / 2 - ox

    def _glow(self, sb, color, level):
        key = (sb, color, level)
        g = self._glows.get(key)
        if g is None:
            r = sb * 2.0 + 6
            w = math.ceil(r * 2)
            g = pygame.Surface((w, w))
            intensity = 0.55 + 0.45 * level / (PULSE_LEVELS - 1)
            for i in range(12):
                t = i / 11
                k = intensity * (t ** 2) * 0.8
                pygame.draw.circle(g, tuple(int(v * k) for v in color), (w / 2, w / 2), r * (1 - 0.8 * t))
            g = self._glows[key] = _blur(g, 2).convert()
        return g

    def _bubble(self, sb, level):
        key = (sb, level)
        b = self._bubbles.get(key)
        if b is None:
            r = sb * 1.6 + 5
            w = math.ceil(r * 2 + 2)
            a = (level + 1) / PULSE_LEVELS
            big = pygame.Surface((w * SS, w * SS), pygame.SRCALPHA)
            pygame.draw.circle(big, (110, 150, 255, int(50 * a)), (w * SS / 2, w * SS / 2), r * SS)
            pygame.draw.circle(big, (170, 205, 255, int(200 * a)), (w * SS / 2, w * SS / 2), r * SS,
                               width=int(1.5 * SS))
            pygame.draw.circle(big, (240, 248, 255, int(220 * a)),
                               (w * SS / 2 - r * SS * 0.45, w * SS / 2 - r * SS * 0.45), r * SS * 0.16)
            b = self._bubbles[key] = pygame.transform.smoothscale(big, (w, w)).convert_alpha()
        return b

    def _carcass(self, sb, carn, level):
        key = (sb, carn, level)
        spr = self._carcasses.get(key)
        if spr is None:
            spr = self._carcasses[key] = self._build_carcass(sb, carn, level)
        return spr

    @staticmethod
    def _build_carcass(sb, carn, level):
        """Meat with exposed ribs; `level` 0 is fresh, CARCASS_LEVELS - 1 nearly gone."""
        rnd = random.Random(sb * 31 + carn)
        rot = 1 - level / (CARCASS_LEVELS - 1)
        w = math.ceil(sb * 2.6) + 4
        big = pygame.Surface((w * SS, w * SS), pygame.SRCALPHA)
        c, u = w * SS / 2, sb * SS
        meat = _mix((96, 70, 52), (150, 34, 36) if not carn else (130, 40, 28), rot)
        rim = _mix(meat, (0, 0, 0), 0.45)
        bone = _mix((170, 160, 140), (236, 228, 210), rot)
        shrink = 0.6 + 0.4 * rot
        tilt = rnd.uniform(0, math.pi)
        _shaded_ellipse(big, rim, meat, c, c, 1.05 * u * shrink, 0.7 * u * shrink, tilt,
                        hx=-0.2 * u, hy=-0.2 * u)
        for i in range(4):
            off = (i - 1.5) * 0.32 * u * shrink
            dx, dy = math.cos(tilt) * off, math.sin(tilt) * off
            nx, ny = -math.sin(tilt) * 0.55 * u, math.cos(tilt) * 0.55 * u
            pygame.draw.line(big, bone, (c + dx - nx, c + dy - ny), (c + dx + nx, c + dy + ny),
                             max(1, int(0.14 * u)))
        pygame.draw.line(big, bone, (c - math.cos(tilt) * 0.8 * u, c - math.sin(tilt) * 0.8 * u),
                         (c + math.cos(tilt) * 0.8 * u, c + math.sin(tilt) * 0.8 * u),
                         max(1, int(0.2 * u)))
        small = pygame.transform.smoothscale(big, (w, w))
        small.fill((255, 255, 255, int(110 + 145 * rot)), special_flags=pygame.BLEND_RGBA_MULT)
        return small.convert_alpha(), w / 2

    def _food_sprites(self, radius):
        sprites = self._food.get(radius)
        if sprites is None:
            sprites = self._food[radius] = [
                [self._build_food(v, lvl, radius) for lvl in range(FOOD_LEVELS)]
                for v in range(FOOD_VARIANTS)]
        return sprites

    @staticmethod
    def _build_food(variant, level, radius=FOOD_DRAW_RADIUS):
        rnd = random.Random(variant * 97 + 3)
        fresh = 1 - level / (FOOD_LEVELS - 1)
        w = radius * 2 + 2
        big = pygame.Surface((w * SS, w * SS), pygame.SRCALPHA)
        c, u = w * SS / 2, radius * SS
        leaf_rim = _mix((84, 62, 32), (30, 112, 46), fresh)
        leaf = _mix((140, 104, 50), (104, 206, 84), fresh)
        vein = _mix((150, 120, 70), (150, 214, 120), fresh)
        n = rnd.randint(4, 6)
        rot = rnd.uniform(0, 2 * math.pi)
        shrink = 0.75 + 0.25 * fresh
        for i in range(n):
            a = rot + i * 2 * math.pi / n + rnd.uniform(-0.3, 0.3)
            lx, ly = c + math.cos(a) * 0.48 * u * shrink, c + math.sin(a) * 0.48 * u * shrink
            _ellipse(big, leaf_rim, lx, ly, 0.52 * u * shrink, 0.26 * u * shrink, a)
            _ellipse(big, leaf, lx, ly, 0.42 * u * shrink, 0.17 * u * shrink, a)
            pygame.draw.line(big, vein, (c, c), (c + math.cos(a) * 0.85 * u * shrink,
                                                 c + math.sin(a) * 0.85 * u * shrink), SS)
        berry = _mix((92, 52, 42), (246, 52, 78), fresh)
        for i in range(3):
            a = rot + i * 2 * math.pi / 3 + 0.5
            bx, by = c + math.cos(a) * 0.2 * u, c + math.sin(a) * 0.2 * u
            pygame.draw.circle(big, _mix(berry, (0, 0, 0), 0.35), (bx, by), 0.24 * u)
            pygame.draw.circle(big, berry, (bx, by), 0.2 * u)
            pygame.draw.circle(big, _mix(berry, (255, 255, 255), 0.6), (bx - 0.07 * u, by - 0.07 * u),
                               0.06 * u)
        small = pygame.transform.smoothscale(big, (w, w))
        alpha = int(50 + 205 * fresh)
        small.fill((255, 255, 255, alpha), special_flags=pygame.BLEND_RGBA_MULT)
        return small.convert_alpha()

    # --- frame ---------------------------------------------------------------
    def _draw_background(self, screen, cam_x, cam_y, zoom):
        """Blit the background under the camera. When zoomed, a scaled copy of the view
        plus a margin is cached and only rebuilt when the camera leaves it or the zoom
        changes, so a frame doesn't pay for a full-screen smoothscale."""
        sw, sh = screen.get_size()
        if zoom == 1.0:
            screen.blit(self.background, (0, 0), (int(cam_x), int(cam_y), sw, sh))
            return
        vw, vh = sw / zoom, sh / zoom
        world = self.background.get_rect()
        view = pygame.Rect(int(cam_x), int(cam_y), math.ceil(vw) + 1, math.ceil(vh) + 1)
        cache = self._bg_cache
        if cache is None or cache[0] != zoom or not cache[1].contains(view.clip(world)):
            mx, my = vw * BG_CACHE_MARGIN, vh * BG_CACHE_MARGIN
            rect = pygame.Rect(int(cam_x - mx), int(cam_y - my),
                               math.ceil(vw + 2 * mx) + 1, math.ceil(vh + 2 * my) + 1)
            rect = rect.clip(world)
            scaled = pygame.transform.smoothscale(
                self.background.subsurface(rect),
                (round(rect.width * zoom), round(rect.height * zoom)))
            cache = self._bg_cache = (zoom, rect, scaled)
        _, rect, scaled = cache
        screen.blit(scaled, (0, 0), (round((cam_x - rect.x) * zoom),
                                     round((cam_y - rect.y) * zoom), sw, sh))

    def draw_world(self, screen, sim_state):
        """Draw the camera's view at sim_state["zoom"]: positions and sprite sizes are scaled,
        and sprites are built (and cached) at their on-screen size, so zooming stays sharp."""
        self.frame += 1
        cam_x, cam_y = sim_state["cam_x"], sim_state["cam_y"]
        z = sim_state.get("zoom", 1.0)
        sw, sh = screen.get_size()
        self._draw_background(screen, cam_x, cam_y, z)

        food_r = max(2, round(FOOD_DRAW_RADIUS * z))
        food_sprites = self._food_sprites(food_r)
        fo = food_r + 1
        lifetime_scale = FOOD_LEVELS / config.FOOD_MAX_LIFETIME
        food_blits = []
        for f in sim_state["food_items"]:
            x, y = (f.pos.x - cam_x) * z, (f.pos.y - cam_y) * z
            if -fo <= x <= sw + fo and -fo <= y <= sh + fo:
                lvl = min(FOOD_LEVELS - 1, int(f.lifetime * lifetime_scale))
                food_blits.append((food_sprites[f.id % FOOD_VARIANTS][lvl], (x - fo, y - fo)))
        screen.blits(food_blits, doreturn=False)

        carcass_blits = []
        for cc in sim_state.get("carcasses", ()):
            x, y = (cc.pos.x - cam_x) * z, (cc.pos.y - cam_y) * z
            r = (cc.radius * 1.4 + 3) * z
            if -r <= x <= sw + r and -r <= y <= sh + r:
                lvl = min(CARCASS_LEVELS - 1, int((1 - cc.meat / cc.max_meat) * CARCASS_LEVELS))
                spr, half = self._carcass(max(2, int(cc.radius * z + 0.5)), cc.is_carnivore, lvl)
                carcass_blits.append((spr, (x - half, y - half)))
        screen.blits(carcass_blits, doreturn=False)

        selected = sim_state.get("selected_creature")
        visible = []
        for c in sim_state["creatures"] + sim_state["carnivores"]:
            x, y = (c.pos.x - cam_x) * z, (c.pos.y - cam_y) * z
            r = (c.size * 2.2 + 8) * z
            if -r <= x <= sw + r and -r <= y <= sh + r:
                visible.append((c, x, y))

        now = time.time()
        pulse = int((math.sin(self.frame * 0.08) + 1) * 0.5 * (PULSE_LEVELS - 0.01))
        steps = config.SPRITE_ANGLE_STEPS
        angle_scale = steps / (2 * math.pi)
        rit = config.RESPAWN_IMMUNITY_TICKS
        tick = sim_state.get("tick", 0)
        rotated, phases = self._rotated, self._phase
        sprites, bubbles, signallers = [], [], []
        for c, x, y in visible:
            sb = int(c.size * z + 0.5)
            if sb < 2:
                sb = 2
            hl = int(c.health / c.max_health * HEALTH_LEVELS) if c.max_health else 0
            hl = 0 if hl < 0 else HEALTH_LEVELS - 1 if hl >= HEALTH_LEVELS else hl

            frame = 0
            if sb > 3:
                phase = phases.get(c.id, 0.0)
                if c.current_speed > 0.3:
                    phase += c.current_speed / (c.size * 0.5 + 2) * 0.35
                    phases[c.id] = phase
                frame = int(phase) % WALK_FRAMES

            key = (c.is_carnivore, sb, hl, frame, color_bucket(c.genes["color"]),
                   int(c.angle * angle_scale + 0.5) % steps)
            entry = rotated.get(key)
            if entry is None:
                entry = self._creature_sprite(*key)
            spr, ox, oy = entry

            glow_color = None
            if c.last_hit_tick is not None and tick - c.last_hit_tick < HIT_FLASH_TICKS:
                glow_color = config.COLOR_HIT
            elif c.is_best:
                glow_color = config.COLOR_BEST_CREATURE
            elif c.birth_time and now - c.birth_time < 3:
                glow_color = config.COLOR_BIRTH
            if glow_color:
                g = self._glow(sb, glow_color, pulse)
                gw = g.get_width() / 2
                screen.blit(g, (x - gw, y - gw), special_flags=pygame.BLEND_RGB_ADD)

            sprites.append((spr, (x + ox, y + oy)))
            if c.signal > config.SIGNAL_RENDER_THRESHOLD:
                signallers.append((c, x, y))

            if c.respawn_immunity_ticks > 0 and c.is_carnivore:
                lvl = min(PULSE_LEVELS - 1, int(c.respawn_immunity_ticks / rit * PULSE_LEVELS))
                b = self._bubble(sb, lvl)
                bw = b.get_width() / 2
                bubbles.append((b, (x - bw, y - bw)))
        screen.blits(sprites, doreturn=False)
        screen.blits(bubbles, doreturn=False)
        for c, x, y in signallers:
            t = ((self.frame + c.id * 7) % 30) / 30
            color = (255, 230, 120) if c.is_carnivore else (150, 230, 255)
            pygame.gfxdraw.aacircle(screen, int(x), int(y),
                                    int((c.size * 1.3 + 4 + t * 16) * z),
                                    (*color, int(220 * (1 - t) * c.signal)))

        self._draw_crowns(screen, cam_x, cam_y, z)
        for (c, _, _), (spr, pos) in zip(visible, sprites):
            if self._under_crown(c):
                spr.set_alpha(CROWN_GHOST_ALPHA)
                screen.blit(spr, pos)
                spr.set_alpha(255)
        self._draw_night(screen, sim_state.get("light", 1.0), food_blits, fo, z)

        for c, x, y in visible:
            if c is selected:
                self._draw_selection(screen, c, x, y, z)
            elif c.is_best:
                self._draw_whiskers(screen, c, x, y, z)

        if self.frame % 600 == 0:
            alive = {c.id for c in sim_state["creatures"]} | {c.id for c in sim_state["carnivores"]}
            self._phase = {k: v for k, v in self._phase.items() if k in alive}

    def _under_crown(self, c):
        x, y = c.pos.x, c.pos.y
        for t in self.trees:
            dx, dy, r = x - t.x, y - t.y, t.crown_radius + c.size
            if dx * dx + dy * dy < r * r:
                return True
        return False

    def _draw_crowns(self, screen, cam_x, cam_y, z):
        if len(self._crowns) > 4 * len(self.trees):
            self._crowns.clear()
        sw, sh = screen.get_size()
        blits = []
        for i, t in enumerate(self.trees):
            x, y = (t.x - cam_x) * z, (t.y - cam_y) * z
            r = max(2, round(t.crown_radius * z))
            m = r * 1.2
            if -m <= x <= sw + m and -m <= y <= sh + m:
                entry = self._crowns.get((i, r))
                if entry is None:
                    entry = self._crowns[(i, r)] = _build_crown(r, i)
                spr, half = entry
                blits.append((spr, (x - half, y - half)))
        screen.blits(blits, doreturn=False)

    def _draw_night(self, screen, light, food_blits, fo, z):
        """Darken the world, then let food glow faintly through the dark."""
        darkness = 1 - light
        alpha = int(darkness * config.NIGHT_MAX_ALPHA)
        if alpha < 4:
            return
        if self._night is None or self._night.get_size() != screen.get_size():
            self._night = pygame.Surface(screen.get_size()).convert()
            self._night.fill(NIGHT_COLOR)
        self._night.set_alpha(alpha)
        screen.blit(self._night, (0, 0))
        level = int(darkness * PULSE_LEVELS)
        if level >= 2:
            g = self._glow(max(2, round(2 * z)), (40, 130, 60), min(PULSE_LEVELS - 1, level))
            gw = g.get_width() / 2
            add = pygame.BLEND_RGB_ADD
            screen.blits([(g, (x + fo - gw, y + fo - gw), None, add) for _, (x, y) in food_blits],
                         doreturn=False)

    def _draw_whiskers(self, screen, c, x, y, z):
        color = config.COLOR_CARNIVORE_WHISKER if c.is_carnivore else config.COLOR_WHISKER
        for i, a in enumerate(c.whisker_angles):
            ex = x + math.cos(c.angle + a) * c.whiskers[i] * z
            ey = y + math.sin(c.angle + a) * c.whiskers[i] * z
            pygame.draw.aaline(screen, color, (x, y), (ex, ey))
            pygame.draw.circle(screen, color, (ex, ey), 2)
            seen = c.target_whiskers[i]
            if seen is not None:  # food / prey on this ray
                seen *= z
                tx, ty = x + math.cos(c.angle + a) * seen, y + math.sin(c.angle + a) * seen
                pygame.draw.circle(screen, config.COLOR_TARGET_WHISKER, (tx, ty), 3)

    def _draw_selection(self, screen, c, x, y, z):
        sd, sa = (c.effective_sight or c.sight_distance) * z, c.sight_angle
        n = max(8, int(sa * 10))
        arc = [(int(x + math.cos(c.angle - sa / 2 + sa * i / n) * sd),
                int(y + math.sin(c.angle - sa / 2 + sa * i / n) * sd)) for i in range(n + 1)]
        pts = arc if sa >= 2 * math.pi - 1e-3 else [(int(x), int(y))] + arc
        pygame.gfxdraw.filled_polygon(screen, pts, (255, 255, 255, 18))
        pygame.gfxdraw.aapolygon(screen, pts, (255, 255, 255, 70))
        self._draw_whiskers(screen, c, x, y, z)
        r = (c.size * 1.6 + 7) * z
        rect = pygame.Rect(0, 0, r * 2, r * 2)
        rect.center = (int(x), int(y))
        dashes = 10
        for k in range(dashes):
            a0 = self.frame * 0.04 + k * 2 * math.pi / dashes
            pygame.draw.arc(screen, (255, 255, 255), rect, a0, a0 + 0.35, 2)
