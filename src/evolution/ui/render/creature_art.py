"""Creature sprites, drawn facing +x; u = body radius in supersampled pixels."""
import math

import pygame

from .shapes import blur, ellipse, shaded_ellipse
from .style import (
    CARNIVORE_PALETTE,
    HERBIVORE_PALETTE,
    SS,
    faded_palette,
    mix,
    tinted_palette,
)


def draw_herbivore(s, cx, cy, u, p, frame):
    ph = frame * math.pi / 2
    sway = math.sin(ph + 1.0)
    n = 18
    for i in range(n):
        t = i / (n - 1)
        r = 0.26 * (1 - t) ** 1.2 + 0.03
        ellipse(s, mix(p["rim"], p["core"], 0.4 * (1 - t)),
                 cx + (-0.75 - 1.1 * t) * u, cy + sway * 0.3 * t * t * u, r * u, r * u)
    swing = math.sin(ph) * 0.16 * u
    for fx, side, sgn in ((0.45, 1, 1), (0.45, -1, -1), (-0.6, 1, -1), (-0.6, -1, 1)):
        lx, ly = cx + fx * u + sgn * swing, cy + side * 0.5 * u
        shaded_ellipse(s, p["leg"], p["core"], lx, ly, 0.21 * u, 0.19 * u, steps=4,
                        hy=-side * 0.03 * u)
        for d in (-0.09, 0.0, 0.09):
            ellipse(s, p["claw"], lx + 0.19 * u, ly + d * u, 0.04 * u, 0.035 * u)
    shaded_ellipse(s, p["rim"], p["core"], cx - 0.12 * u, cy, 1.0 * u, 0.56 * u, steps=12,
                    hx=0.1 * u)
    spot = mix(p["core"], p["spot"], 0.6)
    for sx, sy, r in ((-0.5, -0.26, 0.08), (-0.15, 0.3, 0.07), (-0.72, 0.14, 0.06),
                      (0.18, -0.26, 0.06), (0.32, 0.2, 0.05), (-0.32, 0.08, 0.05)):
        ellipse(s, spot, cx + sx * u, cy + sy * u, r * u, r * u * 0.8)
    for i in range(7):
        ellipse(s, p["stripe"], cx + (-0.85 + i * 0.25) * u, cy, 0.07 * u, 0.04 * u)
    neck_sway = math.sin(ph + 2.5) * 0.12
    n = 16
    for i in range(n):
        t = i / (n - 1)
        r = 0.3 - 0.17 * t ** 0.7
        ellipse(s, mix(p["rim"], p["core"], 0.55 + 0.25 * t),
                 cx + (0.55 + 1.3 * t) * u, cy + neck_sway * t * t * u, r * u, r * u)
    hx, hy = cx + 2.02 * u, cy + neck_sway * u
    shaded_ellipse(s, p["rim"], p["core"], hx, hy, 0.26 * u, 0.18 * u, steps=5, hx=0.03 * u)
    ellipse(s, p["crest"], hx - 0.04 * u, hy, 0.1 * u, 0.07 * u)
    for side in (1, -1):
        ey = hy + side * 0.13 * u
        ellipse(s, p["eye"], hx - 0.02 * u, ey, 0.06 * u, 0.05 * u)
        ellipse(s, p["pupil"], hx + 0.0 * u, ey, 0.035 * u, 0.035 * u)
        ellipse(s, p["nose"], hx + 0.2 * u, hy + side * 0.05 * u, 0.025 * u, 0.02 * u)


def _head_outline(cx, cy, u, k=1.0, dx=0.0):
    """Top-down T-Rex skull outline, shrunk by k toward its centre and shifted by dx (in u)."""
    half = ((0.84, 0.32), (0.98, 0.43), (1.22, 0.41), (1.5, 0.32), (1.73, 0.23), (1.85, 0.12),
            (1.88, 0.0))
    mid = 1.36
    pts = [(mid + (x - mid) * k + dx, -y * k) for x, y in half]
    pts += [(x, -y) for x, y in reversed(pts[:-1])]
    return [(cx + x * u, cy + y * u) for x, y in pts]


def draw_carnivore(s, cx, cy, u, p, frame):
    ph = frame * math.pi / 2
    sway = math.sin(ph + 1.0)
    n = 26
    for i in range(n):
        t = i / (n - 1)
        r = 0.38 * (1 - t) ** 1.2 + 0.03
        ellipse(s, mix(p["rim"], p["core"], 0.35 * (1 - t)),
                 cx + (-0.5 - 1.75 * t) * u, cy + sway * 0.38 * t * t * u, r * u, r * u)
    swing = math.sin(ph) * 0.24 * u
    for side in (1, -1):
        lx, ly = cx - 0.12 * u + side * swing, cy + side * 0.5 * u
        fx, fy = lx + 0.34 * u, ly + side * 0.1 * u
        ellipse(s, p["leg"], fx, fy, 0.16 * u, 0.11 * u)
        for a in (-0.55, 0.0, 0.55):
            ellipse(s, p["claw"], fx + 0.16 * u * math.cos(a), fy + 0.13 * u * math.sin(a),
                     0.055 * u, 0.035 * u, rot=a)
        shaded_ellipse(s, p["leg"], p["core"], lx, ly, 0.4 * u, 0.22 * u, steps=5,
                        hx=0.05 * u, hy=-side * 0.04 * u)
    shaded_ellipse(s, p["rim"], p["core"], cx - 0.1 * u, cy, 0.88 * u, 0.5 * u, steps=7,
                    hx=0.08 * u)
    shaded_ellipse(s, p["rim"], p["core"], cx + 0.68 * u, cy, 0.4 * u, 0.34 * u, steps=5)
    for sx, sy, r in ((-0.5, 0.22, 0.07), (-0.2, -0.27, 0.06), (0.15, 0.24, 0.05),
                      (-0.75, -0.12, 0.05), (0.4, -0.18, 0.045)):
        ellipse(s, mix(p["rim"], p["core"], 0.4), cx + sx * u, cy + sy * u, r * u, r * u)
    for i in range(9):
        x = -1.3 + i * 0.2
        k = 1 - abs(x + 0.4) * 0.35
        ellipse(s, mix(p["stripe"], p["rim"], 0.3), cx + x * u, cy, 0.08 * u * k, 0.05 * u * k)
    for side in (1, -1):
        ax, ay = cx + 0.6 * u, cy + side * 0.36 * u
        ellipse(s, p["leg"], ax, ay, 0.17 * u, 0.075 * u, rot=side * 0.5)
        ellipse(s, p["core"], ax - 0.01 * u, ay - side * 0.01 * u, 0.12 * u, 0.04 * u,
                 rot=side * 0.5)
        for d in (-0.04, 0.04):
            ellipse(s, p["claw"], ax + 0.15 * u, ay + (side * 0.09 + d) * u, 0.035 * u, 0.028 * u)
    pygame.draw.polygon(s, p["rim"], _head_outline(cx, cy, u))
    pygame.draw.polygon(s, mix(p["rim"], p["core"], 0.6), _head_outline(cx, cy, u, 0.86))
    pygame.draw.polygon(s, p["core"], _head_outline(cx, cy, u, 0.64, -0.03))
    ellipse(s, p["snout"], cx + 1.58 * u, cy, 0.24 * u, 0.15 * u)
    for side in (1, -1):
        for x, y in ((1.25, 0.39), (1.38, 0.36), (1.5, 0.32), (1.62, 0.275), (1.74, 0.22)):
            pygame.draw.polygon(s, p["fang"], [
                (cx + (x - 0.045) * u, cy + side * (y - 0.03) * u),
                (cx + (x + 0.045) * u, cy + side * (y - 0.03) * u),
                (cx + (x + 0.01) * u, cy + side * (y + 0.06) * u),
            ])
        ey = cy + side * 0.27 * u
        ellipse(s, p["stripe"], cx + 1.1 * u, cy + side * 0.21 * u, 0.16 * u, 0.06 * u,
                 rot=side * 0.3)
        ellipse(s, p["eye"], cx + 1.08 * u, ey, 0.1 * u, 0.06 * u, rot=side * 0.35)
        ellipse(s, p["pupil"], cx + 1.1 * u, ey, 0.022 * u, 0.055 * u, rot=side * 0.35)
        ellipse(s, p["nose"], cx + 1.79 * u, cy + side * 0.07 * u, 0.035 * u, 0.025 * u)


def build_creature(carn, sb, hl, frame, cb):
    """Unrotated sprite facing +x, its shadow silhouette, and the x offset of the
    sprite centre from the creature origin."""
    x0, x1, y_half = (-2.32, 1.92, 0.86) if carn else (-1.92, 2.32, 0.76)
    w = math.ceil((x1 - x0) * sb) + 2
    h = math.ceil(2 * y_half * sb) + 2
    ox = 1 - x0 * sb
    big = pygame.Surface((w * SS, h * SS), pygame.SRCALPHA)
    pal = faded_palette(tinted_palette(CARNIVORE_PALETTE if carn else HERBIVORE_PALETTE, cb), hl)
    (draw_carnivore if carn else draw_herbivore)(big, ox * SS, h * SS / 2, sb * SS, pal, frame)
    sprite = pygame.transform.smoothscale(big, (w, h))
    shadow = blur(sprite, 2)
    shadow.fill((0, 0, 0), special_flags=pygame.BLEND_RGB_MULT)
    shadow.fill((255, 255, 255, 110), special_flags=pygame.BLEND_RGBA_MULT)
    return sprite, shadow, w / 2 - ox
