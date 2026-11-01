"""Sprites for food and the glow and bubble effects around creatures."""
import math
import random

import pygame

from .shapes import blur, ellipse
from .style import FOOD_DRAW_RADIUS, FOOD_LEVELS, PULSE_LEVELS, SS, mix


def build_glow(sb, color, level):
    r = sb * 2.0 + 6
    w = math.ceil(r * 2)
    g = pygame.Surface((w, w))
    intensity = 0.55 + 0.45 * level / (PULSE_LEVELS - 1)
    for i in range(12):
        t = i / 11
        k = intensity * (t ** 2) * 0.8
        pygame.draw.circle(g, tuple(int(v * k) for v in color), (w / 2, w / 2), r * (1 - 0.8 * t))
    return blur(g, 2).convert()


def build_bubble(sb, level):
    """The respawn-immunity bubble; `level` sets its opacity."""
    r = sb * 1.6 + 5
    w = math.ceil(r * 2 + 2)
    a = (level + 1) / PULSE_LEVELS
    big = pygame.Surface((w * SS, w * SS), pygame.SRCALPHA)
    pygame.draw.circle(big, (110, 150, 255, int(50 * a)), (w * SS / 2, w * SS / 2), r * SS)
    pygame.draw.circle(big, (170, 205, 255, int(200 * a)), (w * SS / 2, w * SS / 2), r * SS,
                       width=int(1.5 * SS))
    pygame.draw.circle(big, (240, 248, 255, int(220 * a)),
                       (w * SS / 2 - r * SS * 0.45, w * SS / 2 - r * SS * 0.45), r * SS * 0.16)
    return pygame.transform.smoothscale(big, (w, w)).convert_alpha()


def build_food(variant, level, radius=FOOD_DRAW_RADIUS):
    """A leafy plant with berries; `level` 0 is fresh, FOOD_LEVELS - 1 nearly rotten."""
    rnd = random.Random(variant * 97 + 3)
    fresh = 1 - level / (FOOD_LEVELS - 1)
    w = radius * 2 + 2
    big = pygame.Surface((w * SS, w * SS), pygame.SRCALPHA)
    c, u = w * SS / 2, radius * SS
    leaf_rim = mix((84, 62, 32), (30, 112, 46), fresh)
    leaf = mix((140, 104, 50), (104, 206, 84), fresh)
    vein = mix((150, 120, 70), (150, 214, 120), fresh)
    n = rnd.randint(4, 6)
    rot = rnd.uniform(0, 2 * math.pi)
    shrink = 0.75 + 0.25 * fresh
    for i in range(n):
        a = rot + i * 2 * math.pi / n + rnd.uniform(-0.3, 0.3)
        lx, ly = c + math.cos(a) * 0.48 * u * shrink, c + math.sin(a) * 0.48 * u * shrink
        ellipse(big, leaf_rim, lx, ly, 0.52 * u * shrink, 0.26 * u * shrink, a)
        ellipse(big, leaf, lx, ly, 0.42 * u * shrink, 0.17 * u * shrink, a)
        pygame.draw.line(big, vein, (c, c), (c + math.cos(a) * 0.85 * u * shrink,
                                             c + math.sin(a) * 0.85 * u * shrink), SS)
    berry = mix((92, 52, 42), (246, 52, 78), fresh)
    for i in range(3):
        a = rot + i * 2 * math.pi / 3 + 0.5
        bx, by = c + math.cos(a) * 0.2 * u, c + math.sin(a) * 0.2 * u
        pygame.draw.circle(big, mix(berry, (0, 0, 0), 0.35), (bx, by), 0.24 * u)
        pygame.draw.circle(big, berry, (bx, by), 0.2 * u)
        pygame.draw.circle(big, mix(berry, (255, 255, 255), 0.6), (bx - 0.07 * u, by - 0.07 * u),
                           0.06 * u)
    small = pygame.transform.smoothscale(big, (w, w))
    alpha = int(50 + 205 * fresh)
    small.fill((255, 255, 255, alpha), special_flags=pygame.BLEND_RGBA_MULT)
    return small.convert_alpha()
