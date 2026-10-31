"""Drawing primitives shared by the sprite and terrain art."""
import math

import numpy as np
import pygame

from .style import mix


def ellipse_pts(cx, cy, rx, ry, rot=0.0, n=28):
    cr, sr = math.cos(rot), math.sin(rot)
    pts = []
    for i in range(n):
        a = 2 * math.pi * i / n
        x, y = rx * math.cos(a), ry * math.sin(a)
        pts.append((cx + x * cr - y * sr, cy + x * sr + y * cr))
    return pts


def ellipse(surf, color, cx, cy, rx, ry, rot=0.0):
    pygame.draw.polygon(surf, color, ellipse_pts(cx, cy, rx, ry, rot))


def shaded_ellipse(surf, rim, core, cx, cy, rx, ry, rot=0.0, steps=6, hx=0.0, hy=0.0):
    """Ellipse shaded from its rim colour to a highlight core shifted by (hx, hy)."""
    for i in range(steps):
        t = i / (steps - 1)
        k = 1 - 0.55 * t
        ellipse(surf, mix(rim, core, t), cx + hx * t, cy + hy * t, rx * k, ry * k, rot)



def blur(surf, factor):
    w, h = surf.get_size()
    small = pygame.transform.smoothscale(surf, (max(1, w // factor), max(1, h // factor)))
    return pygame.transform.smoothscale(small, (w, h))


def value_noise(rng, w, h, cell):
    """Smooth noise in [0, 1] with features roughly `cell` pixels across, shape (w, h)."""
    small = (rng.random((max(2, w // cell), max(2, h // cell))) * 255).astype(np.uint8)
    surf = pygame.transform.smoothscale(pygame.surfarray.make_surface(np.dstack([small] * 3)), (w, h))
    return pygame.surfarray.array_red(surf).astype(np.float32) / 255.0



def blur_array(a, factor):
    """Soft blur of a [0, 1] (w, h) array by smoothscaling down `factor` times and back."""
    w, h = a.shape
    surf = pygame.surfarray.make_surface(np.dstack([(a * 255).astype(np.uint8)] * 3))
    small = pygame.transform.smoothscale(surf, (max(1, w // factor), max(1, h // factor)))
    return pygame.surfarray.array_red(pygame.transform.smoothscale(small, (w, h))).astype(np.float32) / 255
