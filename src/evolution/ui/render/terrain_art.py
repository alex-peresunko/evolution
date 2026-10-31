"""The static background (grass, lakes, rocks, tree ground) and tree crowns."""
import math
import random

import numpy as np
import pygame

from ... import config
from ...world.terrain import Tree
from .shapes import blur, blur_array, shaded_ellipse, value_noise
from .style import LAKE_DEEP, LAKE_MUD, LAKE_SHALLOW, LAKE_SHORE_WIDTH, SS, mix


def _draw_rock(bg, rect, rng, rnd):
    """Bevelled, lit boulder filling `rect`, from a rounded-box distance field."""
    x, y, w, h = int(rect.x), int(rect.y), int(rect.width), int(rect.height)
    m = min(w, h)
    rad = m * 0.3

    pad = 18
    shadow = pygame.Surface((w + pad * 2, h + pad * 2), pygame.SRCALPHA)
    pygame.draw.rect(shadow, (0, 0, 0, 150), (pad, pad, w, h), border_radius=int(rad))
    bg.blit(blur(shadow, 7), (x - pad + 9, y - pad + 11))

    px = (np.arange(w, dtype=np.float32) + 0.5 - w / 2)[:, None]
    py = (np.arange(h, dtype=np.float32) + 0.5 - h / 2)[None, :]
    qx = np.abs(px) - (w / 2 - rad - 1)
    qy = np.abs(py) - (h / 2 - rad - 1)
    d = (np.sqrt(np.maximum(qx, 0) ** 2 + np.maximum(qy, 0) ** 2)
         + np.minimum(np.maximum(qx, qy), 0) - rad)
    d += (value_noise(rng, w, h, 14) - 0.5) * m * 0.07

    bevel = m * 0.24
    height = np.clip(-d / bevel, 0, 1)
    height = 1 - (1 - height) ** 2
    mottle = value_noise(rng, w, h, 9)
    height = height + 0.05 * mottle + 0.025 * value_noise(rng, w, h, 3)
    gx, gy = np.gradient(height * bevel * 0.8)
    nz = 1 / np.sqrt(gx ** 2 + gy ** 2 + 1)
    light = np.array([-0.5, -0.6, 0.62], np.float32)
    light /= np.linalg.norm(light)
    lam = np.clip((-gx * light[0] - gy * light[1] + light[2]) * nz, 0, 1)

    tone = rnd.uniform(-10, 10)
    base = np.array([104 + tone, 99 + tone, 92 + tone], np.float32)
    shade = (0.28 + 0.95 * lam) * (0.82 + 0.36 * mottle)
    col = base[None, None, :] * shade[..., None]
    moss = np.clip((value_noise(rng, w, h, 16) - 0.6) * 5, 0, 1) * (height > 0.6)
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
    bg.blit(blur(shadow, 6), (tree.x - w / 2 + r * 0.22, tree.y - w / 2 + r * 0.28))

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
    shaded_ellipse(big, dark, (132, 98, 64), c, c, t * SS, t * SS, steps=6,
                    hx=-0.3 * t * SS, hy=-0.35 * t * SS)
    for k in range(3):
        pygame.draw.circle(big, mix(bark, dark, 0.5), (c, c), t * SS * (0.25 + 0.2 * k), width=SS)
    small = pygame.transform.smoothscale(big, (big.get_width() // SS,) * 2)
    bg.blit(small, (tree.x - small.get_width() / 2, tree.y - small.get_height() / 2))


def build_crown(r, seed):
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
    leaves = value_noise(rng, w, w, max(2, int(r * 0.09)))
    twigs = value_noise(rng, w, w, max(2, int(r * 0.035)))
    bump = height + 0.05 * leaves + 0.022 * twigs
    gx, gy = np.gradient(bump * r * 0.9)
    nz = 1 / np.sqrt(gx ** 2 + gy ** 2 + 1)
    light = np.array([-0.5, -0.6, 0.62], np.float32)
    light /= np.linalg.norm(light)
    lam = np.clip((-gx * light[0] - gy * light[1] + light[2]) * nz, 0, 1)
    occl = np.clip(height / 0.55, 0, 1) ** 0.6

    tone = rnd.uniform(-1, 1)
    dark = np.array(mix((16, 48, 24), (40, 56, 16), 0.5 + 0.35 * tone), np.float32)
    lit = np.array(mix((96, 164, 66), (156, 176, 60), 0.4 + 0.35 * tone), np.float32)
    t = np.clip(lam * (0.55 + 0.45 * occl) * (0.8 + 0.4 * leaves), 0, 1)[..., None] ** 1.2
    col = dark * (1 - t) + lit * t
    sparkle = np.clip((twigs - 0.78) * 6, 0, 1) * np.clip((lam - 0.6) * 3, 0, 1)
    col += sparkle[..., None] * np.array([60, 60, 30], np.float32)

    crown = pygame.Surface((w, w), pygame.SRCALPHA)
    pygame.surfarray.pixels3d(crown)[:] = np.clip(col, 0, 255).astype(np.uint8)
    alpha = np.clip(edge * r * 0.6, 0, 1) * config.TREE_CROWN_ALPHA
    pygame.surfarray.pixels_alpha(crown)[:] = alpha.astype(np.uint8)
    return crown.convert_alpha(), c



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
    blurred = sum(blur_array(water_mask, f) for f in (8, 16, 32, 64)) / 4
    depth = np.clip((blurred - 0.5) * 1.6, 0, 1) * water_mask
    mud = np.clip((LAKE_SHORE_WIDTH - edge) / 4, 0, 1)[..., None]
    col[:] = col * (1 - mud) + np.array(LAKE_MUD, np.float32) * mud * (0.85 + 0.3 * ripple[..., None])
    t = (depth ** 0.8)[..., None]
    water = (np.array(LAKE_SHALLOW, np.float32) * (1 - t) + np.array(LAKE_DEEP, np.float32) * t)
    water *= 0.92 + 0.16 * ripple[..., None]
    wet = np.clip(0.5 - edge, 0, 1)[..., None]
    col[:] = col * (1 - wet) + water * wet


def build_background(w, h, obstacles, lakes=()):
    rng = np.random.default_rng(7)
    rnd = random.Random(7)

    def blotch(cell):
        return value_noise(rng, w, h, cell)

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
        g = mix((26, 48, 32), (70, 104, 60), rnd.random() ** 1.5)
        for _ in range(rnd.randint(2, 4)):
            a = -math.pi / 2 + rnd.uniform(-0.7, 0.7)
            ln = rnd.uniform(3, 7)
            pygame.draw.aaline(bg, g, (bx, by), (bx + math.cos(a) * ln, by + math.sin(a) * ln))
    for _ in range(w * h // 9000):
        px, py = rnd.uniform(0, w), rnd.uniform(0, h)
        r = rnd.uniform(1.5, 3.5)
        if not dry(px, py, 0):
            continue
        stone = mix((58, 58, 54), (98, 94, 86), rnd.random())
        pygame.draw.ellipse(bg, (18, 24, 20), (px - r + 1, py - r * 0.7 + 1, r * 2, r * 1.4))
        pygame.draw.ellipse(bg, stone, (px - r, py - r * 0.7, r * 2, r * 1.4))
    for _ in range(w * h // 45000):
        px, py = rnd.uniform(0, w), rnd.uniform(0, h)
        if not dry(px, py):
            continue
        petal = rnd.choice(((214, 206, 150), (196, 180, 220), (220, 220, 220)))
        for k in range(5):
            a = k * 2 * math.pi / 5
            pygame.draw.circle(bg, mix(petal, (40, 50, 40), 0.45),
                               (px + math.cos(a) * 1.6, py + math.sin(a) * 1.6), 1.2)
        pygame.draw.circle(bg, (230, 190, 70), (px, py), 1)

    for obs in obstacles:
        if isinstance(obs, Tree):
            _draw_tree_ground(bg, obs, rnd)
        else:
            _draw_rock(bg, obs.rect, rng, rnd)
    return bg.convert()
