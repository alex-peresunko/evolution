"""The static background (grass, lakes, rocks, tree ground) and tree crowns."""
import math
import random

import numpy as np
import pygame
import pygame.gfxdraw

from ... import config
from ...world.terrain import Tree
from .shapes import blur, shaded_ellipse, value_noise
from .style import (
    LAKE_BED,
    LAKE_DEEP,
    LAKE_MUD,
    LAKE_SAND,
    LAKE_SHALLOW,
    LAKE_SHORE_WIDTH,
    SS,
    mix,
)

LIGHT = np.array([-0.5, -0.6, 0.62], np.float32)
LIGHT /= np.linalg.norm(LIGHT)


def _lambert(height):
    """Brightness of a height field (in px) under the scene's light, 0..1."""
    gx, gy = np.gradient(height)
    return np.clip((-gx * LIGHT[0] - gy * LIGHT[1] + LIGHT[2]) / np.sqrt(gx ** 2 + gy ** 2 + 1),
                   0, 1)


def _cracks(w, h, m, rnd):
    """Fractures running across a (w, h) rock: 1 on a crack, fading to 0 beside it."""
    surf = pygame.Surface((w, h))
    for _ in range(rnd.randint(2, 3)):
        a = rnd.uniform(0, 2 * math.pi)
        x, y = w / 2 - math.cos(a) * w * 0.6, h / 2 - math.sin(a) * h * 0.6
        x, y = x + rnd.uniform(-0.25, 0.25) * w, y + rnd.uniform(-0.25, 0.25) * h
        pts = [(x, y)]
        for _ in range(14):
            a += rnd.uniform(-0.55, 0.55)
            x, y = x + math.cos(a) * m * 0.11, y + math.sin(a) * m * 0.11
            pts.append((x, y))
            if rnd.random() < 0.2:   # a side branch
                b = a + rnd.choice((-1, 1)) * rnd.uniform(0.6, 1.2)
                pygame.draw.aaline(surf, (255, 255, 255), (x, y),
                                   (x + math.cos(b) * m * 0.16, y + math.sin(b) * m * 0.16))
        pygame.draw.lines(surf, (255, 255, 255), False, pts, 2)
    return pygame.surfarray.array_red(blur(surf, 2)).astype(np.float32) / 255


def _draw_rock(bg, rect, rng, rnd):
    """A boulder filling `rect`: a block broken into flat facets, bedded, cracked and weathered,
    with rubble and grass at its foot."""
    x, y, w, h = int(rect.x), int(rect.y), int(rect.width), int(rect.height)
    m = min(w, h)
    rad = m * 0.2

    def noise(cell):
        return value_noise(rng, w, h, cell)

    px = (np.arange(w, dtype=np.float32) + 0.5 - w / 2)[:, None]
    py = (np.arange(h, dtype=np.float32) + 0.5 - h / 2)[None, :]
    qx = np.abs(px) - (w / 2 - rad - 1)
    qy = np.abs(py) - (h / 2 - rad - 1)
    d = (np.sqrt(np.maximum(qx, 0) ** 2 + np.maximum(qy, 0) ** 2)
         + np.minimum(np.maximum(qx, qy), 0) - rad)
    warp = (noise(22) - 0.5) * m * 0.1 + (noise(7) - 0.5) * m * 0.035
    height = (-d + warp) * 1.6
    n = rnd.randint(7, 10)
    off = rnd.uniform(0, 2 * math.pi)
    for i in range(n):   # each plane breaks off a corner or a side
        a = off + i * 2 * math.pi / n + rnd.uniform(-0.3, 0.3)
        ca, sa = math.cos(a), math.sin(a)
        reach = (abs(ca) * w + abs(sa) * h) / 2 * rnd.uniform(0.78, 0.98)
        height = np.minimum(height, (reach - px * ca - py * sa) * rnd.uniform(0.45, 1.3)
                            + warp * 0.5)
    top = m * rnd.uniform(0.2, 0.3)
    height = np.minimum(height, top + (noise(18) - 0.5) * m * 0.1)
    edge = height.copy()

    bed = rnd.uniform(0, math.pi)   # bedding planes show as parallel grooves
    along = (px * math.cos(bed) + py * math.sin(bed) + (noise(30) - 0.5) * m * 0.3) / (m * 0.15)
    grooves = np.clip(1 - np.abs(along % 1 - 0.5) * 14, 0, 1) * np.clip((noise(26) - 0.45) * 3, 0, 1)
    cracks = _cracks(w, h, m, rnd)
    mottle = noise(9)
    height = (height - 1.3 * grooves - 3.5 * cracks
              + 3.0 * (mottle - 0.5) + 1.4 * (noise(3) - 0.5))
    lam = _lambert(height)
    foot = np.clip(edge / (m * 0.07), 0, 1) ** 0.5   # darker where the rock meets the ground

    tone = rnd.uniform(-10, 10)
    base = np.array([112 + tone, 106 + tone, 98 + tone * 0.6], np.float32)
    warm = (noise(34) - 0.5)[..., None] * np.array([16, 6, -8], np.float32)
    shade = (0.22 + 1.0 * lam) * (0.8 + 0.4 * mottle) * (0.5 + 0.5 * foot)
    shade *= (1 - 0.3 * grooves) * (1 - 0.6 * cracks)
    col = (base[None, None, :] + warm) * shade[..., None]
    grain = rng.random((w, h), dtype=np.float32)
    col *= np.where(grain < 0.08, 0.8, np.where(grain > 0.95, 1.14, 1.0))[..., None]
    lichen = np.clip((noise(6) - 0.68) * 6, 0, 1) * np.clip((edge - top * 0.5) / 4, 0, 1)
    col += lichen[..., None] * 0.4 * (np.array([164, 160, 118], np.float32) * (0.4 + 0.7 * lam)[..., None] - col)
    moss = np.clip((noise(16) - 0.56) * 4, 0, 1) * np.clip((0.7 - lam) * 3.5, 0, 1)
    col += moss[..., None] * 0.85 * (np.array([58, 98, 46], np.float32) * (0.45 + 0.9 * lam)[..., None] - col)

    rock = pygame.Surface((w, h), pygame.SRCALPHA)
    pygame.surfarray.pixels3d(rock)[:] = np.clip(col, 0, 255).astype(np.uint8)
    pygame.surfarray.pixels_alpha(rock)[:] = (np.clip(edge + 0.5, 0, 1) * 255).astype(np.uint8)

    pad = 20
    shadow = pygame.Surface((w + pad * 2, h + pad * 2), pygame.SRCALPHA)
    shadow.blit(rock, (pad, pad))
    shadow.fill((0, 0, 0, 255), special_flags=pygame.BLEND_RGBA_MULT)
    shadow.fill((255, 255, 255, 165), special_flags=pygame.BLEND_RGBA_MULT)
    bg.blit(blur(shadow, 3), (x - pad + 1, y - pad + 2))
    bg.blit(blur(shadow, 7), (x - pad + top * 0.3, y - pad + top * 0.38))
    bg.blit(rock, (x, y))

    cx, cy = x + w / 2, y + h / 2
    for _ in range(rnd.randint(7, 12)):   # rubble that has fallen off
        a = rnd.uniform(0, 2 * math.pi)
        ca, sa = math.cos(a), math.sin(a)
        out = 1 / max(abs(ca) / (w / 2), abs(sa) / (h / 2)) + rnd.uniform(-2, 9)
        sx, sy, r = cx + ca * out, cy + sa * out, rnd.uniform(1.5, 4.5)
        stone = mix((70, 68, 62), (128, 122, 112), rnd.random())
        pygame.draw.ellipse(bg, (10, 14, 12), (sx - r + 1, sy - r * 0.75 + 1.5, r * 2, r * 1.5))
        pygame.draw.ellipse(bg, mix(stone, (0, 0, 0), 0.45), (sx - r, sy - r * 0.75, r * 2, r * 1.5))
        pygame.draw.ellipse(bg, stone, (sx - r, sy - r * 0.75, r * 1.6, r * 1.15))
    for _ in range(rnd.randint(8, 14)):   # grass growing up against the foot
        a = rnd.uniform(0, 2 * math.pi)
        ca, sa = math.cos(a), math.sin(a)
        out = 1 / max(abs(ca) / (w / 2), abs(sa) / (h / 2)) * rnd.uniform(0.93, 1.03)
        gx, gy = cx + ca * out, cy + sa * out
        g = mix((30, 56, 34), (78, 112, 62), rnd.random() ** 1.5)
        for _ in range(rnd.randint(2, 4)):
            b = -math.pi / 2 + rnd.uniform(-0.7, 0.7)
            ln = rnd.uniform(4, 8)
            pygame.draw.aaline(bg, g, (gx, gy), (gx + math.cos(b) * ln, gy + math.sin(b) * ln))


def _draw_tree_ground(bg, tree, rnd, index):
    """Crown shadow, roots and trunk under a tree; the crown itself is drawn over creatures.
    `index` is the tree's place among the trees, which seeds its crown."""
    r = tree.crown_radius
    shadow, half = build_crown(round(r), index)
    shadow.fill((0, 0, 0, 255), special_flags=pygame.BLEND_RGBA_MULT)
    shadow.fill((255, 255, 255, 150), special_flags=pygame.BLEND_RGBA_MULT)
    bg.blit(blur(shadow, 5), (tree.x - half + r * 0.22, tree.y - half + r * 0.28))

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


def _grow(rnd, segs, tufts, x, y, a, length, thick, depth):
    steps = 3
    for k in range(steps):
        a += rnd.uniform(-0.35, 0.35)
        nx, ny = x + math.cos(a) * length / steps, y + math.sin(a) * length / steps
        segs.append((x, y, nx, ny, thick * (1 - 0.25 * k)))
        x, y = nx, ny
        tx, ty = x + rnd.uniform(-0.06, 0.06), y + rnd.uniform(-0.06, 0.06)
        d = math.hypot(tx, ty)
        tufts.append((tx, ty, min(rnd.uniform(0.16, 0.26), 1.13 - d),
                      0.9 * (1 - 0.75 * d) + rnd.uniform(-0.08, 0.08), rnd.uniform(-1, 1)))
        if depth < 1 and k < steps - 1:
            _grow(rnd, segs, tufts, x, y, a + rnd.choice((-1, 1)) * rnd.uniform(0.6, 1.1),
                  length * 0.5, thick * 0.55, depth + 1)


def _boughs(rnd):
    """A tree's branches from above, in crown radii from the trunk: wood segments
    (x0, y0, x1, y1, thickness) and the leaf tufts they carry (x, y, radius, height, tone)."""
    segs, tufts = [], []
    n = rnd.randint(7, 9)
    off = rnd.uniform(0, 2 * math.pi)
    for i in range(n):
        a = off + i * 2 * math.pi / n
        _grow(rnd, segs, tufts, 0.0, 0.0, a + rnd.uniform(-0.2, 0.2), rnd.uniform(0.68, 0.92),
              0.08, 0)
        d = rnd.uniform(0.3, 0.55)   # the inner canopy between the boughs
        tufts.append((math.cos(a + 0.4) * d, math.sin(a + 0.4) * d, rnd.uniform(0.24, 0.32),
                      0.9 * (1 - 0.75 * d) + rnd.uniform(0.0, 0.1), rnd.uniform(-1, 1)))
    tufts.append((rnd.uniform(-0.08, 0.08), rnd.uniform(-0.08, 0.08), 0.24, 0.92, rnd.uniform(-1, 1)))
    return segs, tufts


def build_crown(r, seed):
    """Crown of radius `r` px seen from above: tufts of leaves on spreading boughs, lit as a
    height field, with the boughs showing in the gaps; returns (sprite, half width). The same
    seed gives the same tree at every radius."""
    rnd = random.Random(seed)
    rng = np.random.default_rng(seed)
    segs, tufts = _boughs(rnd)
    w = math.ceil(r * 2.3) + 4
    ss = 2 if r < 110 else 1
    size, scale = w * ss, r * ss
    c = size / 2

    height = np.full((size, size), -1.0, np.float32)
    cover = np.zeros((size, size), np.float32)   # 1 at a tuft's centre, 0 at its edge
    tone = np.zeros((size, size), np.float32)
    for x, y, cr, z, tn in tufts:
        x0, x1 = max(0, int(c + (x - cr) * scale)), min(size, int(c + (x + cr) * scale) + 2)
        y0, y1 = max(0, int(c + (y - cr) * scale)), min(size, int(c + (y + cr) * scale) + 2)
        dx = (np.arange(x0, x1, dtype=np.float32) + 0.5 - c)[:, None] / scale - x
        dy = (np.arange(y0, y1, dtype=np.float32) + 0.5 - c)[None, :] / scale - y
        q = 1 - (dx * dx + dy * dy) / (cr * cr)
        h = z + cr * 0.9 * np.sqrt(np.maximum(q, 0))
        win = (q > 0) & (h > height[x0:x1, y0:y1])
        height[x0:x1, y0:y1][win] = h[win]
        tone[x0:x1, y0:y1][win] = tn
        np.maximum(cover[x0:x1, y0:y1], q, out=cover[x0:x1, y0:y1])

    sprays = value_noise(rng, size, size, max(2, int(scale * 0.07)))
    leaves = value_noise(rng, size, size, max(2, int(scale * 0.025)))
    grain = rng.random((size, size), dtype=np.float32)
    foliage = (cover > 0) & (cover + 0.55 * (sprays - 0.5) + 0.3 * (leaves - 0.5) > 0.16)

    bump = height + 0.13 * sprays + 0.07 * leaves
    lam = _lambert(bump * scale * 0.9)
    light = LIGHT
    shadow = np.zeros((size, size), np.float32)   # cast by higher tufts toward the light
    for s in (0.05, 0.1, 0.18):
        shift = (round(-light[0] * s * scale), round(-light[1] * s * scale))
        over = np.roll(bump, shift, axis=(0, 1)) - bump - 0.6 * s
        np.maximum(shadow, np.clip(over / 0.1, 0, 1), out=shadow)
    depth = np.clip((height - 0.1) / 0.75, 0, 1) ** 0.7

    variety = rnd.uniform(-1, 1)
    dark = np.array(mix((12, 38, 22), (30, 44, 14), 0.5 + 0.35 * variety), np.float32)
    lit = np.array(mix((92, 158, 62), (150, 172, 58), 0.4 + 0.35 * variety), np.float32)
    t = lam * (0.45 + 0.55 * depth) * (0.7 + 0.6 * leaves) * (1 - 0.6 * shadow)
    t = np.clip(t, 0, 1)[..., None] ** 1.15
    col = dark * (1 - t) + lit * t
    col += tone[..., None] * np.array([12, 5, -8], np.float32) * (0.4 + t)
    col *= np.where(grain < 0.14, 0.72, 1.0)[..., None]
    col += ((grain > 0.9) * lam * (1 - shadow))[..., None] * np.array([46, 50, 22], np.float32)

    wood = pygame.Surface((size, size), pygame.SRCALPHA)
    for x0, y0, x1, y1, thick in segs:
        p0, p1 = (c + x0 * scale, c + y0 * scale), (c + x1 * scale, c + y1 * scale)
        pygame.draw.line(wood, (40, 28, 18), p0, p1, max(1, round(thick * scale)))
        if thick * scale >= 3:
            pygame.draw.line(wood, (78, 58, 38), (p0[0] - 1, p0[1] - 1), (p1[0] - 1, p1[1] - 1),
                             max(1, round(thick * scale * 0.4)))
    col = np.where(foliage[..., None], col, pygame.surfarray.pixels3d(wood))
    bare = pygame.surfarray.pixels_alpha(wood) > 0
    col = np.where((foliage | bare)[..., None], col, dark)   # no black fringe when scaled down

    crown = pygame.Surface((size, size), pygame.SRCALPHA)
    pygame.surfarray.pixels3d(crown)[:] = np.clip(col, 0, 255).astype(np.uint8)
    pygame.surfarray.pixels_alpha(crown)[:] = np.where(
        foliage, config.TREE_CROWN_ALPHA, np.where(bare, 255, 0)).astype(np.uint8)
    if ss > 1:
        crown = pygame.transform.smoothscale(crown, (w, w))
    return crown.convert_alpha(), w / 2



def _shore_gaps(w, h, lakes):
    """Per pixel, the gap to the nearest shoreline (< 0 = water, inf far from any lake), and
    the pixel boxes (x0, y0, x1, y1) that hold the lakes with their shores, none overlapping."""
    edge = np.full((w, h), np.inf, np.float32)
    boxes = []
    for lake in lakes:
        m = lake.max_radius + 2 * LAKE_SHORE_WIDTH
        x0, x1 = max(0, int(lake.x - m)), min(w, int(lake.x + m) + 1)
        y0, y1 = max(0, int(lake.y - m)), min(h, int(lake.y + m) + 1)
        d = lake.shore_gap(np.arange(x0, x1, dtype=np.float32)[:, None] + 0.5,
                           np.arange(y0, y1, dtype=np.float32)[None, :] + 0.5)
        edge[x0:x1, y0:y1] = np.minimum(edge[x0:x1, y0:y1], d)
        box = [x0, y0, x1, y1]
        merged = True
        while merged:
            merged = False
            for other in boxes:
                if (box[0] < other[2] and other[0] < box[2]
                        and box[1] < other[3] and other[1] < box[3]):
                    boxes.remove(other)
                    box = [min(box[0], other[0]), min(box[1], other[1]),
                           max(box[2], other[2]), max(box[3], other[3])]
                    merged = True
                    break
        boxes.append(box)
    return edge, boxes


def _paint_lake(col, edge, rng):
    """Paint one lake into its box of background colours: a sandy shore, the bed showing
    through the shallows, deep water, wind streaks and the bank's shadow."""
    w, h = edge.shape

    def noise(cell, cell_y=None):
        return value_noise(rng, w, h, cell, cell_y)

    fine, mid = noise(4), noise(13)
    grain = rng.random((w, h), dtype=np.float32)
    pebbles = np.where(grain > 0.975, 1.35, np.where(grain < 0.03, 0.7, 1.0))[..., None]

    reach = LAKE_SHORE_WIDTH * (0.7 + 1.0 * mid)   # a ragged edge to the grass
    s = np.clip(edge / reach, 0, 1)                # 0 at the waterline, 1 where grass starts
    damp = np.clip(s * 3, 0, 1)[..., None]
    sand = (np.array(LAKE_MUD, np.float32) * (1 - damp)
            + np.array(LAKE_SAND, np.float32) * damp) * (0.82 + 0.36 * fine[..., None]) * pebbles
    beach = np.clip((1 - s) * 2.5, 0, 1)[..., None]
    col[:] = col * (1 - beach) + sand * beach

    water = edge < 0
    murk = (1 - np.exp(np.minimum(edge, 0) / 13))[..., None]   # how far the bed is hidden
    depth = np.clip(1 - np.exp(np.minimum(edge, 0) / 55) + 0.2 * (noise(40) - 0.5), 0, 1)
    depth = depth[..., None]
    bed = np.array(LAKE_BED, np.float32) * (0.75 + 0.5 * fine[..., None]) * pebbles
    body = np.array(LAKE_SHALLOW, np.float32) * (1 - depth) + np.array(LAKE_DEEP, np.float32) * depth
    wet = bed * (1 - murk) + body * murk
    streaks = 0.65 * noise(26, 5) + 0.35 * noise(11, 3)
    wet *= (1 - 0.14 * np.clip((0.42 - streaks) * 4, 0, 1))[..., None]
    glint = np.clip((streaks - 0.63) * 5, 0, 1) * (0.5 + 0.5 * mid)
    wet += (glint[..., None] * murk) * np.array([16, 24, 30], np.float32)
    lap = np.clip(1 - np.abs(edge + 1.2) / 1.4, 0, 1) * (0.4 + 0.6 * fine)
    wet += lap[..., None] * np.array([44, 50, 44], np.float32)
    for shift, dark in (((2, 3), 0.2), ((5, 6), 0.14)):   # the bank toward the light shades it
        wet *= (1 - dark * np.roll(~water, shift, axis=(0, 1)))[..., None]
    col[water] = wet[water]


def _lily_pad(bg, x, y, r, rnd):
    a = rnd.uniform(0, 2 * math.pi)
    pts = [(x, y)] + [(x + math.cos(a + 0.3 + (2 * math.pi - 0.6) * k / 10) * r,
                       y + math.sin(a + 0.3 + (2 * math.pi - 0.6) * k / 10) * r) for k in range(11)]
    green = mix((34, 84, 44), (70, 124, 58), rnd.random())
    pygame.gfxdraw.filled_polygon(bg, [(px + 1, py + 2) for px, py in pts], (4, 16, 26, 130))
    pygame.gfxdraw.filled_polygon(bg, pts, green)
    pygame.gfxdraw.aapolygon(bg, pts, mix(green, (0, 0, 0), 0.4))
    pygame.draw.line(bg, mix(green, (200, 230, 160), 0.3), (x, y),
                     (x - math.cos(a) * r * 0.8, y - math.sin(a) * r * 0.8))
    if rnd.random() < 0.12:
        petal = rnd.choice(((236, 226, 232), (232, 170, 196)))
        for k in range(6):
            b = k * math.pi / 3
            pygame.draw.circle(bg, petal, (x + math.cos(b) * 1.4, y + math.sin(b) * 1.4), 1.2)
        pygame.draw.circle(bg, (236, 196, 70), (x, y), 1)


def _decorate_lakes(bg, lakes, edge, rnd):
    """Reed beds along the shores and lily pads in the shallows."""
    w, h = edge.shape

    def gap(x, y):
        return edge[int(x), int(y)] if 0 <= x < w and 0 <= y < h else math.inf

    for lake in lakes:
        for _ in range(int(lake.radius / 16)):
            a = rnd.uniform(0, 2 * math.pi)
            for _ in range(rnd.randint(1, 4)):
                b = a + rnd.uniform(-0.12, 0.12)
                out = lake.shore(b) - rnd.uniform(9, 32)
                x, y = lake.x + math.cos(b) * out, lake.y + math.sin(b) * out
                if -42 < gap(x, y) < -8:
                    _lily_pad(bg, x, y, rnd.uniform(2.5, 4.6), rnd)
        for _ in range(int(lake.radius / 9)):
            a = rnd.uniform(0, 2 * math.pi)
            for _ in range(rnd.randint(4, 10)):
                b = a + rnd.uniform(-0.07, 0.07)
                out = lake.shore(b) + rnd.uniform(-7, 3)
                x, y = lake.x + math.cos(b) * out, lake.y + math.sin(b) * out
                if not -8 < gap(x, y) < 4:
                    continue
                lean, ln = rnd.uniform(-0.45, 0.45), rnd.uniform(7, 14)
                tip = (x + math.sin(lean) * ln, y - math.cos(lean) * ln)
                pygame.draw.aaline(bg, (8, 18, 16), (x + 1, y + 1), (x + 1 + ln * 0.5, y + 1 + ln * 0.3))
                pygame.draw.aaline(bg, mix((38, 64, 34), (104, 124, 62), rnd.random()), (x, y), tip)
                if rnd.random() < 0.3:   # a seed head
                    pygame.draw.line(bg, (92, 62, 38), tip,
                                     (tip[0] - math.sin(lean) * 3.5, tip[1] + math.cos(lean) * 3.5), 2)


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
    edge, lake_boxes = _shore_gaps(w, h, lakes)
    for x0, y0, x1, y1 in lake_boxes:
        _paint_lake(col[x0:x1, y0:y1], edge[x0:x1, y0:y1], rng)
    bg = pygame.surfarray.make_surface(np.clip(col, 0, 255).astype(np.uint8))

    def dry(x, y, margin=LAKE_SHORE_WIDTH):
        return edge[min(w - 1, int(x)), min(h - 1, int(y))] >= margin

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

    _decorate_lakes(bg, lakes, edge, rnd)
    trees = 0
    for obs in obstacles:
        if isinstance(obs, Tree):
            _draw_tree_ground(bg, obs, rnd, trees)
            trees += 1
        else:
            _draw_rock(bg, obs.rect, rng, rnd)
    return bg.convert()
