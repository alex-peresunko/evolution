"""Pygame renderer: procedurally drawn, pre-rotated, cached sprites.

Only game.py imports this module, so the simulation library stays pygame-free.
Every sprite is drawn once at SS× resolution and smoothscaled down (anti-aliasing),
then cached by quantized (size, health, walk frame, angle); a frame is mostly blits.
"""
import math
import random
import time
from itertools import islice

import numpy as np
import pygame
import pygame.gfxdraw

from . import config

SS = 4  # supersampling factor for sprite art
FOOD_DRAW_RADIUS = 8
FOOD_VARIANTS = 6
FOOD_LEVELS = 8
HEALTH_LEVELS = 4
WALK_FRAMES = 4
PULSE_LEVELS = 4

HERBIVORE_PALETTE = {
    "rim": (34, 100, 68), "core": (150, 228, 140), "leg": (28, 78, 54), "spot": (196, 242, 178),
    "tail": (238, 246, 224), "tail_rim": (168, 198, 158), "ear_in": (238, 158, 170),
    "eye": (255, 255, 255), "pupil": (15, 15, 20), "nose": (236, 116, 138),
}
CARNIVORE_PALETTE = {
    "rim": (108, 30, 24), "core": (242, 124, 62), "leg": (84, 25, 20), "stripe": (58, 18, 14),
    "claw": (236, 226, 210), "snout": (232, 172, 124), "eye": (255, 212, 60),
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


# =============================================================================
# Creature art (facing +x; u = body radius in supersampled pixels)
# =============================================================================
def _draw_herbivore(s, cx, cy, u, p, frame):
    swing = math.sin(frame * math.pi / 2) * 0.16 * u
    for fx, side, sgn in ((0.45, 1, 1), (0.45, -1, -1), (-0.45, 1, -1), (-0.45, -1, 1)):
        _ellipse(s, p["leg"], cx + fx * u + sgn * swing, cy + side * 0.72 * u, 0.22 * u, 0.14 * u)
    _shaded_ellipse(s, p["tail_rim"], p["tail"], cx - 1.0 * u, cy, 0.28 * u, 0.28 * u, steps=4)
    _shaded_ellipse(s, p["rim"], p["core"], cx - 0.1 * u, cy, 0.95 * u, 0.78 * u, steps=7,
                    hx=-0.05 * u)
    for sx, sy, r in ((-0.38, -0.3, 0.12), (-0.08, 0.34, 0.1), (-0.6, 0.12, 0.09), (0.12, -0.14, 0.08)):
        _ellipse(s, p["spot"], cx + sx * u, cy + sy * u, r * u, r * u)
    for side in (1, -1):
        _ellipse(s, p["rim"], cx + 0.32 * u, cy + side * 0.45 * u, 0.42 * u, 0.17 * u, rot=-side * 0.6)
        _ellipse(s, p["ear_in"], cx + 0.3 * u, cy + side * 0.46 * u, 0.28 * u, 0.08 * u,
                 rot=-side * 0.6)
    _shaded_ellipse(s, p["rim"], p["core"], cx + 0.75 * u, cy, 0.55 * u, 0.5 * u, steps=6,
                    hx=0.05 * u)
    for side in (1, -1):
        ey = cy + side * 0.26 * u
        _ellipse(s, p["eye"], cx + 0.95 * u, ey, 0.15 * u, 0.15 * u)
        _ellipse(s, p["pupil"], cx + 1.0 * u, ey, 0.09 * u, 0.09 * u)
        _ellipse(s, p["eye"], cx + 1.03 * u, ey - 0.04 * u, 0.035 * u, 0.035 * u)
    _ellipse(s, p["nose"], cx + 1.27 * u, cy, 0.08 * u, 0.1 * u)


def _draw_carnivore(s, cx, cy, u, p, frame):
    ph = frame * math.pi / 2
    swing = math.sin(ph) * 0.2 * u
    for fx, side, sgn in ((0.5, 1, 1), (0.5, -1, -1), (-0.5, 1, -1), (-0.5, -1, 1)):
        lx, ly = cx + fx * u + sgn * swing, cy + side * 0.55 * u
        _ellipse(s, p["leg"], lx, ly, 0.25 * u, 0.11 * u, rot=side * 0.4)
        _ellipse(s, p["claw"], lx + 0.2 * u, ly + side * 0.09 * u, 0.05 * u, 0.05 * u)
    sway = math.sin(ph + 1.0)
    n = 10
    for i in range(n):
        t = i / (n - 1)
        r = 0.22 * (1 - 0.7 * t)
        _ellipse(s, p["stripe"] if i >= n - 2 else p["rim"],
                 cx + (-0.85 - 1.1 * t) * u, cy + sway * 0.35 * t * t * u, r * u, r * u)
    _shaded_ellipse(s, p["rim"], p["core"], cx - 0.1 * u, cy, 1.05 * u, 0.55 * u, steps=7)
    for sx in (-0.7, -0.42, -0.14, 0.14):
        _ellipse(s, p["stripe"], cx + sx * u, cy, 0.055 * u, 0.4 * u * (1 - abs(sx + 0.28) * 0.5))
    for side in (1, -1):
        pygame.draw.polygon(s, p["rim"], [
            (cx + 0.78 * u, cy + side * 0.16 * u), (cx + 0.5 * u, cy + side * 0.5 * u),
            (cx + 0.98 * u, cy + side * 0.32 * u),
        ])
    _shaded_ellipse(s, p["rim"], p["core"], cx + 0.95 * u, cy, 0.46 * u, 0.38 * u, steps=5)
    pygame.draw.polygon(s, p["snout"], [
        (cx + 1.12 * u, cy - 0.24 * u), (cx + 1.6 * u, cy - 0.09 * u), (cx + 1.66 * u, cy),
        (cx + 1.6 * u, cy + 0.09 * u), (cx + 1.12 * u, cy + 0.24 * u),
    ])
    for side in (1, -1):
        pygame.draw.polygon(s, p["fang"], [
            (cx + 1.36 * u, cy + side * 0.15 * u), (cx + 1.5 * u, cy + side * 0.13 * u),
            (cx + 1.45 * u, cy + side * 0.24 * u),
        ])
        ey = cy + side * 0.2 * u
        _ellipse(s, p["eye"], cx + 1.08 * u, ey, 0.11 * u, 0.065 * u, rot=side * 0.35)
        _ellipse(s, p["pupil"], cx + 1.1 * u, ey, 0.025 * u, 0.06 * u, rot=side * 0.35)
    _ellipse(s, p["nose"], cx + 1.62 * u, cy, 0.06 * u, 0.08 * u)


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


def _build_background(w, h, obstacles):
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
    bg = pygame.surfarray.make_surface(np.clip(col, 0, 255).astype(np.uint8))

    for _ in range(w * h // 700):
        bx, by = rnd.uniform(0, w), rnd.uniform(0, h)
        g = _mix((26, 48, 32), (70, 104, 60), rnd.random() ** 1.5)
        for _ in range(rnd.randint(2, 4)):
            a = -math.pi / 2 + rnd.uniform(-0.7, 0.7)
            ln = rnd.uniform(3, 7)
            pygame.draw.aaline(bg, g, (bx, by), (bx + math.cos(a) * ln, by + math.sin(a) * ln))
    for _ in range(w * h // 9000):
        px, py = rnd.uniform(0, w), rnd.uniform(0, h)
        r = rnd.uniform(1.5, 3.5)
        stone = _mix((58, 58, 54), (98, 94, 86), rnd.random())
        pygame.draw.ellipse(bg, (18, 24, 20), (px - r + 1, py - r * 0.7 + 1, r * 2, r * 1.4))
        pygame.draw.ellipse(bg, stone, (px - r, py - r * 0.7, r * 2, r * 1.4))
    for _ in range(w * h // 45000):
        px, py = rnd.uniform(0, w), rnd.uniform(0, h)
        petal = rnd.choice(((214, 206, 150), (196, 180, 220), (220, 220, 220)))
        for k in range(5):
            a = k * 2 * math.pi / 5
            pygame.draw.circle(bg, _mix(petal, (40, 50, 40), 0.45),
                               (px + math.cos(a) * 1.6, py + math.sin(a) * 1.6), 1.2)
        pygame.draw.circle(bg, (230, 190, 70), (px, py), 1)

    for obs in obstacles:
        _draw_rock(bg, obs.rect, rng, rnd)
    return bg.convert()


# =============================================================================
# Renderer
# =============================================================================
class Renderer:
    def __init__(self, world_w, world_h, obstacles):
        self.background = _build_background(world_w, world_h, obstacles)
        self.minimap_bg = pygame.transform.smoothscale(
            self.background, (config.MINIMAP_WIDTH, config.MINIMAP_HEIGHT))
        self.frame = 0
        self._base = {}
        self._rotated = {}
        self._glows = {}
        self._bubbles = {}
        self._phase = {}
        self._food = [[self._build_food(v, lvl) for lvl in range(FOOD_LEVELS)]
                      for v in range(FOOD_VARIANTS)]

    # --- sprite caches -----------------------------------------------------
    def _creature_sprite(self, carn, sb, hl, frame, ai):
        """Rotated sprite with its drop shadow baked in, plus the blit offset from the
        creature's position to the sprite's top-left corner."""
        key = (carn, sb, hl, frame, ai)
        if len(self._rotated) >= config.SPRITE_CACHE_LIMIT:
            for k in list(islice(self._rotated, config.SPRITE_CACHE_LIMIT // 2)):
                del self._rotated[k]
        base_key = key[:4]
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
    def _build_creature(carn, sb, hl, frame):
        """Unrotated sprite facing +x, its shadow silhouette, and the x offset of the
        sprite centre from the creature origin."""
        x0, x1, y_half = (-2.04, 1.74, 0.8) if carn else (-1.32, 1.4, 0.9)
        w = math.ceil((x1 - x0) * sb) + 2
        h = math.ceil(2 * y_half * sb) + 2
        ox = 1 - x0 * sb
        big = pygame.Surface((w * SS, h * SS), pygame.SRCALPHA)
        pal = _faded_palette(CARNIVORE_PALETTE if carn else HERBIVORE_PALETTE, hl)
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

    @staticmethod
    def _build_food(variant, level):
        rnd = random.Random(variant * 97 + 3)
        fresh = 1 - level / (FOOD_LEVELS - 1)
        w = FOOD_DRAW_RADIUS * 2 + 2
        big = pygame.Surface((w * SS, w * SS), pygame.SRCALPHA)
        c, u = w * SS / 2, FOOD_DRAW_RADIUS * SS
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
    def draw_world(self, screen, sim_state):
        self.frame += 1
        cam_x, cam_y = sim_state["cam_x"], sim_state["cam_y"]
        sw, sh = screen.get_size()
        screen.blit(self.background, (0, 0), (int(cam_x), int(cam_y), sw, sh))

        fo = FOOD_DRAW_RADIUS + 1
        lifetime_scale = FOOD_LEVELS / config.FOOD_MAX_LIFETIME
        food_blits = []
        for f in sim_state["food_items"]:
            x, y = f.pos.x - cam_x, f.pos.y - cam_y
            if -fo <= x <= sw + fo and -fo <= y <= sh + fo:
                lvl = min(FOOD_LEVELS - 1, int(f.lifetime * lifetime_scale))
                food_blits.append((self._food[f.id % FOOD_VARIANTS][lvl], (x - fo, y - fo)))
        screen.blits(food_blits, doreturn=False)

        selected = sim_state.get("selected_creature")
        visible = []
        for c in sim_state["creatures"] + sim_state["carnivores"]:
            x, y = c.pos.x - cam_x, c.pos.y - cam_y
            r = c.size * 2.2 + 8
            if -r <= x <= sw + r and -r <= y <= sh + r:
                visible.append((c, x, y))

        now = time.time()
        pulse = int((math.sin(self.frame * 0.08) + 1) * 0.5 * (PULSE_LEVELS - 0.01))
        steps = config.SPRITE_ANGLE_STEPS
        angle_scale = steps / (2 * math.pi)
        rit = config.RESPAWN_IMMUNITY_TICKS
        rotated, phases = self._rotated, self._phase
        sprites, bubbles = [], []
        for c, x, y in visible:
            sb = int(c.size + 0.5)
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

            key = (c.is_carnivore, sb, hl, frame, int(c.angle * angle_scale + 0.5) % steps)
            entry = rotated.get(key)
            if entry is None:
                entry = self._creature_sprite(*key)
            spr, ox, oy = entry

            glow_color = None
            if c.is_best:
                glow_color = config.COLOR_BEST_CREATURE
            elif c.birth_time and now - c.birth_time < 3:
                glow_color = config.COLOR_BIRTH
            if glow_color:
                g = self._glow(sb, glow_color, pulse)
                gw = g.get_width() / 2
                screen.blit(g, (x - gw, y - gw), special_flags=pygame.BLEND_RGB_ADD)

            sprites.append((spr, (x + ox, y + oy)))

            if c.respawn_immunity_ticks > 0 and c.is_carnivore:
                lvl = min(PULSE_LEVELS - 1, int(c.respawn_immunity_ticks / rit * PULSE_LEVELS))
                b = self._bubble(sb, lvl)
                bw = b.get_width() / 2
                bubbles.append((b, (x - bw, y - bw)))
        screen.blits(sprites, doreturn=False)
        screen.blits(bubbles, doreturn=False)

        for c, x, y in visible:
            if c is selected:
                self._draw_selection(screen, c, x, y)
            elif c.is_best:
                self._draw_whiskers(screen, c, x, y)

        if self.frame % 600 == 0:
            alive = {c.id for c in sim_state["creatures"]} | {c.id for c in sim_state["carnivores"]}
            self._phase = {k: v for k, v in self._phase.items() if k in alive}

    def _draw_whiskers(self, screen, c, x, y):
        color = config.COLOR_CARNIVORE_WHISKER if c.is_carnivore else config.COLOR_WHISKER
        for i, a in enumerate(c.whisker_angles):
            ex = x + math.cos(c.angle + a) * c.whiskers[i]
            ey = y + math.sin(c.angle + a) * c.whiskers[i]
            pygame.draw.aaline(screen, color, (x, y), (ex, ey))
            pygame.draw.circle(screen, color, (ex, ey), 2)

    def _draw_selection(self, screen, c, x, y):
        sd, sa = c.sight_distance, c.sight_angle
        n = max(8, int(sa * 10))
        arc = [(int(x + math.cos(c.angle - sa / 2 + sa * i / n) * sd),
                int(y + math.sin(c.angle - sa / 2 + sa * i / n) * sd)) for i in range(n + 1)]
        pts = arc if sa >= 2 * math.pi - 1e-3 else [(int(x), int(y))] + arc
        pygame.gfxdraw.filled_polygon(screen, pts, (255, 255, 255, 18))
        pygame.gfxdraw.aapolygon(screen, pts, (255, 255, 255, 70))
        self._draw_whiskers(screen, c, x, y)
        r = c.size * 1.6 + 7
        rect = pygame.Rect(0, 0, r * 2, r * 2)
        rect.center = (int(x), int(y))
        dashes = 10
        for k in range(dashes):
            a0 = self.frame * 0.04 + k * 2 * math.pi / dashes
            pygame.draw.arc(screen, (255, 255, 255), rect, a0, a0 + 0.35, 2)
