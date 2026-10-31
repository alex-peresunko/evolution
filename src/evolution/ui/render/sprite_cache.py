"""Sprites built on first use and kept, keyed by everything that changes their pixels."""
import math
from itertools import islice

import pygame

from ... import config
from .creature_art import build_creature
from .item_art import build_bubble, build_carcass, build_food, build_glow
from .style import FOOD_LEVELS, FOOD_VARIANTS
from .terrain_art import build_crown


class SpriteCache:
    def __init__(self, tree_count):
        self._base = {}
        self.rotated = {}
        self._glows = {}
        self._bubbles = {}
        self._carcasses = {}
        self._food = {}    # draw radius -> [variant][level] sprites
        self._crowns = {}  # (tree index, on-screen radius) -> (sprite, half width)
        self._tree_count = tree_count

    def creature(self, carn, sb, hl, frame, cb, ai):
        """Rotated sprite (angle bucket `ai`) with its drop shadow baked in, plus the blit
        offset from the creature's position to the sprite's top-left corner. Callers look up
        `rotated` first; this builds a missing entry."""
        key = (carn, sb, hl, frame, cb, ai)
        if len(self.rotated) >= config.SPRITE_CACHE_LIMIT:
            for k in list(islice(self.rotated, config.SPRITE_CACHE_LIMIT // 2)):
                del self.rotated[k]
            self._base.clear()  # zooming in builds large bases; they are cheap to redraw
        base_key = key[:5]
        entry = self._base.get(base_key)
        if entry is None:
            entry = self._base[base_key] = build_creature(*base_key)
        base, shadow_base, center_dx = entry
        theta = ai * 2 * math.pi / config.SPRITE_ANGLE_STEPS
        rot = pygame.transform.rotozoom(base, -math.degrees(theta), 1)
        shadow = pygame.transform.rotozoom(shadow_base, -math.degrees(theta), 1)
        rw, rh = rot.get_size()
        sx, sy = int(sb * 0.3) + 1, int(sb * 0.4) + 1
        comp = pygame.Surface((rw + sx, rh + sy), pygame.SRCALPHA)
        comp.blit(shadow, (sx, sy))
        comp.blit(rot, (0, 0))
        val = self.rotated[key] = (comp.convert_alpha(),
                                   center_dx * math.cos(theta) - rw / 2,
                                   center_dx * math.sin(theta) - rh / 2)
        return val

    def glow(self, sb, color, level):
        key = (sb, color, level)
        g = self._glows.get(key)
        if g is None:
            g = self._glows[key] = build_glow(sb, color, level)
        return g

    def bubble(self, sb, level):
        key = (sb, level)
        b = self._bubbles.get(key)
        if b is None:
            b = self._bubbles[key] = build_bubble(sb, level)
        return b

    def carcass(self, sb, carn, level):
        key = (sb, carn, level)
        spr = self._carcasses.get(key)
        if spr is None:
            spr = self._carcasses[key] = build_carcass(sb, carn, level)
        return spr

    def food(self, radius):
        sprites = self._food.get(radius)
        if sprites is None:
            sprites = self._food[radius] = [
                [build_food(v, lvl, radius) for lvl in range(FOOD_LEVELS)]
                for v in range(FOOD_VARIANTS)]
        return sprites

    def trim_crowns(self):
        """Drop the crowns built for earlier zoom levels; call once per frame."""
        if len(self._crowns) > 4 * self._tree_count:
            self._crowns.clear()

    def crown(self, index, radius):
        entry = self._crowns.get((index, radius))
        if entry is None:
            entry = self._crowns[(index, radius)] = build_crown(radius, index)
        return entry
