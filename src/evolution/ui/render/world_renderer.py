"""Draws the camera's view of the world: background, food, carcasses, creatures, tree crowns,
night and the selected creature's senses."""
import math
import time

import pygame
import pygame.gfxdraw

from ... import config
from ...world.terrain import Tree
from .sprite_cache import SpriteCache
from .style import (
    BG_CACHE_MARGIN,
    CARCASS_LEVELS,
    CROWN_GHOST_ALPHA,
    FOOD_DRAW_RADIUS,
    FOOD_LEVELS,
    FOOD_VARIANTS,
    HEALTH_LEVELS,
    HIT_FLASH_TICKS,
    NIGHT_COLOR,
    PULSE_LEVELS,
    WALK_FRAMES,
    color_bucket,
)
from .terrain_art import build_background


class WorldRenderer:
    def __init__(self, world_w, world_h, obstacles, lakes=()):
        self.background = build_background(world_w, world_h, obstacles, lakes)
        self.minimap_bg = pygame.transform.smoothscale(
            self.background, (config.MINIMAP_WIDTH, config.MINIMAP_HEIGHT))
        self.frame = 0
        self.trees = [o for o in obstacles if isinstance(o, Tree)]
        self.sprites = SpriteCache(len(self.trees))
        self._phase = {}        # creature id -> walk animation phase
        self._night = None
        self._bg_cache = None   # (zoom, world rect, scaled background of that rect)

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

    def draw(self, screen, world, view):
        """Draw the camera's view at view.zoom: positions and sprite sizes are scaled, and
        sprites are built (and cached) at their on-screen size, so zooming stays sharp."""
        self.frame += 1
        cache = self.sprites
        cam_x, cam_y = view.cam_x, view.cam_y
        z = view.zoom
        sw, sh = screen.get_size()
        self._draw_background(screen, cam_x, cam_y, z)

        food_r = max(2, round(FOOD_DRAW_RADIUS * z))
        food_sprites = cache.food(food_r)
        fo = food_r + 1
        lifetime_scale = FOOD_LEVELS / config.FOOD_MAX_LIFETIME
        food_blits = []
        for f in world.food:
            x, y = (f.pos.x - cam_x) * z, (f.pos.y - cam_y) * z
            if -fo <= x <= sw + fo and -fo <= y <= sh + fo:
                lvl = min(FOOD_LEVELS - 1, int(f.lifetime * lifetime_scale))
                food_blits.append((food_sprites[f.id % FOOD_VARIANTS][lvl], (x - fo, y - fo)))
        screen.blits(food_blits, doreturn=False)

        steps = config.SPRITE_ANGLE_STEPS
        angle_scale = steps / (2 * math.pi)
        carcass_blits = []
        for cc in world.carcasses:
            x, y = (cc.pos.x - cam_x) * z, (cc.pos.y - cam_y) * z
            r = (cc.radius * 2.6 + 3) * z
            if -r <= x <= sw + r and -r <= y <= sh + r:
                lvl = min(CARCASS_LEVELS - 1, int((1 - cc.meat / cc.max_meat) * CARCASS_LEVELS))
                spr, ox, oy = cache.carcass(cc.species.is_predator,
                                            max(2, int(cc.radius * z + 0.5)), lvl,
                                            color_bucket(cc.color),
                                            int(cc.angle * angle_scale + 0.5) % steps)
                carcass_blits.append((spr, (x + ox, y + oy)))
        screen.blits(carcass_blits, doreturn=False)

        creatures = world.creatures()
        poses = view.poses
        visible = []
        for c in creatures:
            pose = poses.get(c.id) if poses else None
            if pose is None:
                wx, wy, angle = c.pos.x, c.pos.y, c.angle
            else:
                wx, wy, angle = pose
            x, y = (wx - cam_x) * z, (wy - cam_y) * z
            r = (c.size * 2.2 + 8) * z
            if -r <= x <= sw + r and -r <= y <= sh + r:
                visible.append((c, x, y, angle))

        now = time.time()
        pulse = int((math.sin(self.frame * 0.08) + 1) * 0.5 * (PULSE_LEVELS - 0.01))
        rit = config.RESPAWN_IMMUNITY_TICKS
        tick = world.tick
        rotated, phases = cache.rotated, self._phase
        slow_motion = view.ticks_per_frame < 1
        sprites, bubbles, signallers = [], [], []
        for c, x, y, angle in visible:
            sb = int(c.size * z + 0.5)
            if sb < 2:
                sb = 2
            hl = int(c.health / c.max_health * HEALTH_LEVELS) if c.max_health else 0
            hl = 0 if hl < 0 else HEALTH_LEVELS - 1 if hl >= HEALTH_LEVELS else hl

            frame = 0
            if sb > 3:
                phase = phases.get(c.id, 0.0)
                if c.current_speed > 0.3:
                    stride = c.current_speed / (c.size * 0.5 + 2) * 0.35
                    phase += stride * view.ticks_per_frame if slow_motion else stride
                    phases[c.id] = phase
                frame = int(phase) % WALK_FRAMES

            predator = c.species.is_predator
            key = (predator, sb, hl, frame, color_bucket(c.genes["color"]),
                   int(angle * angle_scale + 0.5) % steps)
            entry = rotated.get(key)
            if entry is None:
                entry = cache.creature(*key)
            spr, ox, oy = entry

            glow_color = None
            if c.last_hit_tick is not None and tick - c.last_hit_tick < HIT_FLASH_TICKS:
                glow_color = config.COLOR_HIT
            elif c.is_best:
                glow_color = config.COLOR_BEST_CREATURE
            elif c.birth_time and now - c.birth_time < 3:
                glow_color = config.COLOR_BIRTH
            if glow_color:
                g = cache.glow(sb, glow_color, pulse)
                gw = g.get_width() / 2
                screen.blit(g, (x - gw, y - gw), special_flags=pygame.BLEND_RGB_ADD)

            sprites.append((spr, (x + ox, y + oy)))
            if c.signal > config.SIGNAL_RENDER_THRESHOLD:
                signallers.append((c, x, y))

            if c.respawn_immunity_ticks > 0 and predator:
                lvl = min(PULSE_LEVELS - 1, int(c.respawn_immunity_ticks / rit * PULSE_LEVELS))
                b = cache.bubble(sb, lvl)
                bw = b.get_width() / 2
                bubbles.append((b, (x - bw, y - bw)))
        screen.blits(sprites, doreturn=False)
        screen.blits(bubbles, doreturn=False)
        for c, x, y in signallers:
            t = ((self.frame + c.id * 7) % 30) / 30
            color = (255, 230, 120) if c.species.is_predator else (150, 230, 255)
            pygame.gfxdraw.aacircle(screen, int(x), int(y),
                                    int((c.size * 1.3 + 4 + t * 16) * z),
                                    (*color, int(220 * (1 - t) * c.signal)))

        self._draw_crowns(screen, cam_x, cam_y, z)
        for (c, _, _, _), (spr, pos) in zip(visible, sprites):
            if self._under_crown(c):
                spr.set_alpha(CROWN_GHOST_ALPHA)
                screen.blit(spr, pos)
                spr.set_alpha(255)
        self._draw_night(screen, world.light, food_blits, fo, z)

        for c, x, y, angle in visible:
            if c is view.selected:
                self._draw_selection(screen, c, x, y, angle, z)
            elif c.is_best:
                self._draw_whiskers(screen, c, x, y, angle, z)

        if self.frame % 600 == 0:
            alive = {c.id for c in creatures}
            self._phase = {k: v for k, v in self._phase.items() if k in alive}

    def _under_crown(self, c):
        x, y = c.pos.x, c.pos.y
        for t in self.trees:
            dx, dy, r = x - t.x, y - t.y, t.crown_radius + c.size
            if dx * dx + dy * dy < r * r:
                return True
        return False

    def _draw_crowns(self, screen, cam_x, cam_y, z):
        self.sprites.trim_crowns()
        sw, sh = screen.get_size()
        blits = []
        for i, t in enumerate(self.trees):
            x, y = (t.x - cam_x) * z, (t.y - cam_y) * z
            r = max(2, round(t.crown_radius * z))
            m = r * 1.2
            if -m <= x <= sw + m and -m <= y <= sh + m:
                spr, half = self.sprites.crown(i, r)
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
            g = self.sprites.glow(max(2, round(2 * z)), (40, 130, 60),
                                  min(PULSE_LEVELS - 1, level))
            gw = g.get_width() / 2
            add = pygame.BLEND_RGB_ADD
            screen.blits([(g, (x + fo - gw, y + fo - gw), None, add) for _, (x, y) in food_blits],
                         doreturn=False)

    def _draw_whiskers(self, screen, c, x, y, angle, z):
        color = (config.COLOR_CARNIVORE_WHISKER if c.species.is_predator
                 else config.COLOR_WHISKER)
        for i, a in enumerate(c.whisker_angles):
            ex = x + math.cos(angle + a) * c.whiskers[i] * z
            ey = y + math.sin(angle + a) * c.whiskers[i] * z
            pygame.draw.aaline(screen, color, (x, y), (ex, ey))
            pygame.draw.circle(screen, color, (ex, ey), 2)
            seen = c.target_whiskers[i]
            if seen is not None:  # food / prey on this ray
                seen *= z
                tx, ty = x + math.cos(angle + a) * seen, y + math.sin(angle + a) * seen
                pygame.draw.circle(screen, config.COLOR_TARGET_WHISKER, (tx, ty), 3)

    def _draw_selection(self, screen, c, x, y, angle, z):
        """Sight cone, whiskers and a rotating dashed ring."""
        sd, sa = (c.effective_sight or c.sight_distance) * z, c.sight_angle
        n = max(8, int(sa * 10))
        arc = [(int(x + math.cos(angle - sa / 2 + sa * i / n) * sd),
                int(y + math.sin(angle - sa / 2 + sa * i / n) * sd)) for i in range(n + 1)]
        pts = arc if sa >= 2 * math.pi - 1e-3 else [(int(x), int(y))] + arc
        pygame.gfxdraw.filled_polygon(screen, pts, (255, 255, 255, 18))
        pygame.gfxdraw.aapolygon(screen, pts, (255, 255, 255, 70))
        self._draw_whiskers(screen, c, x, y, angle, z)
        r = (c.size * 1.6 + 7) * z
        rect = pygame.Rect(0, 0, r * 2, r * 2)
        rect.center = (int(x), int(y))
        dashes = 10
        for k in range(dashes):
            a0 = self.frame * 0.04 + k * 2 * math.pi / dashes
            pygame.draw.arc(screen, (255, 255, 255), rect, a0, a0 + 0.35, 2)
