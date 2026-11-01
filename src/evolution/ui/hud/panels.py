"""Translucent rounded HUD panels and text blocks."""
import pygame

from ... import config

_PANELS = {}
_fade_layer = None
_FONTS = {}


def font(size):
    f = _FONTS.get(size)
    if f is None:
        f = _FONTS[size] = pygame.font.Font(None, size)
    return f


def draw_panel(screen, rect):
    """Rounded HUD panel background, cached per size; opaque, since draw_faded fades the
    whole panel."""
    key = (rect.width, rect.height)
    panel = _PANELS.get(key)
    if panel is None:
        panel = pygame.Surface(key, pygame.SRCALPHA)
        pygame.draw.rect(panel, (12, 18, 16), panel.get_rect(), border_radius=8)
        pygame.draw.rect(panel, (255, 255, 255, 45), panel.get_rect(), width=1, border_radius=8)
        _PANELS[key] = panel
    screen.blit(panel, rect.topleft)


def draw_faded(screen, rect, alpha, draw, *args):
    """Run draw(surface, *args), which paints in screen coordinates inside rect, on a reusable
    transparent layer, then blit that rect at alpha, so panel, text and lines fade together."""
    global _fade_layer
    layer = _fade_layer
    if layer is None or layer.get_size() != screen.get_size():
        layer = _fade_layer = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
    layer.fill((0, 0, 0, 0), rect)
    layer.set_clip(rect)
    draw(layer, *args)
    layer.set_clip(None)
    layer.set_alpha(alpha)
    screen.blit(layer, rect.topleft, area=rect)


def draw_text_block(screen, text_font, lines, x, y, title=None, right=None):
    """right: the panel's right edge on screen; when given, x is ignored."""
    surfs = ([text_font.render(title, True, config.COLOR_BEST_CREATURE)] if title else [])
    surfs += [text_font.render(line, True, config.COLOR_TEXT) for line in lines]
    line_h = text_font.get_linesize()
    pad = 6
    text_w = max(s.get_width() for s in surfs)
    if right is not None:
        x = right - pad - text_w
    rect = pygame.Rect(x - pad, y - pad, text_w + pad * 2, line_h * len(surfs) + pad * 2)

    def paint(surface, y=y):
        draw_panel(surface, rect)
        for surf in surfs:
            surface.blit(surf, (x, y))
            y += line_h

    draw_faded(screen, rect, config.HUD_ALPHA, paint)
