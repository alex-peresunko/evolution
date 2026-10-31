"""Sprite art constants, palettes and colour helpers."""
import colorsys

from ... import config

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


def mix(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))



def faded_palette(palette, health_level):
    t = (HEALTH_LEVELS - 1 - health_level) * 0.2
    out = {}
    for k, c in palette.items():
        if k in _NO_FADE:
            out[k] = c
        else:
            lum = int(0.3 * c[0] + 0.59 * c[1] + 0.11 * c[2])
            out[k] = mix(c, (lum, lum, int(lum * 0.95)), t)
    return out


def color_bucket(color_gene):
    return min(config.COLOR_BUCKETS - 1, int(color_gene * config.COLOR_BUCKETS))


def tinted_palette(palette, bucket):
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


LAKE_SHORE_WIDTH = 9
LAKE_MUD = (58, 52, 38)
LAKE_SHALLOW = (54, 104, 108)
LAKE_DEEP = (18, 46, 74)
