"""Live network diagram of the selected creature: node activations, strongest connections
(weight * activation), override badge, top input drivers and output sparklines."""
import math

import pygame

from ... import config
from ...brain.layout import OUTPUTS
from ...brain.view import (
    INPUT_LABELS,
    OUTPUT_LABELS,
    activations,
    edge_contributions,
    labels,
    network_layout,
    strong_edges,
    top_drivers,
)
from .panels import draw_panel, font

_EDGE_LAYER = {}


def activation_color(v):
    """Diverging color: blue at -1, gray at 0, orange at +1."""
    v = max(-1.0, min(1.0, float(v)))
    end = config.COLOR_BRAIN_POS if v > 0 else config.COLOR_BRAIN_NEG
    return tuple(round(z + (e - z) * abs(v)) for z, e in zip(config.COLOR_BRAIN_ZERO, end))


def panel_rect():
    w, h = config.BRAIN_PANEL_SIZE
    return pygame.Rect(10, config.SCREEN_HEIGHT - h - 10, w, h)


def draw_brain_panel(screen, c, history):
    panel = panel_rect()
    (x0, y0), (w, h) = panel.topleft, panel.size
    draw_panel(screen, panel)
    small = font(15)
    screen.blit(small.render(f"Brain - {c.species.title} #{c.id}", True,
                             config.COLOR_BEST_CREATURE), (x0 + 8, y0 + 6))

    acts = activations(c.brain, c.brain_inputs)
    in_labels = labels(INPUT_LABELS, len(acts[0]), "in")
    out_labels = labels(OUTPUT_LABELS, len(acts[-1]), "out")
    row_h, spark_h = 14, 32
    top = y0 + 26
    net_bottom = y0 + h - 8 - spark_h - 14 - 3 * row_h - 6  # room for badge, drivers, sparklines
    nodes, radii = network_layout([len(a) for a in acts], x0 + 128, top, x0 + w - 70, net_bottom)

    edges = strong_edges(edge_contributions(c.brain, acts))
    layer = _EDGE_LAYER.get((w, h))
    if layer is None:
        layer = _EDGE_LAYER[(w, h)] = pygame.Surface((w, h), pygame.SRCALPHA)
    layer.fill((0, 0, 0, 0))
    peak = max((abs(v) for *_, v in edges), default=1.0) or 1.0
    for k, i, j, v in sorted(edges, key=lambda e: abs(e[3])):
        strength = abs(v) / peak
        color = (*(config.COLOR_BRAIN_POS if v > 0 else config.COLOR_BRAIN_NEG),
                 round(50 + 205 * strength))
        (ax, ay), (bx, by) = nodes[k][i], nodes[k + 1][j]
        pygame.draw.line(layer, color, (ax - x0, ay - y0), (bx - x0, by - y0),
                         1 + round(2 * strength))
    screen.blit(layer, (x0, y0))

    for k, a in enumerate(acts):
        for (nx, ny), v in zip(nodes[k], a):
            pygame.draw.circle(screen, activation_color(v), (nx, ny), radii[k])
            if radii[k] >= 3:
                pygame.draw.circle(screen, (200, 200, 200), (nx, ny), radii[k], 1)
    in_gap = (net_bottom - top) / len(acts[0])
    for idx in range(0, len(acts[0]), math.ceil(11 / in_gap)):  # skip labels that would overlap
        ny, v = nodes[0][idx][1], acts[0][idx]
        screen.blit(small.render(in_labels[idx], True, config.COLOR_TEXT), (x0 + 8, ny - 5))
        screen.blit(small.render(f"{v:+.2f}", True, activation_color(v)), (x0 + 82, ny - 5))
    out_gap = (net_bottom - top) / len(acts[-1])
    for idx in range(0, len(acts[-1]), math.ceil(11 / out_gap)):
        (nx, ny), v = nodes[-1][idx], acts[-1][idx]
        name = small.render(out_labels[idx], True, config.COLOR_TEXT)
        value = small.render(f"{v:+.2f}", True, activation_color(v))
        if out_gap >= 24:  # name above value
            screen.blit(name, (nx + 10, ny - 11))
            screen.blit(value, (nx + 10, ny + 1))
        else:
            screen.blit(name, (nx + 10, ny - 5))
            screen.blit(value, (nx + 14 + name.get_width(), ny - 5))

    y = net_bottom + 6
    badge = ("EATING (stop_eat breaks off)" if c.eating_ticks > 0
             else "MATE-SEEKING (steering blended)" if c.mate_override else None)
    if badge:
        screen.blit(small.render(badge, True, config.COLOR_BEST_CREATURE), (x0 + 8, y))
    y += row_h
    for out_idx in (OUTPUTS.rotation, OUTPUTS.speed):
        drivers = "  ".join(f"{label} {v:+.2f}"
                            for label, v in top_drivers(c.brain, acts, out_idx))
        screen.blit(small.render(f"{out_labels[out_idx]}: {drivers}", True, config.COLOR_TEXT),
                    (x0 + 8, y))
        y += row_h

    n_out, gap = len(out_labels), 6
    spark_w = (w - 16 - gap * (n_out - 1)) // n_out
    for k, label in enumerate(out_labels):
        sx = x0 + 8 + k * (spark_w + gap)
        screen.blit(small.render(label, True, config.COLOR_TEXT), (sx, y + 2))
        rect = pygame.Rect(sx, y + 14, spark_w, spark_h)
        pygame.draw.rect(screen, (30, 36, 34), rect)
        pygame.draw.line(screen, (80, 80, 80), (rect.left, rect.centery),
                         (rect.right - 1, rect.centery))
        if len(history) > 1:
            step = (spark_w - 1) / (config.BRAIN_VIEW_HISTORY - 1)
            pts = [(rect.left + i * step, rect.centery - max(-1.0, min(1.0, o[k])) * spark_h / 2)
                   for i, o in enumerate(history)]
            pygame.draw.lines(screen, activation_color(history[-1][k]), False, pts)
