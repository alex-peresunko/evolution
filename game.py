import random
import math
import os
import subprocess
import sys
import time
import ctypes
import ctypes.util
from collections import deque
from multiprocessing import cpu_count

import pygame
from prometheus_client import start_http_server

from evolution import config
from evolution.entities import Creature, Food, Obstacle
from evolution.simulation import (update_world, create_start_populations, build_sim_state,
                                  food_cap, restore_progress, restore_start_progress,
                                  save_hall_of_fame, season_name, species_history)
from evolution.brain_history import (add_start_arguments, archive_after_start,
                                     check_load_argument, format_listing, start_source)
from evolution.genetics import mutate_brain, mutate_genes
from evolution.neural_network import NeuralNetwork
from evolution.metrics import GameMetrics
from evolution.brain_store import save_brain
from evolution.brain_view import (INPUT_LABELS, OUTPUT_LABELS, activations, edge_contributions,
                                  labels, network_layout, strong_edges, top_drivers)
from evolution.camera import follow_target, min_zoom, viewport_origin, zoom_at
from evolution.render import Renderer
from evolution.islands import (clear_island_files, import_if_better, publish_hall_of_fame,
                               worker_file)

# pygame-specific vector/rect factories
_vec2 = pygame.math.Vector2
def _rect(x, y, w, h): return pygame.Rect(x, y, w, h)

MINIMAP_RECT = None  # initialized after pygame.init()
_CHART_FONTS = {}
SPEEDS = (1, 2, 4, 8, 16, 32)  # simulation ticks per drawn frame
_PANELS = {}
_EDGE_LAYER = {}


# =============================================================================
# Island worker helpers
# =============================================================================
def _start_worker_processes(sim_state):
    n = max(1, (cpu_count() or 2) - 1)
    os.makedirs("saved_brains", exist_ok=True)
    clear_island_files()
    sim_state["island_seen"], sim_state["island_published"] = {}, {}
    publish_hall_of_fame(sim_state, sim_state["island_published"])  # workers start from our best
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    workers = []
    for i in range(n):
        log_path = os.path.join("saved_brains", f"worker_{i}.log")
        log_file = open(log_path, "w", encoding="utf-8")
        p = subprocess.Popen(
            [sys.executable, "game_headless.py", "--worker-id", str(i)],
            stdout=log_file, stderr=log_file, env=env,
        )
        workers.append({"id": i, "proc": p, "log": log_file})
    sim_state["worker_processes"] = workers
    print(f"[Islands] Started {n} worker process(es). Logs: saved_brains/worker_N.log")


def _stop_worker_processes(sim_state):
    """Stop the workers (they save their Hall of Fame on SIGTERM), then import what they saved."""
    workers = sim_state.get("worker_processes", [])
    if not workers:
        return
    for w in workers:
        w["proc"].terminate()
    for w in workers:
        try:
            w["proc"].wait(timeout=10)
        except subprocess.TimeoutExpired:
            w["proc"].kill()
            w["proc"].wait()
        w["log"].close()
    _sync_worker_brains(sim_state)
    sim_state["worker_processes"] = []
    print("[Islands] Worker processes stopped.")


def _sync_worker_brains(sim_state):
    """Import worker brains that beat the main Hall of Fame (immigrants join the population),
    then publish the main Hall of Fame for the workers."""
    for w in sim_state.get("worker_processes", []):
        for species in ("herbivore", "carnivore"):
            imported = import_if_better(sim_state, species, worker_file(w["id"], species),
                                        sim_state["island_seen"])
            if imported:
                old_score, new_score = imported
                sim_state["hof_unsaved"].add(species)  # persisted by the next autosave
                print(f"[Island sync] {species.capitalize()} HoF upgraded by worker {w['id']}: "
                      f"{old_score:.3f} -> {new_score:.3f} (+{new_score - old_score:.3f}); "
                      f"{config.ISLAND_IMMIGRANTS} immigrants joined")
    publish_hall_of_fame(sim_state, sim_state["island_published"])


# =============================================================================
# Rendering helpers
# =============================================================================
def _draw_panel(screen, rect, alpha=config.HUD_PANEL_ALPHA):
    """Translucent rounded HUD panel, cached per size and alpha."""
    key = (rect.width, rect.height, alpha)
    panel = _PANELS.get(key)
    if panel is None:
        panel = pygame.Surface(key[:2], pygame.SRCALPHA)
        pygame.draw.rect(panel, (12, 18, 16, alpha), panel.get_rect(), border_radius=8)
        pygame.draw.rect(panel, (255, 255, 255, 45), panel.get_rect(), width=1, border_radius=8)
        _PANELS[key] = panel
    screen.blit(panel, rect.topleft)


def _time_of_day(light):
    phase = "Day" if light > 0.66 else "Dusk/Dawn" if light > 0.33 else "Night"
    return f"Time: {phase} ({int(light * 100)}% light)"


def _draw_text_block(screen, font, lines, x, y, title=None, right=None):
    """right: the panel's right edge on screen; when given, x is ignored."""
    surfs = ([font.render(title, True, config.COLOR_BEST_CREATURE)] if title else [])
    surfs += [font.render(line, True, config.COLOR_TEXT) for line in lines]
    line_h = font.get_linesize()
    pad = 6
    text_w = max(s.get_width() for s in surfs)
    if right is not None:
        x = right - pad - text_w
    _draw_panel(screen, pygame.Rect(x - pad, y - pad, text_w + pad * 2,
                                    line_h * len(surfs) + pad * 2))
    for surf in surfs:
        screen.blit(surf, (x, y))
        y += line_h


def _chart_font(size):
    font = _CHART_FONTS.get(size)
    if font is None:
        font = _CHART_FONTS[size] = pygame.font.Font(None, size)
    return font


def draw_dual_chart(screen, rect, title, series, scale, generation):
    """Translucent panel with one line per (history, color, label) series, each scaled to its
    own peak (species scores differ by orders of magnitude), sized by the zoomable chart scale."""
    font = _chart_font(round(config.CHART_FONT_SIZE * scale))
    line_h = font.get_linesize()
    pad = round(4 * scale)
    _draw_panel(screen, rect, config.CHART_PANEL_ALPHA)
    screen.blit(font.render(title, True, config.COLOR_TEXT), (rect.x + pad, rect.y + pad))
    gens = font.render(f"Gen: {generation}", True, config.COLOR_TEXT)
    screen.blit(gens, (rect.right - gens.get_width() - pad, rect.y + pad))
    info_y = rect.y + pad + line_h
    for i, (history, color, label) in enumerate(series):  # peaks: first left, second right
        peak = font.render(f"{label}: {int(max(history, default=0))}", True, color)
        x = rect.x + pad if i == 0 else rect.right - peak.get_width() - pad
        screen.blit(peak, (x, info_y))
    top, bottom = info_y + line_h + pad, rect.bottom - pad
    left, width = rect.x + pad, rect.width - pad * 2
    for history, color, _ in series:
        if len(history) < 2:
            continue
        max_score = max(history) or 1
        points = [
            (left + i / (len(history) - 1) * width, bottom - (score / max_score) * (bottom - top))
            for i, score in enumerate(history)
        ]
        pygame.draw.lines(screen, color, False, points, max(1, round(scale)))


def draw_gene_chart(screen, rect, species, gene_history, scale):
    """Species mean of each gene as a ratio to its default, on one shared axis with a dashed
    1.0 baseline: shows which traits selection pushes up or down."""
    font = _chart_font(round(config.CHART_FONT_SIZE * scale))
    line_h = font.get_linesize()
    pad = round(4 * scale)
    _draw_panel(screen, rect, config.CHART_PANEL_ALPHA)
    screen.blit(font.render(f"Gene drift - {species}s (x default)", True, config.COLOR_TEXT),
                (rect.x + pad, rect.y + pad))
    defaults = config.DEFAULT_CARNIVORE_GENES if species == "carnivore" else config.DEFAULT_HERBIVORE_GENES
    series = [(gene, color, [v / defaults[gene] for v in gene_history.get(gene, ())])
              for gene, color in config.GENE_CHART_COLORS.items()]
    legend_font = _chart_font(round(config.CHART_FONT_SIZE * scale * 0.8))
    legend_h = legend_font.get_linesize()
    x, y = rect.x + pad, rect.y + pad + line_h
    for gene, color, ratios in series:  # legend, wrapped to the panel width
        label = legend_font.render(f"{gene} {ratios[-1]:.2f}" if ratios else gene, True, color)
        if x + label.get_width() > rect.right - pad:
            x, y = rect.x + pad, y + legend_h
        screen.blit(label, (x, y))
        x += label.get_width() + pad * 3
    top, bottom = y + legend_h + pad, rect.bottom - pad
    left, width = rect.x + pad, rect.width - pad * 2
    values = [r for _, _, ratios in series for r in ratios] + [1.0]
    lo, hi = min(values), max(values)
    span = (hi - lo) or 1.0
    lo, hi = lo - span * 0.05, hi + span * 0.05

    def to_y(r):
        return bottom - (r - lo) / (hi - lo) * (bottom - top)

    base_y = to_y(1.0)
    dash = round(6 * scale)
    for dx in range(0, width, dash * 2):
        pygame.draw.line(screen, (150, 150, 150), (left + dx, base_y),
                         (left + min(width, dx + dash), base_y))
    for _, color, ratios in series:
        if len(ratios) > 1:
            points = [(left + i / (len(ratios) - 1) * width, to_y(r)) for i, r in enumerate(ratios)]
            pygame.draw.lines(screen, color, False, points, max(1, round(scale)))


_WORLD_SIZE = (config.WORLD_WIDTH, config.WORLD_HEIGHT)
_SCREEN_SIZE = (config.SCREEN_WIDTH, config.SCREEN_HEIGHT)


def _view_size(sim_state):
    """The camera's view in world units."""
    zoom = sim_state["zoom"]
    return config.SCREEN_WIDTH / zoom, config.SCREEN_HEIGHT / zoom


def _screen_to_world(pos, sim_state):
    zoom = sim_state["zoom"]
    return sim_state["cam_x"] + pos[0] / zoom, sim_state["cam_y"] + pos[1] / zoom


def _set_viewport_center(wx, wy, sim_state):
    sim_state["cam_x"], sim_state["cam_y"] = viewport_origin(
        wx, wy, _WORLD_SIZE, _view_size(sim_state))


def _zoom(sim_state, notches, anchor):
    """Zoom by ZOOM_STEP per wheel notch around the screen point anchor."""
    zoom = sim_state["zoom"]
    new_zoom = min(config.ZOOM_MAX, max(min_zoom(_WORLD_SIZE, _SCREEN_SIZE),
                                        zoom * config.ZOOM_STEP ** notches))
    if abs(new_zoom - 1.0) < 1e-6:
        new_zoom = 1.0  # snap back to the renderer's direct, unscaled path
    sim_state["cam_x"], sim_state["cam_y"] = zoom_at(
        (sim_state["cam_x"], sim_state["cam_y"]), zoom, new_zoom, anchor, _WORLD_SIZE, _SCREEN_SIZE)
    sim_state["zoom"] = new_zoom


def _follow_selected(sim_state):
    """Keep the camera on the selected creature; when it dies, switch to its species' best."""
    c = follow_target(sim_state, sim_state["selected_creature"])
    sim_state["selected_creature"] = c
    if c is not None:
        _set_viewport_center(c.pos.x, c.pos.y, sim_state)


def _record_brain(sim_state):
    """Append the selected creature's outputs to the sparkline history (reset on a new target)."""
    c = sim_state["selected_creature"]
    if c is None or c.brain_outputs is None or not c.is_alive():
        return
    if sim_state["brain_history_owner"] is not c:
        sim_state["brain_history"].clear()
        sim_state["brain_history_owner"] = c
    sim_state["brain_history"].append(c.brain_outputs.copy())


def _activation_color(v):
    """Diverging color: blue at -1, gray at 0, orange at +1."""
    v = max(-1.0, min(1.0, float(v)))
    end = config.COLOR_BRAIN_POS if v > 0 else config.COLOR_BRAIN_NEG
    return tuple(round(z + (e - z) * abs(v)) for z, e in zip(config.COLOR_BRAIN_ZERO, end))


def draw_brain_panel(screen, sim_state):
    """Live network diagram of the selected creature: node activations, strongest
    connections (weight * activation), override badge, top input drivers, output sparklines."""
    c = sim_state["selected_creature"]
    w, h = config.BRAIN_PANEL_SIZE
    x0, y0 = 10, config.SCREEN_HEIGHT - h - 10
    _draw_panel(screen, pygame.Rect(x0, y0, w, h))
    font = _chart_font(15)
    kind = "Carnivore" if c.is_carnivore else "Herbivore"
    screen.blit(font.render(f"Brain - {kind} #{c.id}", True, config.COLOR_BEST_CREATURE),
                (x0 + 8, y0 + 6))

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
            pygame.draw.circle(screen, _activation_color(v), (nx, ny), radii[k])
            if radii[k] >= 3:
                pygame.draw.circle(screen, (200, 200, 200), (nx, ny), radii[k], 1)
    in_gap = (net_bottom - top) / len(acts[0])
    for idx in range(0, len(acts[0]), math.ceil(11 / in_gap)):  # skip labels that would overlap
        ny, v = nodes[0][idx][1], acts[0][idx]
        screen.blit(font.render(in_labels[idx], True, config.COLOR_TEXT), (x0 + 8, ny - 5))
        screen.blit(font.render(f"{v:+.2f}", True, _activation_color(v)), (x0 + 82, ny - 5))
    out_gap = (net_bottom - top) / len(acts[-1])
    for idx in range(0, len(acts[-1]), math.ceil(11 / out_gap)):
        (nx, ny), v = nodes[-1][idx], acts[-1][idx]
        name = font.render(out_labels[idx], True, config.COLOR_TEXT)
        value = font.render(f"{v:+.2f}", True, _activation_color(v))
        if out_gap >= 24:  # name above value
            screen.blit(name, (nx + 10, ny - 11))
            screen.blit(value, (nx + 10, ny + 1))
        else:
            screen.blit(name, (nx + 10, ny - 5))
            screen.blit(value, (nx + 14 + name.get_width(), ny - 5))

    y = net_bottom + 6
    badge = ("EATING (brain ignored)" if c.eating_ticks > 0
             else "MATE-SEEKING (steering blended)" if c.mate_override else None)
    if badge:
        screen.blit(font.render(badge, True, config.COLOR_BEST_CREATURE), (x0 + 8, y))
    y += row_h
    for out_idx in (0, 1):
        drivers = "  ".join(f"{label} {v:+.2f}" for label, v in top_drivers(c.brain, acts, out_idx))
        screen.blit(font.render(f"{out_labels[out_idx]}: {drivers}", True, config.COLOR_TEXT),
                    (x0 + 8, y))
        y += row_h

    history = sim_state["brain_history"]
    n_out, gap = len(out_labels), 6
    spark_w = (w - 16 - gap * (n_out - 1)) // n_out
    for k, label in enumerate(out_labels):
        sx = x0 + 8 + k * (spark_w + gap)
        screen.blit(font.render(label, True, config.COLOR_TEXT), (sx, y + 2))
        rect = pygame.Rect(sx, y + 14, spark_w, spark_h)
        pygame.draw.rect(screen, (30, 36, 34), rect)
        pygame.draw.line(screen, (80, 80, 80), (rect.left, rect.centery),
                         (rect.right - 1, rect.centery))
        if len(history) > 1:
            step = (spark_w - 1) / (config.BRAIN_VIEW_HISTORY - 1)
            pts = [(rect.left + i * step, rect.centery - max(-1.0, min(1.0, o[k])) * spark_h / 2)
                   for i, o in enumerate(history)]
            pygame.draw.lines(screen, _activation_color(history[-1][k]), False, pts)


def draw_minimap(screen, sim_state):
    mmap = sim_state["renderer"].minimap_bg.copy()
    sx = config.MINIMAP_WIDTH / config.WORLD_WIDTH
    sy = config.MINIMAP_HEIGHT / config.WORLD_HEIGHT
    for obs in sim_state["obstacles"]:
        mr = pygame.Rect(int(obs.rect.x * sx), int(obs.rect.y * sy),
                         max(2, int(obs.rect.width * sx)), max(2, int(obs.rect.height * sy)))
        pygame.draw.rect(mmap, config.COLOR_OBSTACLE, mr)
    for c in sim_state["creatures"] + sim_state["carnivores"]:
        if c.is_best:
            pygame.draw.circle(mmap, config.COLOR_BEST_CREATURE,
                               (int(c.pos.x * sx), int(c.pos.y * sy)), 3)
    vw, vh = _view_size(sim_state)
    vr = pygame.Rect(int(sim_state["cam_x"] * sx), int(sim_state["cam_y"] * sy),
                     max(1, int(vw * sx)), max(1, int(vh * sy)))
    pygame.draw.rect(mmap, config.COLOR_TEXT, vr, 1)
    screen.blit(mmap, (config.MINIMAP_X, config.MINIMAP_Y))
    pygame.draw.rect(screen, config.COLOR_TEXT,
                     (config.MINIMAP_X, config.MINIMAP_Y, config.MINIMAP_WIDTH, config.MINIMAP_HEIGHT), 1)


def draw_elements(screen, font, sim_state):
    sim_state["renderer"].draw_world(screen, sim_state)
    draw_minimap(screen, sim_state)
    text_y = config.MINIMAP_Y + config.MINIMAP_HEIGHT + 15
    limit_status = "Limited (60)" if sim_state["fps_limited"] else "Unlimited"
    stats = [
        f"Generation: {sim_state['generation']}",
        f"Tick: {sim_state['tick']}",
        f"Herbivores: {len(sim_state['creatures'])}",
        f"Carnivores: {len(sim_state['carnivores'])}",
        f"Lineages: H {sim_state['lineages']['herbivore']}  C {sim_state['lineages']['carnivore']}",
        f"Food: {len(sim_state['food_items'])}   Carcasses: {len(sim_state.get('carcasses', ()))}",
        _time_of_day(sim_state.get("light", 1.0)),
        f"Season: {season_name(sim_state['tick'])} (food cap {food_cap(sim_state['season']):.0f})",
        f"Top Herbivore Score: {max([c.score for c in sim_state['creatures']] + [0]):.2f}",
        f"Top Carnivore Score: {max([c.score for c in sim_state['carnivores']] + [0]):.2f}",
        f"Longest H: {sim_state['max_age_herbivore']} ticks",
        f"Longest C: {sim_state['max_age_carnivore']} ticks",
        f"FPS: {int(sim_state['clock'].get_fps())} [{limit_status}]",
        f"Speed: x{sim_state['ticks_per_frame']}  ({sim_state['tps']} ticks/s)",
        f"Zoom: x{sim_state['zoom']:.2f}",
    ]
    _draw_text_block(screen, font, stats, 10, text_y)
    selected = sim_state.get("selected_creature")
    if selected:
        info_lines = [f"{k}: {v}" for k, v in selected.get_info().items()]
        _draw_text_block(screen, font, info_lines, None, 16, title="Selected Creature",
                         right=config.SCREEN_WIDTH - 10)
        if sim_state["show_brain"] and selected.brain_inputs is not None:
            draw_brain_panel(screen, sim_state)
    scale = sim_state["chart_scale"]
    chart_w, chart_h = round(config.CHART_SIZE[0] * scale), round(config.CHART_SIZE[1] * scale)
    spacing = round(5 * scale)
    charts = [("HoF Score", "score"), ("Max Longevity", "longevity")]
    for i, (title, metric) in enumerate(charts):
        rect = pygame.Rect(config.SCREEN_WIDTH - chart_w - 10,
                           config.SCREEN_HEIGHT - 10 - chart_h * (i + 1) - spacing * i,
                           chart_w, chart_h)
        series = [(sim_state[f"herbivore_{metric}_history"], config.COLOR_CREATURE, "Herb"),
                  (sim_state[f"carnivore_{metric}_history"], config.COLOR_CARNIVORE, "Carn")]
        draw_dual_chart(screen, rect, title, series, scale, sim_state["generation"])
    if sim_state["show_genes"]:
        species = "carnivore" if selected and selected.is_carnivore else "herbivore"
        gene_w = round(chart_w * 1.6)
        rect = pygame.Rect(config.SCREEN_WIDTH - chart_w - gene_w - 20,
                           config.SCREEN_HEIGHT - 10 - chart_h * 2 - spacing, gene_w,
                           chart_h * 2 + spacing)
        draw_gene_chart(screen, rect, species, sim_state["gene_history"][species], scale)
    if sim_state["paused"]:
        screen.blit(font.render("PAUSED", True, (255, 100, 100)),
                    (config.SCREEN_WIDTH // 2 - 60, 20))
    pygame.display.flip()


# =============================================================================
# Input handling
# =============================================================================
def handle_events(sim_state, prometheus_metrics):
    global MINIMAP_RECT
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sim_state["running"] = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_s:
                best_h = max(sim_state["creatures"], key=lambda c: c.score, default=None)
                if best_h:
                    save_brain(best_h.brain, "best_brain_herbivore.json",
                               generation=sim_state["generation"], genes=best_h.genes,
                               score=best_h.score, metrics=prometheus_metrics,
                               history=species_history(sim_state, "herbivore"))
                best_c = max(sim_state["carnivores"], key=lambda c: c.score, default=None)
                if best_c:
                    save_brain(best_c.brain, "best_brain_carnivore.json",
                               generation=sim_state["generation"], genes=best_c.genes,
                               score=best_c.score, metrics=prometheus_metrics,
                               history=species_history(sim_state, "carnivore"))
            if event.key == pygame.K_l:
                try:
                    brain, gen, genes = NeuralNetwork.load("best_brain_herbivore.json",
                                                           config.BRAIN_TOPOLOGY)
                    sim_state["generation"] = gen
                    new_herbivores = [Creature(sim_state["world_bounds"], brain=brain, genes=genes,
                                               vec2_factory=_vec2, rect_factory=_rect)]
                    for _ in range(config.NUM_HERBIVOROUS - 1):
                        mb = brain.copy()
                        mg = mutate_genes(genes, config.DEFAULT_HERBIVORE_GENES)
                        mutate_brain(mb, mg)
                        new_herbivores.append(Creature(sim_state["world_bounds"], brain=mb, genes=mg,
                                                       vec2_factory=_vec2, rect_factory=_rect))
                    sim_state["creatures"] = new_herbivores
                    restore_progress(sim_state, "herbivore", brain, genes, "best_brain_herbivore.json")
                    print(f"Loaded new herbivore population from generation {gen}.")
                except FileNotFoundError:
                    print("No herbivore brain file found to load.")
                try:
                    brain_c, gen_c, genes_c = NeuralNetwork.load("best_brain_carnivore.json",
                                                              config.BRAIN_TOPOLOGY)
                    new_carnivores = [Creature(sim_state["world_bounds"], brain=brain_c,
                                               is_carnivore=True, genes=genes_c,
                                               vec2_factory=_vec2, rect_factory=_rect)]
                    for _ in range(config.NUM_CARNIVORES - 1):
                        mb = brain_c.copy()
                        mg = mutate_genes(genes_c, config.DEFAULT_CARNIVORE_GENES,
                                         gene_constraints=config.CARNIVORE_GENE_MIN_MAX)
                        mutate_brain(mb, mg)
                        new_carnivores.append(Creature(sim_state["world_bounds"], brain=mb,
                                                       is_carnivore=True, genes=mg,
                                                       vec2_factory=_vec2, rect_factory=_rect))
                    sim_state["carnivores"] = new_carnivores
                    sim_state["generation"] = max(sim_state["generation"], gen_c)
                    restore_progress(sim_state, "carnivore", brain_c, genes_c,
                                     "best_brain_carnivore.json")
                    print("Loaded new carnivore population.")
                except FileNotFoundError:
                    print("No carnivore brain file found to load.")
            if event.key == pygame.K_SPACE:
                sim_state["paused"] = not sim_state["paused"]
            if event.key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS,
                             pygame.K_MINUS, pygame.K_KP_MINUS):
                step = -1 if event.key in (pygame.K_MINUS, pygame.K_KP_MINUS) else 1
                i = SPEEDS.index(sim_state["ticks_per_frame"]) + step
                sim_state["ticks_per_frame"] = SPEEDS[max(0, min(len(SPEEDS) - 1, i))]
            if event.key in (pygame.K_LEFTBRACKET, pygame.K_RIGHTBRACKET):
                step = config.CHART_SCALE_STEP * (1 if event.key == pygame.K_RIGHTBRACKET else -1)
                sim_state["chart_scale"] = max(config.CHART_SCALE_MIN, min(
                    config.CHART_SCALE_MAX, sim_state["chart_scale"] + step))
            if event.key == pygame.K_RETURN:
                sim_state["fps_limited"] = not sim_state["fps_limited"]
                print(f"FPS Limiting {'Enabled' if sim_state['fps_limited'] else 'Disabled'}")
            if event.key == pygame.K_g:
                sim_state["show_genes"] = not sim_state["show_genes"]
            if event.key == pygame.K_n:
                sim_state["show_brain"] = not sim_state["show_brain"]
            if event.key == pygame.K_d:
                sim_state["drawing_enabled"] = not sim_state["drawing_enabled"]
                print(f"Drawing {'Enabled' if sim_state['drawing_enabled'] else 'Disabled'}")
            if event.key == pygame.K_b:
                sim_state["background_mode"] = not sim_state["background_mode"]
                if sim_state["background_mode"]:
                    _start_worker_processes(sim_state)
                    print("--- ISLAND WORKERS ENGAGED ---")
                else:
                    _stop_worker_processes(sim_state)
                    print("--- Island Workers Stopped ---")

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if MINIMAP_RECT and MINIMAP_RECT.collidepoint(event.pos):
                sim_state["minimap_dragging"] = True
                sim_state["selected_creature"] = None  # a manual pan stops following
                wx = (event.pos[0] - config.MINIMAP_X) / config.MINIMAP_WIDTH * config.WORLD_WIDTH
                wy = (event.pos[1] - config.MINIMAP_Y) / config.MINIMAP_HEIGHT * config.WORLD_HEIGHT
                _set_viewport_center(wx, wy, sim_state)
            else:
                world_mx, world_my = _screen_to_world(event.pos, sim_state)
                found = False
                for c in sim_state["creatures"] + sim_state["carnivores"]:
                    if c.pos.distance_to((world_mx, world_my)) < c.size:
                        sim_state["selected_creature"] = (
                            c if sim_state["selected_creature"] != c else None
                        )
                        found = True
                        break
                if not found:
                    sim_state["selected_creature"] = None
                    food = Food(sim_state["world_bounds"], vec2_factory=_vec2)
                    food.pos = _vec2(world_mx, world_my)
                    sim_state["food_items"].append(food)
        elif event.type == pygame.MOUSEMOTION:
            if sim_state.get("minimap_dragging"):
                wx = (event.pos[0] - config.MINIMAP_X) / config.MINIMAP_WIDTH * config.WORLD_WIDTH
                wy = (event.pos[1] - config.MINIMAP_Y) / config.MINIMAP_HEIGHT * config.WORLD_HEIGHT
                _set_viewport_center(wx, wy, sim_state)
        elif event.type == pygame.MOUSEWHEEL and event.y:
            _zoom(sim_state, event.y, pygame.mouse.get_pos())
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            sim_state["minimap_dragging"] = False


# =============================================================================
# Main
# =============================================================================
def _set_performance_qos():
    try:
        libc = ctypes.CDLL(ctypes.util.find_library('c'))
        QOS_CLASS_USER_INTERACTIVE = 0x21
        libc.pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0)
        print("QoS set to USER_INTERACTIVE (P-cores preferred).")
    except Exception:
        pass


def main(new=False, load=None):
    _set_performance_qos()
    global MINIMAP_RECT
    pygame.init()
    screen = pygame.display.set_mode((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))
    pygame.display.set_caption("Evolution Sim | S to Save, L to Load, SPACE to Pause, +/- Speed")
    font = pygame.font.Font(None, config.HUD_FONT_SIZE)
    MINIMAP_RECT = pygame.Rect(
        config.MINIMAP_X, config.MINIMAP_Y, config.MINIMAP_WIDTH, config.MINIMAP_HEIGHT
    )

    world_bounds = (config.WORLD_WIDTH, config.WORLD_HEIGHT)
    source = start_source(new, load)
    herbivores, carnivores, generation = create_start_populations(
        world_bounds, source, vec2_factory=_vec2, rect_factory=_rect)
    food_items = [Food(world_bounds, vec2_factory=_vec2) for _ in range(config.NUM_FOOD)]
    obstacles = [
        Obstacle(
            random.randint(100, config.WORLD_WIDTH - 100),
            random.randint(100, config.WORLD_HEIGHT - 100),
            random.randint(50, 150),
            random.randint(50, 150),
            rect_factory=_rect,
        )
        for _ in range(config.NUM_OBSTACLES)
    ]

    sim_state = build_sim_state(world_bounds, herbivores, carnivores, food_items, obstacles)
    sim_state.update({
        "generation": generation,
        "paused": False,
        "clock": pygame.time.Clock(),
        "selected_creature": None,
        "fps_limited": True,
        "drawing_enabled": True,
        "background_mode": False,
        "worker_processes": [],
        "cam_x": 0.0,
        "cam_y": 0.0,
        "zoom": 1.0,
        "minimap_dragging": False,
        "ticks_per_frame": 1,
        "chart_scale": config.CHART_SCALE_DEFAULT,
        "tps": 0,
        "show_brain": True,
        "show_genes": True,
        "brain_history": deque(maxlen=config.BRAIN_VIEW_HISTORY),
        "brain_history_owner": None,
        "renderer": Renderer(config.WORLD_WIDTH, config.WORLD_HEIGHT, obstacles),
    })

    restore_start_progress(sim_state, source)
    snapshot = archive_after_start(source, new)
    if snapshot:
        print(f"Previous best brains kept in {snapshot}")
    prometheus_metrics = GameMetrics(sim_state)

    tps_window_start, tps_window_tick = time.perf_counter(), sim_state["tick"]
    try:
        while sim_state["running"]:
            handle_events(sim_state, prometheus_metrics)
            if not sim_state["paused"]:
                for _ in range(sim_state["ticks_per_frame"]):
                    update_world(sim_state, prometheus_metrics)
                    _record_brain(sim_state)
                    if (sim_state.get("background_mode") and sim_state["worker_processes"]
                            and sim_state["tick"] % config.WORKER_SYNC_INTERVAL == 0):
                        _sync_worker_brains(sim_state)
            _follow_selected(sim_state)
            now = time.perf_counter()
            if now - tps_window_start >= 1.0:
                sim_state["tps"] = int((sim_state["tick"] - tps_window_tick) / (now - tps_window_start))
                tps_window_start, tps_window_tick = now, sim_state["tick"]
            if sim_state["drawing_enabled"]:
                draw_elements(screen, font, sim_state)
            if sim_state["fps_limited"]:
                sim_state["clock"].tick(60)
            else:
                sim_state["clock"].tick()
            dt = sim_state["clock"].get_time() / 1000.0
            prometheus_metrics.update(sim_state, dt)
    finally:  # also on Ctrl+C or a crash, so workers are never left running
        _stop_worker_processes(sim_state)
        print("Saving Hall of Fame on exit...")
        save_hall_of_fame(sim_state, prometheus_metrics)
        pygame.quit()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    add_start_arguments(parser)
    args = parser.parse_args()
    if args.list_brains:
        print(format_listing())
        sys.exit(0)
    check_load_argument(parser, args)
    start_http_server(8000)
    main(new=args.new, load=args.load)
