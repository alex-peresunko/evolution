import random
import math
import os
import subprocess
import sys
import time
import ctypes
import ctypes.util
from multiprocessing import cpu_count

import pygame
from prometheus_client import start_http_server

from evolution import config
from evolution.entities import Creature, Food, Obstacle
from evolution.simulation import (update_world, create_initial_population, build_sim_state,
                                  restore_progress, save_hall_of_fame, species_history)
from evolution.genetics import mutate_genes
from evolution.neural_network import NeuralNetwork
from evolution.metrics import GameMetrics
from evolution.brain_store import save_brain
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
def _draw_panel(screen, rect):
    """Translucent rounded HUD panel, cached per size."""
    key = (rect.width, rect.height)
    panel = _PANELS.get(key)
    if panel is None:
        panel = pygame.Surface(key, pygame.SRCALPHA)
        pygame.draw.rect(panel, (12, 18, 16, 165), panel.get_rect(), border_radius=8)
        pygame.draw.rect(panel, (255, 255, 255, 45), panel.get_rect(), width=1, border_radius=8)
        _PANELS[key] = panel
    screen.blit(panel, rect.topleft)


def _time_of_day(light):
    phase = "Day" if light > 0.66 else "Dusk/Dawn" if light > 0.33 else "Night"
    return f"Time: {phase} ({int(light * 100)}% light)"


def _draw_text_block(screen, font, lines, x, y, title=None):
    surfs = ([font.render(title, True, config.COLOR_BEST_CREATURE)] if title else [])
    surfs += [font.render(line, True, config.COLOR_TEXT) for line in lines]
    line_h = font.get_linesize()
    pad = 6
    _draw_panel(screen, pygame.Rect(x - pad, y - pad, max(s.get_width() for s in surfs) + pad * 2,
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
    _draw_panel(screen, rect)
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


def _set_viewport_center(wx, wy, sim_state):
    sim_state["cam_x"] = max(0.0, min(float(config.WORLD_WIDTH - config.SCREEN_WIDTH),
                                      wx - config.SCREEN_WIDTH / 2))
    sim_state["cam_y"] = max(0.0, min(float(config.WORLD_HEIGHT - config.SCREEN_HEIGHT),
                                      wy - config.SCREEN_HEIGHT / 2))


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
    vr = pygame.Rect(int(sim_state["cam_x"] * sx), int(sim_state["cam_y"] * sy),
                     max(1, int(config.SCREEN_WIDTH * sx)), max(1, int(config.SCREEN_HEIGHT * sy)))
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
        f"Food: {len(sim_state['food_items'])}   Carcasses: {len(sim_state.get('carcasses', ()))}",
        _time_of_day(sim_state.get("light", 1.0)),
        f"Top Herbivore Score: {max([c.score for c in sim_state['creatures']] + [0]):.2f}",
        f"Top Carnivore Score: {max([c.score for c in sim_state['carnivores']] + [0]):.2f}",
        f"Longest H: {sim_state['max_age_herbivore']} ticks",
        f"Longest C: {sim_state['max_age_carnivore']} ticks",
        f"FPS: {int(sim_state['clock'].get_fps())} [{limit_status}]",
        f"Speed: x{sim_state['ticks_per_frame']}  ({sim_state['tps']} ticks/s)",
    ]
    _draw_text_block(screen, font, stats, 10, text_y)
    selected = sim_state.get("selected_creature")
    if selected and selected.is_alive():
        info_lines = [f"{k}: {v}" for k, v in selected.get_info().items()]
        _draw_text_block(screen, font, info_lines, config.SCREEN_WIDTH - 450, 16,
                         title="Selected Creature")
    else:
        sim_state["selected_creature"] = None
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
                        mb.mutate(config.MUTATION_RATE, config.MUTATION_AMOUNT)
                        mg = mutate_genes(genes, config.DEFAULT_HERBIVORE_GENES)
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
                        mb.mutate(config.MUTATION_RATE, config.MUTATION_AMOUNT)
                        mg = mutate_genes(genes_c, config.DEFAULT_CARNIVORE_GENES,
                                         gene_constraints=config.CARNIVORE_GENE_MIN_MAX)
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
                wx = (event.pos[0] - config.MINIMAP_X) / config.MINIMAP_WIDTH * config.WORLD_WIDTH
                wy = (event.pos[1] - config.MINIMAP_Y) / config.MINIMAP_HEIGHT * config.WORLD_HEIGHT
                _set_viewport_center(wx, wy, sim_state)
            else:
                world_mx = event.pos[0] + sim_state["cam_x"]
                world_my = event.pos[1] + sim_state["cam_y"]
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


def main():
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
    herbivores, h_gen = create_initial_population(
        config.NUM_HERBIVOROUS, False, world_bounds,
        vec2_factory=_vec2, rect_factory=_rect,
    )
    carnivores, c_gen = create_initial_population(
        config.NUM_CARNIVORES, True, world_bounds,
        vec2_factory=_vec2, rect_factory=_rect,
    )
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
        "generation": max(h_gen, c_gen),
        "paused": False,
        "clock": pygame.time.Clock(),
        "selected_creature": None,
        "fps_limited": True,
        "drawing_enabled": True,
        "background_mode": False,
        "worker_processes": [],
        "cam_x": 0.0,
        "cam_y": 0.0,
        "minimap_dragging": False,
        "ticks_per_frame": 1,
        "chart_scale": config.CHART_SCALE_DEFAULT,
        "tps": 0,
        "renderer": Renderer(config.WORLD_WIDTH, config.WORLD_HEIGHT, obstacles),
    })

    prometheus_metrics = GameMetrics(sim_state)

    tps_window_start, tps_window_tick = time.perf_counter(), sim_state["tick"]
    try:
        while sim_state["running"]:
            handle_events(sim_state, prometheus_metrics)
            if not sim_state["paused"]:
                for _ in range(sim_state["ticks_per_frame"]):
                    update_world(sim_state, prometheus_metrics)
                    if (sim_state.get("background_mode") and sim_state["worker_processes"]
                            and sim_state["tick"] % config.WORKER_SYNC_INTERVAL == 0):
                        _sync_worker_brains(sim_state)
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
    start_http_server(8000)
    main()
