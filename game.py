import random
import math
import json
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
from evolution.simulation import update_world, create_initial_population, build_sim_state
from evolution.genetics import mutate_genes
from evolution.neural_network import NeuralNetwork
from evolution.metrics import GameMetrics
from evolution.brain_store import save_brain

# pygame-specific vector/rect factories
_vec2 = pygame.math.Vector2
def _rect(x, y, w, h): return pygame.Rect(x, y, w, h)

MINIMAP_RECT = None  # initialized after pygame.init()


# =============================================================================
# Island worker helpers
# =============================================================================
def _start_worker_processes(sim_state):
    n = max(1, (cpu_count() or 2) - 1)
    os.makedirs("saved_brains", exist_ok=True)
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
        workers.append({"id": i, "proc": p, "log": log_file,
                        "herbivore_mtime": 0, "carnivore_mtime": 0})
    sim_state["worker_processes"] = workers
    print(f"[Islands] Started {n} worker process(es). Logs: saved_brains/worker_N.log")


def _stop_worker_processes(sim_state):
    for w in sim_state.get("worker_processes", []):
        w["proc"].terminate()
        if w.get("log"):
            w["log"].close()
    sim_state["worker_processes"] = []
    print("[Islands] Worker processes stopped.")


def _sync_worker_brains(sim_state):
    hof = sim_state["hall_of_fame"]
    for w in sim_state.get("worker_processes", []):
        for species, suffix in [("herbivore", "herbivore.json"), ("carnivore", "carnivore.json")]:
            filepath = os.path.join("saved_brains", f"worker_{w['id']}_{suffix}")
            mtime_key = f"{species}_mtime"
            try:
                mtime = os.path.getmtime(filepath)
                if mtime <= w[mtime_key]:
                    continue
                w[mtime_key] = mtime
                with open(filepath) as f:
                    data = json.load(f)
                worker_score = data.get("score", 0)
                if worker_score > hof[species]["score"]:
                    old_score = hof[species]["score"]
                    brain, _, genes = NeuralNetwork.load(
                        f"worker_{w['id']}_{suffix}", config.BRAIN_TOPOLOGY
                    )
                    hof[species] = {"score": worker_score, "brain": brain, "genes": genes}
                    delta = worker_score - old_score
                    print(
                        f"[Island sync] {species.capitalize()} HoF upgraded by worker {w['id']}: "
                        f"{old_score:.3f} -> {worker_score:.3f} (+{delta:.3f})"
                    )
            except (FileNotFoundError, json.JSONDecodeError, Exception):
                pass


# =============================================================================
# Rendering helpers
# =============================================================================
def draw_score_chart(screen, font, history, position, size, color, title):
    small_font = pygame.font.Font(None, config.CHART_FONT_SIZE)
    line_h = small_font.get_linesize()
    surf = pygame.Surface(size)
    surf.set_alpha(config.CHART_OPACITY)
    surf.fill((40, 40, 50))
    pygame.draw.rect(surf, config.COLOR_TEXT, surf.get_rect(), 1)
    surf.blit(small_font.render(title, True, color), (3, 3))
    if len(history) >= 2:
        max_score = max(history) or 1
        points = [
            (i / (len(history) - 1) * size[0],
             size[1] - (score / max_score) * (size[1] - line_h * 2 - 4))
            for i, score in enumerate(history)
        ]
        pygame.draw.lines(surf, color, False, points, 1)
        surf.blit(small_font.render(f"Peak: {int(max_score)}", True, config.COLOR_TEXT), (3, line_h + 4))
        gen_label_surf = small_font.render(f"Gens: {len(history)}", True, config.COLOR_TEXT)
        surf.blit(gen_label_surf, (size[0] - gen_label_surf.get_width() - 3, 3))
    screen.blit(surf, position)


def _set_viewport_center(wx, wy, sim_state):
    sim_state["cam_x"] = max(0.0, min(float(config.WORLD_WIDTH - config.SCREEN_WIDTH),
                                      wx - config.SCREEN_WIDTH / 2))
    sim_state["cam_y"] = max(0.0, min(float(config.WORLD_HEIGHT - config.SCREEN_HEIGHT),
                                      wy - config.SCREEN_HEIGHT / 2))


def draw_minimap(screen, sim_state):
    mmap = pygame.Surface((config.MINIMAP_WIDTH, config.MINIMAP_HEIGHT), pygame.SRCALPHA)
    mmap.fill((20, 20, 30, 190))
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
    screen.fill(config.COLOR_BACKGROUND)
    cam = (sim_state["cam_x"], sim_state["cam_y"])
    vx, vy = cam
    for item in sim_state["obstacles"]:
        if item.rect.colliderect((vx, vy, config.SCREEN_WIDTH, config.SCREEN_HEIGHT)):
            item.draw(screen, cam_offset=cam)
    for item in sim_state["food_items"]:
        if (vx - 10 <= item.pos.x <= vx + config.SCREEN_WIDTH + 10
                and vy - 10 <= item.pos.y <= vy + config.SCREEN_HEIGHT + 10):
            item.draw(screen, cam_offset=cam)
    for c in sim_state["creatures"] + sim_state["carnivores"]:
        r = c.size + 5
        if (vx - r <= c.pos.x <= vx + config.SCREEN_WIDTH + r
                and vy - r <= c.pos.y <= vy + config.SCREEN_HEIGHT + r):
            c.draw(screen, c == sim_state.get("selected_creature"), cam_offset=cam)
    draw_minimap(screen, sim_state)
    text_y = config.MINIMAP_Y + config.MINIMAP_HEIGHT + 15
    limit_status = "Limited (60)" if sim_state["fps_limited"] else "Unlimited"
    stats = [
        f"Generation: {sim_state['generation']}",
        f"Tick: {sim_state['tick']}",
        f"Herbivores: {len(sim_state['creatures'])}",
        f"Carnivores: {len(sim_state['carnivores'])}",
        f"Food: {len(sim_state['food_items'])}",
        f"Top Herbivore Score: {max([c.score for c in sim_state['creatures']] + [0])}",
        f"Top Carnivore Score: {max([c.score for c in sim_state['carnivores']] + [0])}",
        f"Longest H: {sim_state['max_age_herbivore']} ticks",
        f"Longest C: {sim_state['max_age_carnivore']} ticks",
        f"FPS: {int(sim_state['clock'].get_fps())} [{limit_status}]",
    ]
    for line in stats:
        screen.blit(font.render(line, True, config.COLOR_TEXT), (10, text_y))
        text_y += 30
    selected = sim_state.get("selected_creature")
    if selected and selected.is_alive():
        info_lines = [f"{k}: {v}" for k, v in selected.get_info().items()]
        x, y = config.SCREEN_WIDTH - 450, 10
        screen.blit(font.render("Selected Creature", True, config.COLOR_TEXT), (x, y))
        y += 30
        for line in info_lines:
            screen.blit(font.render(line, True, config.COLOR_TEXT), (x, y))
            y += 25
    else:
        sim_state["selected_creature"] = None
    chart_width, chart_height, chart_spacing = 150, 60, 5
    chart_x_start = config.SCREEN_WIDTH - chart_width - 10
    draw_score_chart(screen, font, sim_state["carnivore_score_history"],
                     (chart_x_start, config.SCREEN_HEIGHT - chart_height - 10),
                     (chart_width, chart_height), config.COLOR_CARNIVORE, "Carnivore HoF Score")
    draw_score_chart(screen, font, sim_state["herbivore_score_history"],
                     (chart_x_start, config.SCREEN_HEIGHT - (chart_height * 2) - chart_spacing - 10),
                     (chart_width, chart_height), config.COLOR_CREATURE, "Herbivore HoF Score")
    draw_score_chart(screen, font, sim_state["carnivore_longevity_history"],
                     (chart_x_start, config.SCREEN_HEIGHT - chart_height * 3 - chart_spacing * 2 - 10),
                     (chart_width, chart_height), config.COLOR_CARNIVORE, "Carnivore Max Longevity")
    draw_score_chart(screen, font, sim_state["herbivore_longevity_history"],
                     (chart_x_start, config.SCREEN_HEIGHT - chart_height * 4 - chart_spacing * 3 - 10),
                     (chart_width, chart_height), config.COLOR_CREATURE, "Herbivore Max Longevity")
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
            _stop_worker_processes(sim_state)
            sim_state["running"] = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_s:
                best_h = max(sim_state["creatures"], key=lambda c: c.score, default=None)
                if best_h:
                    save_brain(best_h.brain, "best_brain_herbivore.json",
                               generation=sim_state["generation"], genes=best_h.genes,
                               score=best_h.score, metrics=prometheus_metrics)
                best_c = max(sim_state["carnivores"], key=lambda c: c.score, default=None)
                if best_c:
                    save_brain(best_c.brain, "best_brain_carnivore.json",
                               generation=sim_state["generation"], genes=best_c.genes,
                               score=best_c.score, metrics=prometheus_metrics)
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
                    print(f"Loaded new herbivore population from generation {gen}.")
                except FileNotFoundError:
                    print("No herbivore brain file found to load.")
                try:
                    brain_c, _, genes_c = NeuralNetwork.load("best_brain_carnivore.json",
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
                    print("Loaded new carnivore population.")
                except FileNotFoundError:
                    print("No carnivore brain file found to load.")
            if event.key == pygame.K_SPACE:
                sim_state["paused"] = not sim_state["paused"]
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
    pygame.display.set_caption("Evolution Sim | S to Save, L to Load, SPACE to Pause")
    font = pygame.font.Font(None, 30)
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
    })

    prometheus_metrics = GameMetrics(sim_state)

    while sim_state["running"]:
        handle_events(sim_state, prometheus_metrics)
        if not sim_state["paused"]:
            update_world(sim_state, prometheus_metrics)
            if sim_state.get("background_mode") and sim_state["tick"] % config.WORKER_SYNC_INTERVAL == 0:
                if sim_state["worker_processes"]:
                    _sync_worker_brains(sim_state)
        if sim_state["drawing_enabled"]:
            draw_elements(screen, font, sim_state)
        if sim_state["fps_limited"]:
            sim_state["clock"].tick(60)
        else:
            sim_state["clock"].tick()
        dt = sim_state["clock"].get_time() / 1000.0
        prometheus_metrics.update(sim_state, dt)
    pygame.quit()


if __name__ == "__main__":
    start_http_server(8000)
    main()
