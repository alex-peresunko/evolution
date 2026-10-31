"""game.py's main loop: the Simulation, the player's view of it, and island workers."""
import ctypes
import ctypes.util
import time

import pygame

from .. import config
from ..app.bootstrap import archive_previous_best, start_simulation
from ..app.console import ExtinctionReporter
from ..app.metrics import GameMetrics
from ..persistence.autosave import HallOfFameAutosaver, save_on_exit
from ..world.geometry import Geometry
from .commands import handle_events
from .hud import draw_hud
from .render import WorldRenderer
from .view_state import ViewState
from .workers import WorkerPool

# pygame's C vectors and rects: much faster than the headless ones, and the renderer needs them.
PYGAME_GEOMETRY = Geometry(pygame.math.Vector2, pygame.Rect)


def draw_frame(screen, font, sim, view, renderer, fps):
    renderer.draw(screen, sim.world, view)
    draw_hud(screen, font, sim, view, renderer.minimap_bg, fps)
    pygame.display.flip()


def _prefer_performance_cores():
    """macOS: ask for the user-interactive QoS class (P-cores); a no-op elsewhere."""
    try:
        libc = ctypes.CDLL(ctypes.util.find_library("c"))
        QOS_CLASS_USER_INTERACTIVE = 0x21
        libc.pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0)
        print("QoS set to USER_INTERACTIVE (P-cores preferred).")
    except Exception:
        pass


class GameApp:
    def __init__(self, source, new=False):
        _prefer_performance_cores()
        pygame.init()
        self.screen = pygame.display.set_mode((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))
        pygame.display.set_caption(
            "Evolution Sim | S to Save, L to Load, SPACE to Pause, +/- Speed")
        self.font = pygame.font.Font(None, config.HUD_FONT_SIZE)
        self.clock = pygame.time.Clock()
        self.metrics = GameMetrics()
        self.autosaver = HallOfFameAutosaver(uptime=self.metrics.uptime_seconds)
        self.sim = start_simulation(source, geometry=PYGAME_GEOMETRY,
                                    listeners=[self.autosaver, ExtinctionReporter()])
        world = self.sim.world
        self.renderer = WorldRenderer(config.WORLD_WIDTH, config.WORLD_HEIGHT, world.obstacles,
                                      world.lakes)
        archive_previous_best(source, new)
        self.view = ViewState()
        self.workers = WorkerPool(self.sim, self.autosaver)

    def run(self):
        sim, view = self.sim, self.view
        tps_window_start, tps_window_tick = time.perf_counter(), sim.tick
        try:
            while view.running:
                handle_events(self)
                if not view.paused:
                    for _ in range(view.ticks_per_frame):
                        sim.step()
                        view.record_brain()
                        self.workers.sync_due()
                view.follow_selected(sim.world)
                now = time.perf_counter()
                if now - tps_window_start >= 1.0:
                    view.tps = int((sim.tick - tps_window_tick) / (now - tps_window_start))
                    tps_window_start, tps_window_tick = now, sim.tick
                if view.drawing_enabled:
                    draw_frame(self.screen, self.font, sim, view, self.renderer,
                               self.clock.get_fps())
                self.clock.tick(60 if view.fps_limited else 0)
        finally:  # also on Ctrl+C or a crash, so workers are never left running
            self.workers.stop()
            print("Saving Hall of Fame on exit...")
            save_on_exit(sim, uptime=self.metrics.uptime_seconds)
            pygame.quit()
