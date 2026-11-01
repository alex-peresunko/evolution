from collections import deque
from dataclasses import dataclass, field

from .. import config
from .camera import follow_target, min_zoom, viewport_origin, zoom_at

WORLD_SIZE = (config.WORLD_WIDTH, config.WORLD_HEIGHT)
SCREEN_SIZE = (config.SCREEN_WIDTH, config.SCREEN_HEIGHT)
SPEEDS = (1 / 8, 1 / 4, 1 / 2, 1, 2, 4, 8, 16, 32)  # simulation ticks per drawn frame


@dataclass
class ViewState:
    """Everything the player controls that is not part of the simulation."""

    running: bool = True
    paused: bool = False
    fps_limited: bool = True
    drawing_enabled: bool = True
    show_brain: bool = True
    show_genes: bool = True
    ticks_per_frame: float = 1
    tick_carry: float = 0.0                  # unspent fraction of a tick (speeds below 1)
    poses: dict | None = None                # {creature id: (x, y, angle)} drawn instead
    tps: int = 0
    chart_scale: float = config.CHART_SCALE_DEFAULT
    cam_x: float = 0.0
    cam_y: float = 0.0
    zoom: float = 1.0
    minimap_dragging: bool = False
    selected: object = None                  # the creature the camera follows
    tentative_pick: object = None            # selected, if picked while its species was tied
    brain_history: deque = field(default_factory=lambda: deque(maxlen=config.BRAIN_VIEW_HISTORY))
    brain_history_owner: object = None

    # --- camera ------------------------------------------------------------------------------

    def view_size(self):
        """The camera's view in world units."""
        return config.SCREEN_WIDTH / self.zoom, config.SCREEN_HEIGHT / self.zoom

    def screen_to_world(self, pos):
        return self.cam_x + pos[0] / self.zoom, self.cam_y + pos[1] / self.zoom

    def center_on(self, wx, wy):
        self.cam_x, self.cam_y = viewport_origin(wx, wy, WORLD_SIZE, self.view_size())

    def zoom_by(self, notches, anchor):
        """Zoom by ZOOM_STEP per wheel notch around the screen point anchor."""
        new_zoom = min(config.ZOOM_MAX, max(min_zoom(WORLD_SIZE, SCREEN_SIZE),
                                            self.zoom * config.ZOOM_STEP ** notches))
        if abs(new_zoom - 1.0) < 1e-6:
            new_zoom = 1.0  # snap back to the renderer's direct, unscaled path
        self.cam_x, self.cam_y = zoom_at((self.cam_x, self.cam_y), self.zoom, new_zoom, anchor,
                                         WORLD_SIZE, SCREEN_SIZE)
        self.zoom = new_zoom

    # --- speed and charts --------------------------------------------------------------------

    def change_speed(self, step):
        i = SPEEDS.index(self.ticks_per_frame) + step
        self.ticks_per_frame = SPEEDS[max(0, min(len(SPEEDS) - 1, i))]
        self.tick_carry = 0.0

    def ticks_due(self):
        """Ticks to simulate this frame: ticks_per_frame on average, whole ticks only."""
        self.tick_carry += self.ticks_per_frame
        due = int(self.tick_carry)
        self.tick_carry -= due
        return due

    def speed_label(self):
        return (f"x{self.ticks_per_frame:g}" if self.ticks_per_frame >= 1
                else f"x1/{round(1 / self.ticks_per_frame)}")

    def scale_charts(self, direction):
        self.chart_scale = max(config.CHART_SCALE_MIN, min(
            config.CHART_SCALE_MAX, self.chart_scale + config.CHART_SCALE_STEP * direction))

    # --- selection ---------------------------------------------------------------------------

    def follow_selected(self, world):
        """Keep the camera on the selected creature; when it dies, switch to its species'
        best."""
        tentative = self.selected is not None and self.selected is self.tentative_pick
        c, tentative = follow_target(world, self.selected, tentative)
        self.selected, self.tentative_pick = c, c if tentative else None
        if c is not None:
            pose = self.poses.get(c.id) if self.poses else None
            self.center_on(*(pose[:2] if pose else (c.pos.x, c.pos.y)))

    def record_brain(self):
        """Append the selected creature's outputs to the sparkline history (reset on a new
        target)."""
        c = self.selected
        if c is None or c.brain_outputs is None or not c.is_alive():
            return
        if self.brain_history_owner is not c:
            self.brain_history.clear()
            self.brain_history_owner = c
        self.brain_history.append(c.brain_outputs.copy())
