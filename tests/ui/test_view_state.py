import pytest

from evolution import config
from evolution.app import bootstrap
from evolution.brain.network import NeuralNetwork
from evolution.species import HERBIVORE
from evolution.ui.view_state import SPEEDS, ViewState


def test_speed_steps_through_the_presets_and_clamps():
    view = ViewState()
    for _ in range(len(SPEEDS) + 2):
        view.change_speed(-1)
    assert view.ticks_per_frame == SPEEDS[0] < 1
    for _ in range(len(SPEEDS) + 2):
        view.change_speed(1)
    assert view.ticks_per_frame == SPEEDS[-1]


@pytest.mark.parametrize("speed, label", [(1 / 8, "x1/8"), (1 / 2, "x1/2"), (1, "x1"), (32, "x32")])
def test_ticks_due_averages_the_speed_in_whole_ticks(speed, label):
    view = ViewState(ticks_per_frame=speed)
    frames = [view.ticks_due() for _ in range(16)]
    assert set(frames) <= {0, int(speed) or 1}
    assert sum(frames) == 16 * speed
    assert view.speed_label() == label


def test_chart_scale_is_clamped():
    view = ViewState()
    for _ in range(100):
        view.scale_charts(1)
    assert view.chart_scale == config.CHART_SCALE_MAX
    for _ in range(100):
        view.scale_charts(-1)
    assert view.chart_scale == config.CHART_SCALE_MIN


def test_zoom_keeps_the_anchor_point_fixed():
    view = ViewState(cam_x=200.0, cam_y=100.0)
    anchor = (300, 200)
    before = view.screen_to_world(anchor)
    view.zoom_by(2, anchor)
    assert view.zoom == pytest.approx(config.ZOOM_STEP ** 2)
    assert view.screen_to_world(anchor) == pytest.approx(before)


def test_zooming_back_snaps_to_one():
    view = ViewState()
    view.zoom_by(3, (0, 0))
    view.zoom_by(-3, (0, 0))
    assert view.zoom == 1.0


def test_resize_past_the_world_zooms_in_and_keeps_the_camera_inside():
    ww, wh = config.WORLD_WIDTH, config.WORLD_HEIGHT
    view = ViewState(cam_x=ww - config.SCREEN_WIDTH, cam_y=wh - config.SCREEN_HEIGHT)
    view.resize((ww * 2, wh))
    assert view.zoom == pytest.approx(2.0)
    vw, vh = view.view_size()
    assert (vw, vh) == pytest.approx((ww, wh / 2))
    assert 0 <= view.cam_x <= ww - vw and 0 <= view.cam_y <= wh - vh


def test_zoom_out_is_limited_by_the_resized_window():
    view = ViewState()
    view.resize((config.WORLD_WIDTH, config.WORLD_HEIGHT))
    view.zoom_by(-10, (0, 0))
    assert view.zoom == 1.0
    assert view.view_size() == (config.WORLD_WIDTH, config.WORLD_HEIGHT)


def test_camera_moves_to_the_leader_after_a_repopulation():
    sim = bootstrap.start_simulation(None, world_bounds=(400, 300))
    view = ViewState(selected=sim.world.herbivores[3])
    data = NeuralNetwork(config.BRAIN_TOPOLOGY).to_data()
    bootstrap.reload_population(sim, HERBIVORE, data=data)  # every founder scores 0
    view.follow_selected(sim.world)
    assert view.tentative_pick is view.selected
    sim.step()
    view.follow_selected(sim.world)
    assert view.selected.is_best and view.tentative_pick is None
