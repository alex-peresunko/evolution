import pytest

from evolution import config
from evolution.ui.view_state import SPEEDS, ViewState


def test_speed_steps_through_the_presets_and_clamps():
    view = ViewState()
    view.change_speed(-1)
    assert view.ticks_per_frame == SPEEDS[0]
    for _ in range(len(SPEEDS) + 2):
        view.change_speed(1)
    assert view.ticks_per_frame == SPEEDS[-1]


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
