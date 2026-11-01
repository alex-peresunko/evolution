import math

from evolution.app import bootstrap
from evolution.ui.render.motion import MotionSmoother


def _sim():
    return bootstrap.start_simulation(None, world_bounds=(400, 300))


def test_poses_move_through_the_last_tick():
    sim = _sim()
    smoother = MotionSmoother()
    smoother.before_tick(sim.world)
    before = {c.id: (c.pos.x, c.pos.y) for c in sim.world.creatures()}
    sim.step()
    for c in sim.world.creatures():
        if c.id not in before:
            continue
        x0, y0 = before[c.id]
        assert smoother.poses(sim.world, 0.0)[c.id][:2] == (x0, y0)
        x, y, _ = smoother.poses(sim.world, 0.5)[c.id]
        assert math.isclose(x, (x0 + c.pos.x) / 2) and math.isclose(y, (y0 + c.pos.y) / 2)
        angle = smoother.poses(sim.world, 1.0)[c.id][2]
        assert math.isclose(math.remainder(angle - c.angle, 2 * math.pi), 0, abs_tol=1e-9)


def test_heading_turns_the_short_way_round():
    sim = _sim()
    c = sim.world.creatures()[0]
    smoother = MotionSmoother()
    c.angle = math.pi - 0.1
    smoother.before_tick(sim.world)
    sim.world.tick += 1
    c.angle = -math.pi + 0.1  # 0.2 rad further on, across the ±pi seam
    angle = smoother.poses(sim.world, 0.5)[c.id][2]
    assert math.isclose(angle, math.pi)


def test_stale_snapshot_draws_real_poses():
    sim = _sim()
    smoother = MotionSmoother()
    smoother.before_tick(sim.world)
    sim.step()
    sim.step()  # a tick taken without a snapshot (e.g. at a speed of 1 or more)
    assert smoother.poses(sim.world, 0.5) is None
