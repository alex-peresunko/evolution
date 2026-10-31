"""Bit-exact behaviour guard: the simulation, imitation and rendering must reproduce the
committed golden trace (tests/golden/trace.json)."""
import json

import pytest

from tests.golden import record


@pytest.fixture(scope="module")
def trace():
    with open(record.TRACE) as f:
        return json.load(f)


def _first_difference(expected, actual):
    for (label, fields), (got_label, got_fields) in zip(expected, actual):
        assert got_label == label
        changed = sorted(f for f in fields if fields[f] != got_fields.get(f))
        if changed:
            return f"{label}: {', '.join(changed)}"
    if len(expected) != len(actual):
        return f"{len(actual)} observations instead of {len(expected)}"
    return None


def test_evolution_matches_golden_trace(trace):
    assert _first_difference(trace["evolution"], record.capture_evolution()) is None


def test_imitation_matches_golden_trace(trace):
    actual = record.capture_imitation()
    assert {f for f, d in trace["imitation"].items() if actual.get(f) != d} == set()


def test_rendering_matches_golden_trace(trace):
    pytest.importorskip("pygame")
    if record.render_platform() != trace["render"]["platform"]:
        pytest.skip("golden frames were recorded on another platform")
    assert _first_difference(trace["render"]["frames"], record.capture_render()) is None
