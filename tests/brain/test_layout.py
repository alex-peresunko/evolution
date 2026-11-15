from evolution import config
from evolution.brain.layout import CALL_OUTPUTS, INPUTS, OUTPUTS


def test_layout_widths_match_the_topology():
    assert INPUTS.width == len(INPUTS.labels) == config.BRAIN_TOPOLOGY[0]
    assert OUTPUTS.width == len(OUTPUTS.labels) == config.BRAIN_TOPOLOGY[-1]


def test_groups_are_slices_and_scalars_are_columns():
    assert INPUTS.obstacle_whiskers == slice(0, config.NUM_WHISKERS)
    assert INPUTS.labels[INPUTS.target_whiskers] == [
        f"food_w{i}" for i in range(config.NUM_WHISKERS)]
    assert INPUTS.labels[INPUTS.touch_other_angle] == "touch_other_ang"
    assert (OUTPUTS.rotation, OUTPUTS.speed, OUTPUTS.signal) == (0, 1, 2)
    assert OUTPUTS.memory == slice(3, 3 + config.NUM_MEMORY)
    assert OUTPUTS.stop_eating == 3 + config.NUM_MEMORY


def test_every_call_channel_has_an_output_and_kin_inputs():
    assert len(CALL_OUTPUTS) == len(set(CALL_OUTPUTS)) == config.NUM_SIGNALS
    assert CALL_OUTPUTS[0] == OUTPUTS.signal
    for n in range(1, config.NUM_SIGNALS):
        assert OUTPUTS.labels[CALL_OUTPUTS[n]] == f"call{n}"
        assert getattr(INPUTS, f"call{n}_strength") == getattr(INPUTS, f"call{n}_angle") + 1


def test_labels_are_unique():
    assert len(set(INPUTS.labels)) == INPUTS.width
    assert len(set(OUTPUTS.labels)) == OUTPUTS.width
