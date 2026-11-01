from evolution import config
from evolution.brain.layout import INPUTS, OUTPUTS


def test_layout_widths_match_the_topology():
    assert INPUTS.width == len(INPUTS.labels) == config.BRAIN_TOPOLOGY[0]
    assert OUTPUTS.width == len(OUTPUTS.labels) == config.BRAIN_TOPOLOGY[-1]


def test_groups_are_slices_and_scalars_are_columns():
    assert INPUTS.obstacle_whiskers == slice(0, config.NUM_WHISKERS)
    assert INPUTS.labels[INPUTS.target_whiskers] == [
        f"food_w{i}" for i in range(config.NUM_WHISKERS)]
    assert INPUTS.eating == INPUTS.width - 1
    assert (OUTPUTS.rotation, OUTPUTS.speed, OUTPUTS.signal) == (0, 1, 2)
    assert OUTPUTS.memory == slice(3, 3 + config.NUM_MEMORY)
    assert OUTPUTS.stop_eating == OUTPUTS.width - 1


def test_labels_are_unique():
    assert len(set(INPUTS.labels)) == INPUTS.width
    assert len(set(OUTPUTS.labels)) == OUTPUTS.width
