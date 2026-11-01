"""Named columns of the brain's input and output vectors. Saved brains are wired to these
positions, so new fields go at the end (NeuralNetwork.from_data keeps the overlapping block)."""
from ..config import BRAIN_INPUTS, BRAIN_OUTPUTS, NUM_MEMORY, NUM_WHISKERS


class Layout:
    """Each field becomes an attribute: a column index for scalars, a slice for groups.
    `labels` names every column, for the brain panel."""

    def __init__(self, fields):
        self.labels = []
        for name, width, label in fields:
            start = len(self.labels)
            if width == 1:
                setattr(self, name, start)
                self.labels.append(label)
            else:
                setattr(self, name, slice(start, start + width))
                self.labels += [label.format(i) for i in range(width)]
        self.width = len(self.labels)


INPUTS = Layout([
    ("obstacle_whiskers", NUM_WHISKERS, "obst_w{}"),  # 1 = touching
    # carnivore: nearest visible prey; herbivore: food angle blended away from the nearest
    # visible predator, and food distance
    ("target_angle", 1, "target_ang"),
    ("target_dist", 1, "target_dist"),
    ("smell_angle", 1, "smell_ang"),                  # prey (carnivore) / predator (herbivore)
    ("smell_strength", 1, "smell_str"),
    ("prey_vx", 1, "prey_vx"),                        # carnivore only
    ("prey_vy", 1, "prey_vy"),
    ("health", 1, "health"),
    ("sin_angle", 1, "sin_ang"),
    ("cos_angle", 1, "cos_ang"),
    ("size", 1, "size"),
    ("speed", 1, "speed"),
    ("sight", 1, "sight"),
    ("sight_angle", 1, "sight_ang"),
    ("stamina", 1, "stamina"),
    ("hunger", 1, "hunger"),
    ("kin_angle", 1, "kin_ang"),
    ("kin_dist", 1, "kin_dist"),
    ("kin_density", 1, "kin_density"),
    ("signal_angle", 1, "signal_ang"),
    ("signal_strength", 1, "signal_str"),
    ("carcass_angle", 1, "carcass_ang"),              # carnivore only
    ("carcass_dist", 1, "carcass_dist"),
    ("light", 1, "light"),                            # 0 = midnight, 1 = noon
    ("memory", NUM_MEMORY, "mem{}_in"),               # own memory outputs of the last tick
    ("target_whiskers", NUM_WHISKERS, "food_w{}"),    # food / prey or carcass along each whisker
    ("season", 1, "season"),                          # -1 = midwinter, 1 = midsummer
    ("carcass_smell_angle", 1, "carcass_smell_ang"),  # carnivore only
    ("carcass_smell_strength", 1, "carcass_smell_str"),
    ("in_water", 1, "in_water"),
    ("eating", 1, "eating"),                          # share of the meal left; 0 = not eating
    ("hurt", 1, "hurt"),                              # 1 just after a bite or retaliation, fading
    # the closest-pressing body of the own / the other species (world.perception._touch)
    ("touch_kin", 1, "touch_kin"),
    ("touch_kin_angle", 1, "touch_kin_ang"),
    ("touch_other", 1, "touch_other"),
    ("touch_other_angle", 1, "touch_other_ang"),
])

OUTPUTS = Layout([
    ("rotation", 1, "turn"),
    ("speed", 1, "speed"),
    ("signal", 1, "signal"),
    ("memory", NUM_MEMORY, "mem{}"),
    ("stop_eating", 1, "stop_eat"),                   # > MEAL_BREAK_THRESHOLD abandons a meal
])

assert INPUTS.width == BRAIN_INPUTS and OUTPUTS.width == BRAIN_OUTPUTS, \
    "config.BRAIN_INPUTS / BRAIN_OUTPUTS must match the layouts"
