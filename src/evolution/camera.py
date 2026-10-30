"""Camera helpers for game.py: which creature to follow, and where the viewport goes."""


def follow_target(sim_state, creature):
    """The creature to keep following: itself while it lives in the population, else the
    highest-scoring creature of its species (None if there is none)."""
    if creature is None:
        return None
    population = sim_state["carnivores" if creature.is_carnivore else "creatures"]
    # Membership matters too: island immigrants replace creatures without killing them.
    if creature.is_alive() and any(c is creature for c in population):
        return creature
    return max(population, key=lambda c: c.score, default=None)


def viewport_origin(wx, wy, world_size, screen_size):
    """Top-left camera position that centers (wx, wy), clamped to the world."""
    (ww, wh), (sw, sh) = world_size, screen_size
    return (max(0.0, min(float(ww - sw), wx - sw / 2)),
            max(0.0, min(float(wh - sh), wy - sh / 2)))
