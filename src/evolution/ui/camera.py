"""Camera math: which creature to follow, and where the viewport goes."""


def follow_target(world, creature, tentative=False):
    """(creature to follow, tentative): itself while it lives in the population, else the
    highest-scoring creature of its species (None if there is none). A pick is tentative while
    the whole species is tied on score (just repopulated, so there is no best yet); a tentative
    creature is re-picked on every call until one leads."""
    if creature is None:
        return None, False
    population = world.population(creature.species)
    # Membership matters too: island immigrants and reloads replace creatures without killing them.
    if not tentative and creature.is_alive() and any(c is creature for c in population):
        return creature, False
    best = max(population, key=lambda c: c.score, default=None)
    return best, best is not None and all(c.score == best.score for c in population)


def clamp_origin(x, y, world_size, view_size):
    """Camera top-left kept so the view (in world units) stays inside the world."""
    (ww, wh), (vw, vh) = world_size, view_size
    return max(0.0, min(float(ww - vw), x)), max(0.0, min(float(wh - vh), y))


def viewport_origin(wx, wy, world_size, view_size):
    """Top-left camera position that centers (wx, wy), clamped to the world."""
    vw, vh = view_size
    return clamp_origin(wx - vw / 2, wy - vh / 2, world_size, view_size)


def min_zoom(world_size, screen_size):
    """Smallest zoom at which the view still fits inside the world."""
    (ww, wh), (sw, sh) = world_size, screen_size
    return max(sw / ww, sh / wh)


def zoom_at(cam, zoom, new_zoom, anchor, world_size, screen_size):
    """Camera origin after zooming from zoom to new_zoom, keeping the world point under the
    screen point anchor in place (as far as the world edges allow)."""
    wx, wy = cam[0] + anchor[0] / zoom, cam[1] + anchor[1] / zoom
    view = (screen_size[0] / new_zoom, screen_size[1] / new_zoom)
    return clamp_origin(wx - anchor[0] / new_zoom, wy - anchor[1] / new_zoom, world_size, view)
