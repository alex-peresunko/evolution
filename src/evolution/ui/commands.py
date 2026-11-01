"""What each key and mouse action does. KEY_COMMANDS is the whole keyboard map."""
import pygame

from ..app.bootstrap import reload_population
from ..persistence.brain_store import save_brain
from ..species import ALL_SPECIES
from ..world.food import Food
from .hud import minimap


def save_living_best(app):
    """Save each species' best living creature over its best_brain file."""
    if app.read_only:
        print("Saving is disabled in watch mode.")
        return
    sim = app.sim
    for species in ALL_SPECIES:
        best = max(sim.world.population(species), key=lambda c: c.score, default=None)
        if best:
            save_brain(best.brain, species.brain_file, generation=sim.generation,
                       genes=best.genes, score=best.score,
                       uptime_seconds=app.metrics.uptime_seconds(),
                       history=sim.stats[species].saved_history())


def reload_species(app, species_list):
    """Restart the given species from their best_brain files, logging each one."""
    generations = []
    for species in species_list:
        progenitor = reload_population(app.sim, species)
        if progenitor is None:
            print(f"No {species.name} brain file found to load.")
        else:
            generations.append(progenitor.generation)
            print(f"Loaded new {species.name} population from generation "
                  f"{progenitor.generation}.")
    if generations:
        app.sim.generation = max(generations)


def reload_best(app):
    """Restart each species from its best_brain file."""
    reload_species(app, ALL_SPECIES)


def toggle_pause(app):
    app.view.paused = not app.view.paused


def faster(app):
    app.view.change_speed(1)


def slower(app):
    app.view.change_speed(-1)


def grow_charts(app):
    app.view.scale_charts(1)


def shrink_charts(app):
    app.view.scale_charts(-1)


def toggle_fps_limit(app):
    app.view.fps_limited = not app.view.fps_limited
    print(f"FPS Limiting {'Enabled' if app.view.fps_limited else 'Disabled'}")


def toggle_genes(app):
    app.view.show_genes = not app.view.show_genes


def toggle_brain(app):
    app.view.show_brain = not app.view.show_brain


def toggle_drawing(app):
    app.view.drawing_enabled = not app.view.drawing_enabled
    print(f"Drawing {'Enabled' if app.view.drawing_enabled else 'Disabled'}")


def toggle_background_mode(app):
    if app.read_only:
        print("Island workers are disabled in watch mode.")
        return
    if app.workers.running:
        app.workers.stop()
        print("--- Island Workers Stopped ---")
    else:
        app.workers.start()
        print("--- ISLAND WORKERS ENGAGED ---")


KEY_COMMANDS = {
    pygame.K_s: save_living_best,
    pygame.K_l: reload_best,
    pygame.K_SPACE: toggle_pause,
    pygame.K_EQUALS: faster,
    pygame.K_PLUS: faster,
    pygame.K_KP_PLUS: faster,
    pygame.K_MINUS: slower,
    pygame.K_KP_MINUS: slower,
    pygame.K_RIGHTBRACKET: grow_charts,
    pygame.K_LEFTBRACKET: shrink_charts,
    pygame.K_RETURN: toggle_fps_limit,
    pygame.K_g: toggle_genes,
    pygame.K_n: toggle_brain,
    pygame.K_d: toggle_drawing,
    pygame.K_b: toggle_background_mode,
}


def click(app, pos):
    """On the minimap: pan there (and stop following). In the world: select or deselect the
    creature under the cursor, or else drop food there."""
    view, world = app.view, app.sim.world
    if minimap.contains(pos):
        view.minimap_dragging = True
        view.selected = None
        view.center_on(*minimap.to_world(pos))
        return
    wx, wy = view.screen_to_world(pos)
    for c in world.creatures():
        if c.pos.distance_to((wx, wy)) < c.size:
            view.selected = c if view.selected is not c else None
            return
    view.selected = None
    food = Food(world.bounds, geometry=world.geometry)  # draws a random position, as spawns do
    food.pos = world.geometry.vec2(wx, wy)
    world.food.append(food)


def handle_events(app):
    view = app.view
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            view.running = False
        elif event.type == pygame.KEYDOWN:
            command = KEY_COMMANDS.get(event.key)
            if command:
                command(app)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            click(app, event.pos)
        elif event.type == pygame.MOUSEMOTION and view.minimap_dragging:
            view.center_on(*minimap.to_world(event.pos))
        elif event.type == pygame.MOUSEWHEEL and event.y:
            view.zoom_by(event.y, pygame.mouse.get_pos())
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            view.minimap_dragging = False
