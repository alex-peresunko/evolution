import pygame

from ... import config
from ...world.terrain import Tree

MINIMAP_AREA = (config.MINIMAP_X, config.MINIMAP_Y, config.MINIMAP_WIDTH, config.MINIMAP_HEIGHT)


def contains(pos):
    return pygame.Rect(MINIMAP_AREA).collidepoint(pos)


def to_world(pos):
    """The world point under a screen point on the minimap."""
    return ((pos[0] - config.MINIMAP_X) / config.MINIMAP_WIDTH * config.WORLD_WIDTH,
            (pos[1] - config.MINIMAP_Y) / config.MINIMAP_HEIGHT * config.WORLD_HEIGHT)


def draw_minimap(screen, background, world, view):
    mmap = background.copy()
    sx = config.MINIMAP_WIDTH / config.WORLD_WIDTH
    sy = config.MINIMAP_HEIGHT / config.WORLD_HEIGHT
    for obs in world.obstacles:
        if isinstance(obs, Tree):
            pygame.draw.circle(mmap, config.COLOR_TREE, (int(obs.x * sx), int(obs.y * sy)),
                               max(2, int(obs.crown_radius * sx)))
            continue
        mr = pygame.Rect(int(obs.rect.x * sx), int(obs.rect.y * sy),
                         max(2, int(obs.rect.width * sx)), max(2, int(obs.rect.height * sy)))
        pygame.draw.rect(mmap, config.COLOR_OBSTACLE, mr)
    for c in world.creatures():
        if c.is_best:
            pygame.draw.circle(mmap, config.COLOR_BEST_CREATURE,
                               (int(c.pos.x * sx), int(c.pos.y * sy)), 3)
    vw, vh = view.view_size()
    vr = pygame.Rect(int(view.cam_x * sx), int(view.cam_y * sy),
                     max(1, int(vw * sx)), max(1, int(vh * sy)))
    pygame.draw.rect(mmap, config.COLOR_TEXT, vr, 1)
    screen.blit(mmap, (config.MINIMAP_X, config.MINIMAP_Y))
    pygame.draw.rect(screen, config.COLOR_TEXT, MINIMAP_AREA, 1)
