import pygame

from ... import config
from ...species import CARNIVORE, HERBIVORE
from . import brain_panel
from .charts import draw_dual_chart, draw_gene_chart
from .minimap import draw_minimap
from .panels import draw_faded, draw_text_block
from .text import creature_info, stats_lines


def draw_hud(screen, font, sim, view, minimap_background, fps):
    """Minimap, statistics, the selected creature's panels, charts and the pause banner."""
    draw_minimap(screen, minimap_background, sim.world, view)
    draw_text_block(screen, font, stats_lines(sim, view, fps), 10,
                    config.MINIMAP_Y + config.MINIMAP_HEIGHT + 15)
    selected = view.selected
    if selected:
        info_lines = [f"{k}: {v}" for k, v in creature_info(selected).items()]
        draw_text_block(screen, font, info_lines, None, 16, title="Selected Creature",
                        right=config.SCREEN_WIDTH - 10)
        if view.show_brain and selected.brain_inputs is not None:
            draw_faded(screen, brain_panel.panel_rect(), config.HUD_ALPHA,
                       brain_panel.draw_brain_panel, selected, view.brain_history)
    scale = view.chart_scale
    chart_w, chart_h = round(config.CHART_SIZE[0] * scale), round(config.CHART_SIZE[1] * scale)
    spacing = round(5 * scale)
    label = f"Gen {sim.generation}  Ext {sim.extinctions}"
    for i, (title, metric) in enumerate([("HoF Score", "score_history"),
                                         ("Max Longevity", "longevity_history")]):
        rect = pygame.Rect(config.SCREEN_WIDTH - chart_w - 10,
                           config.SCREEN_HEIGHT - 10 - chart_h * (i + 1) - spacing * i,
                           chart_w, chart_h)
        series = [(getattr(sim.stats[HERBIVORE], metric), config.COLOR_CREATURE, "Herb"),
                  (getattr(sim.stats[CARNIVORE], metric), config.COLOR_CARNIVORE, "Carn")]
        draw_faded(screen, rect, config.HUD_ALPHA, draw_dual_chart, rect, title, series,
                   scale, label)
    if view.show_genes:
        species = selected.species if selected else HERBIVORE
        gene_w = round(chart_w * 1.6)
        rect = pygame.Rect(config.SCREEN_WIDTH - chart_w - gene_w - 20,
                           config.SCREEN_HEIGHT - 10 - chart_h * 2 - spacing, gene_w,
                           chart_h * 2 + spacing)
        draw_faded(screen, rect, config.HUD_ALPHA, draw_gene_chart, rect, species,
                   sim.stats[species].gene_history, scale)
    if view.paused:
        screen.blit(font.render("PAUSED", True, (255, 100, 100)),
                    (config.SCREEN_WIDTH // 2 - 60, 20))
