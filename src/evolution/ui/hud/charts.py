"""History charts: Hall of Fame score and longevity per species, and gene drift."""
import pygame

from ... import config
from .panels import draw_panel, font


def draw_dual_chart(screen, rect, title, series, scale, label):
    """Translucent panel with one line per (history, color, label) series, each scaled to its
    own peak (species scores differ by orders of magnitude), sized by the zoomable chart scale."""
    chart_font = font(round(config.CHART_FONT_SIZE * scale))
    line_h = chart_font.get_linesize()
    pad = round(4 * scale)
    draw_panel(screen, rect)
    screen.blit(chart_font.render(title, True, config.COLOR_TEXT), (rect.x + pad, rect.y + pad))
    gens = chart_font.render(label, True, config.COLOR_TEXT)
    screen.blit(gens, (rect.right - gens.get_width() - pad, rect.y + pad))
    info_y = rect.y + pad + line_h
    for i, (history, color, name) in enumerate(series):  # peaks: first left, second right
        peak = chart_font.render(f"{name}: {int(max(history, default=0))}", True, color)
        x = rect.x + pad if i == 0 else rect.right - peak.get_width() - pad
        screen.blit(peak, (x, info_y))
    top, bottom = info_y + line_h + pad, rect.bottom - pad
    left, width = rect.x + pad, rect.width - pad * 2
    for history, color, _ in series:
        if len(history) < 2:
            continue
        max_score = max(history) or 1
        points = [
            (left + i / (len(history) - 1) * width, bottom - (score / max_score) * (bottom - top))
            for i, score in enumerate(history)
        ]
        pygame.draw.lines(screen, color, False, points, max(1, round(scale)))


def draw_gene_chart(screen, rect, species, gene_history, scale):
    """Species mean of each gene as a ratio to its default, on one shared axis with a dashed
    1.0 baseline: shows which traits selection pushes up or down."""
    chart_font = font(round(config.CHART_FONT_SIZE * scale))
    line_h = chart_font.get_linesize()
    pad = round(4 * scale)
    draw_panel(screen, rect)
    screen.blit(chart_font.render(f"Gene drift - {species.plural} (x default)", True,
                                  config.COLOR_TEXT),
                (rect.x + pad, rect.y + pad))
    defaults = species.default_genes
    series = [(gene, color, [v / defaults[gene] for v in gene_history.get(gene, ())])
              for gene, color in config.GENE_CHART_COLORS.items()]
    legend_font = font(round(config.CHART_FONT_SIZE * scale * 0.8))
    legend_h = legend_font.get_linesize()
    x, y = rect.x + pad, rect.y + pad + line_h
    for gene, color, ratios in series:  # legend, wrapped to the panel width
        label = legend_font.render(f"{gene} {ratios[-1]:.2f}" if ratios else gene, True, color)
        if x + label.get_width() > rect.right - pad:
            x, y = rect.x + pad, y + legend_h
        screen.blit(label, (x, y))
        x += label.get_width() + pad * 3
    top, bottom = y + legend_h + pad, rect.bottom - pad
    left, width = rect.x + pad, rect.width - pad * 2
    values = [r for _, _, ratios in series for r in ratios] + [1.0]
    lo, hi = min(values), max(values)
    span = (hi - lo) or 1.0
    lo, hi = lo - span * 0.05, hi + span * 0.05

    def to_y(r):
        return bottom - (r - lo) / (hi - lo) * (bottom - top)

    base_y = to_y(1.0)
    dash = round(6 * scale)
    for dx in range(0, width, dash * 2):
        pygame.draw.line(screen, (150, 150, 150), (left + dx, base_y),
                         (left + min(width, dx + dash), base_y))
    for _, color, ratios in series:
        if len(ratios) > 1:
            points = [(left + i / (len(ratios) - 1) * width, to_y(r))
                      for i, r in enumerate(ratios)]
            pygame.draw.lines(screen, color, False, points, max(1, round(scale)))
