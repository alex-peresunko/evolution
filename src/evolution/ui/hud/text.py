"""The HUD's text: world statistics and the selected creature's details."""
from ... import config
from ...species import CARNIVORE, HERBIVORE
from ...world.environment import food_cap, season_name


def time_of_day(light):
    phase = "Day" if light > 0.66 else "Dusk/Dawn" if light > 0.33 else "Night"
    return f"Time: {phase} ({int(light * 100)}% light)"


def stats_lines(sim, view, fps):
    world, stats = sim.world, sim.stats
    limit_status = "Limited (60)" if view.fps_limited else "Unlimited"
    return [
        f"Generation: {sim.generation}   Extinctions: {sim.extinctions}",
        f"Tick: {world.tick}",
        f"Herbivores: {len(world.herbivores)}",
        f"Carnivores: {len(world.carnivores)}",
        f"Lineages: H {stats[HERBIVORE].lineages}  C {stats[CARNIVORE].lineages}",
        f"Food: {len(world.food)}   Carcasses: {len(world.carcasses)}",
        time_of_day(world.light),
        f"Season: {season_name(world.tick)} (food cap {food_cap(world.season):.0f})",
        f"Top Herbivore Score: {max([c.score for c in world.herbivores] + [0]):.2f}",
        f"Top Carnivore Score: {max([c.score for c in world.carnivores] + [0]):.2f}",
        f"Longest H: {stats[HERBIVORE].max_age} ticks",
        f"Longest C: {stats[CARNIVORE].max_age} ticks",
        f"FPS: {int(fps)} [{limit_status}]",
        f"Speed: {view.speed_label()}  ({view.tps} ticks/s)",
        f"Zoom: x{view.zoom:.2f}",
    ]


def creature_info(c):
    status = "Adult" if c.is_adult else f"Juvenile ({c.age}/{config.ADULTHOOD_AGE})"
    info = {
        "ID": c.id,
        "Type": c.species.title,
        "Status": status,
        "Health": int(c.health),
        "Stamina": f"{int(c.stamina)}/{int(c.max_stamina)}" + (" winded" if c.winded else ""),
        "Swimming": c.in_water,
        "Score": round(c.score, 2),
        "Eating": c.eating_ticks,
        "Fat": f"{int(c.fat)}/{config.FAT_MAX}",
        "Upkeep": round(c.upkeep(), 2),
        "Signal": round(c.signal, 2),
        "Memory": " ".join(f"{m:+.2f}" for m in c.memory),
        "Reproductions": c.reproductions,
        "Generation": c.generation,
        "Lineage": f"#{c.lineage} (descent {c.descent})",
        "Is Best": c.is_best,
    }
    for gene, value in c.genes.items():
        info[f"Gene: {gene}"] = round(value, 2)
    return info
