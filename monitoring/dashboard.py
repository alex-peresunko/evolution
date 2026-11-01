"""Generates grafana_dashboard.json; edit the panels here, then run:
    python monitoring/dashboard.py
"""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "grafana_dashboard.json"
DASHBOARD_UID = "6a418c28-cbeb-46f9-a1f4-af86d6e6d051"
# Must match monitoring/grafana/provisioning/datasources/prometheus.yml and the scrape job.
DATASOURCE = {"type": "prometheus", "uid": "fev2gbax7os8we"}
JOB = 'job="evolution"'

SPECIES_COLORS = {"herbivore": "green", "carnivore": "red"}
CAUSE_COLORS = {"bitten": "red", "retaliation": "orange", "starved": "yellow",
                "old_age": "blue", "attrition": "purple"}


def m(name, *matchers, **labels):
    """A metric selector limited to the evolution job; `matchers` are raw, e.g. 'gene!="color"'."""
    selectors = [JOB, *matchers] + [f'{k}="{v}"' for k, v in labels.items()]
    return f"evolution_{name}{{{','.join(selectors)}}}"


def q(expr, legend=""):
    return (expr, legend)


def _colors(mapping):
    return [{"matcher": {"id": "byRegexp", "options": f".*{key}.*"},
             "properties": [{"id": "color", "value": {"mode": "fixed", "fixedColor": color}}]}
            for key, color in mapping.items()]


def timeseries(title, description, *queries, unit="none", stack=False, colors=SPECIES_COLORS,
               min_=None, bars=False, dashed=None, right_axis=None, w=12, h=8):
    custom = {"drawStyle": "bars" if bars else "line", "lineWidth": 2,
              "fillOpacity": 60 if bars else (25 if stack else 8), "showPoints": "never",
              "spanNulls": True, "stacking": {"mode": "normal" if stack else "none"}}
    defaults = {"unit": unit, "custom": custom, "color": {"mode": "palette-classic"}}
    if min_ is not None:
        defaults["min"] = min_
    overrides = _colors(colors or {})
    if dashed:
        overrides.append({"matcher": {"id": "byRegexp", "options": f".*{dashed}.*"},
                          "properties": [{"id": "custom.lineStyle",
                                          "value": {"fill": "dash", "dash": [10, 10]}}]})
    if right_axis:
        overrides.append({"matcher": {"id": "byName", "options": right_axis},
                          "properties": [{"id": "custom.axisPlacement", "value": "right"},
                                         {"id": "color", "value": {"mode": "fixed",
                                                                   "fixedColor": "#b8a77a"}}]})
    return _panel("timeseries", title, description, queries, w, h,
                  {"defaults": defaults, "overrides": overrides},
                  {"legend": {"displayMode": "list", "placement": "bottom", "showLegend": True},
                   "tooltip": {"mode": "multi", "sort": "desc"}})


def stat(title, description, *queries, unit="none", color="text", decimals=None, no_value=None,
         w=4, h=4):
    defaults = {"unit": unit, "color": {"mode": "fixed", "fixedColor": color}}
    if decimals is not None:
        defaults["decimals"] = decimals
    if no_value is not None:
        defaults["noValue"] = no_value
    return _panel("stat", title, description, queries, w, h,
                  {"defaults": defaults, "overrides": _colors(SPECIES_COLORS)},
                  {"reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False},
                   "colorMode": "value", "graphMode": "area", "textMode": "value_and_name"
                   if len(queries) > 1 else "value", "justifyMode": "center"})


def _panel(kind, title, description, queries, w, h, field_config, options):
    return {"type": kind, "title": title, "description": description, "datasource": DATASOURCE,
            "gridPos": {"w": w, "h": h}, "fieldConfig": field_config, "options": options,
            "targets": [{"refId": chr(ord("A") + i), "datasource": DATASOURCE, "expr": expr,
                         "legendFormat": legend, "range": True}
                        for i, (expr, legend) in enumerate(queries)]}


def row(title, *panels, collapsed=False):
    return {"title": title, "panels": list(panels), "collapsed": collapsed}


def per_minute(expr, window="2m"):
    return f"rate({expr}[{window}]) * 60"


def by_species(name):
    return [q(m(name, species=s), s) for s in SPECIES_COLORS]


ROWS = [
    row(
        "Overview",
        stat("Uptime", "How long this simulation process has been running.",
             q(m("running_seconds_total")), unit="s"),
        stat("Generation", "The deepest generation born so far: the longest chain of "
             "parent-to-child births since the founders.",
             q(m("generation")), color="purple"),
        stat("Ticks per second", "Simulation speed. One tick is one step of the world.",
             q(f"rate({m('ticks_total')}[1m])"), decimals=0, color="blue"),
        stat("Population", "Living creatures of each species right now.",
             *by_species("population"), w=5),
        stat("Extinctions", "How many times a species died out in the selected time range. "
             "The simulation refills it from its Hall of Fame right away.",
             q(f"sum(increase({m('extinctions_total')}[$__range]))"), decimals=0, w=3,
             color="orange"),
        stat("Last brain save", "Time since the best brain files in saved_brains/ were last "
             "overwritten by a new Hall of Fame record.",
             q(f"time() - max({m('last_save_timestamp_seconds')})"), unit="s",
             no_value="none this run", color="text"),
    ),
    row(
        "Evolution progress",
        *[timeseries(f"{s.title()} scores", f"Score of the living {s}s: the best one, the mean "
                     "and the median. When only the best rises, one lucky creature carries the "
                     "chart; when the mean and median follow, the whole population improves.",
                     q(m("score", species=s, stat="best"), "best"),
                     q(m("score", species=s, stat="mean"), "mean"),
                     q(m("score", species=s, stat="median"), "median"),
                     colors={"best": SPECIES_COLORS[s], "mean": "blue", "median": "purple"})
          for s in SPECIES_COLORS],
        timeseries("Hall of Fame vs saved brain", "Solid: the Hall of Fame best score (the best "
                   "creature that ever died, fading with HOF_SCORE_HALF_LIFE so old luck "
                   "expires). Dashed: the score of the brain last written to saved_brains/. "
                   "A new record above the dashed line is what gets saved next.",
                   *[q(m("hall_of_fame_score", species=s), f"{s} Hall of Fame")
                     for s in SPECIES_COLORS],
                   *[q(m("saved_score", species=s), f"{s} saved") for s in SPECIES_COLORS],
                   dashed="saved", w=8),
        timeseries("Mean lifespan at death", "Average age, in ticks, of the creatures that died "
                   "in the last 5 minutes. For herbivores this is the cleanest single sign of "
                   "fitness: they live longer once they find food and avoid carnivores.",
                   q(f"sum by (species) (rate({m('death_age_ticks_total')}[5m])) "
                     f"/ sum by (species) (rate({m('deaths_total')}[5m]))", "{{species}}"),
                   w=8, min_=0),
        timeseries("Generations per hour", "How fast new generations are born. A drop means "
                   "creatures breed later or less often.",
                   q(f"deriv({m('generation')}[15m]) * 3600", "generations / hour"),
                   colors={"generations": "purple"}, w=8, min_=0),
    ),
    row(
        "Life and death",
        *[timeseries(f"{s.title()} deaths by cause", "Deaths per minute, stacked by likely "
                     "cause. bitten: died in the tick a carnivore bit it. retaliation: a "
                     "carnivore killed by large prey fighting back. starved: hunger past "
                     "HUNGER_MAX. old_age: older than its lifespan gene. attrition: everything "
                     "else (upkeep, running, signalling, hitting walls and rocks). A shift from "
                     "starved to old_age means the species learned to eat.",
                     q(f"sum by (cause) ({per_minute(m('deaths_total', species=s))})",
                       "{{cause}}"),
                     stack=True, colors=CAUSE_COLORS, unit="short", min_=0)
          for s in SPECIES_COLORS],
        timeseries("Births vs deaths", "Births (solid) and deaths (dashed) per minute. A "
                   "species grows while its births are above its deaths.",
                   *[q(per_minute(m("births_total", species=s)), f"{s} births")
                     for s in SPECIES_COLORS],
                   *[q(f"sum({per_minute(m('deaths_total', species=s))})", f"{s} deaths")
                     for s in SPECIES_COLORS],
                   dashed="deaths", unit="short", min_=0),
        timeseries("Extinctions", "Extinctions per 10 minutes. Each one wipes out the living "
                   "population, which restarts from Hall of Fame elites and random brains.",
                   q(f"increase({m('extinctions_total')}[10m])", "{{species}}"),
                   bars=True, unit="short", min_=0),
    ),
    row(
        "Diversity",
        timeseries("Live lineages", "Founder families still alive: every creature inherits the "
                   "founder id of its first parent. When this falls to 1, one family has taken "
                   "over, and evolution often plateaus soon after.",
                   q(m("lineages"), "{{species}}"), min_=0),
        timeseries("Genetic diversity", "RMS genetic distance between two random members of the "
                   "species, with genes scaled by their allowed range. The dashed line is "
                   "SPECIATION_DISTANCE: pairs farther apart than that cannot breed. A collapse "
                   "toward 0 means the population converged on one genome.",
                   q(m("genetic_diversity"), "{{species}}"),
                   q(m("speciation_distance"), "speciation distance"),
                   colors={**SPECIES_COLORS, "speciation": "#808080"}, dashed="speciation",
                   min_=0),
    ),
    row(
        "Predators and prey",
        timeseries("Populations and food", "Living herbivores and carnivores, with food items on "
                   "the right axis. Predator and prey numbers tend to chase each other in "
                   "cycles: carnivores boom after herbivores do, then herbivores recover.",
                   *by_species("population"), q(m("food"), "food"), right_axis="food",
                   min_=0),
        timeseries("Season and daylight", "Season runs from -1 (midwinter) to 1 (midsummer) and "
                   "scales how much food grows. Daylight runs from 0 (night) to 1 (day); at "
                   "night vision is shorter.",
                   q(m("season"), "season"), q(m("daylight"), "daylight"),
                   colors={"season": "green", "daylight": "yellow"}, min_=-1),
        timeseries("Hunt success", "Herbivores killed by a bite divided by bites landed, over 5 "
                   "minutes. Prey usually takes several bites to kill, so this rises as "
                   "carnivores pick weaker prey or bite harder.",
                   q(f"sum(rate({m('deaths_total', species='herbivore', cause='bitten')}[5m])) "
                     f"/ sum(rate({m('bites_total')}[5m]))", "kills per bite"),
                   colors={"kills": "red"}, unit="percentunit", min_=0, w=6),
        timeseries("Bites", "Carnivore bites that landed, per minute.",
                   q(per_minute(m("bites_total")), "bites / min"), colors={"bites": "red"},
                   unit="short", min_=0, w=6),
        timeseries("Group hunting", "Share of attempted bites that a herd around the prey made "
                   "miss (confusion), and share of landed bites on prey another carnivore bit "
                   "recently (pack bites, which do extra damage). Rising shares mean herds and "
                   "packs are forming.",
                   q(f"sum(rate({m('bites_confused_total')}[5m])) / (sum(rate("
                     f"{m('bites_confused_total')}[5m])) + sum(rate({m('bites_total')}[5m])))",
                     "confused"),
                   q(f"sum(rate({m('pack_bites_total')}[5m])) / sum(rate({m('bites_total')}[5m]))",
                     "pack bites"),
                   colors={"confused": "green", "pack bites": "red"}, unit="percentunit",
                   min_=0, w=6),
        timeseries("Herd density", "Mean kin_density brain input per species: 0 when no kin "
                   "is near, 1 in a dense group. Herds and packs push it up.",
                   *by_species("herd_density"), unit="percentunit", min_=0, w=6),
        timeseries("Carcasses", "Carcasses removed per minute: eaten to the bone by scavenging "
                   "carnivores, or rotted away untouched.",
                   q(per_minute(m("carcasses_removed_total")), "{{fate}}"),
                   colors={"eaten": "red", "rotted": "#808080"}, stack=True, unit="short",
                   min_=0, w=6),
        timeseries("Plant food", "Plant food removed per minute: eaten by herbivores, or rotted "
                   "uneaten. A lot of rot means food is not what limits the herbivores.",
                   q(per_minute(m("food_removed_total")), "{{fate}}"),
                   colors={"eaten": "green", "rotted": "#808080"}, stack=True, unit="short",
                   min_=0, w=6),
    ),
    row(
        "Genes",
        *[timeseries(f"{s.title()} gene drift", f"Population mean of each {s} gene divided by "
                     "the species default, so 1 is the starting genome. 1.5 means the gene "
                     "evolved 50% above its default. color is a neutral gene and is left out.",
                     q(m("gene_drift", 'gene!="color"', species=s), "{{gene}}"),
                     colors={}, h=9)
          for s in SPECIES_COLORS],
    ),
    row(
        "Best creature genes",
        *[timeseries(f"Best {s} genes", f"Raw gene values of the living {s} with the highest "
                     "score. It changes whenever another creature takes the lead.",
                     q(m("best_gene", species=s), "{{gene}}"), colors={})
          for s in SPECIES_COLORS],
        collapsed=True,
    ),
    row(
        "Lifestyle",
        timeseries("Condition", "Mean health, stamina and fat of each species, each as a share "
                   "of its maximum. Fat is stored from food eaten at full health.",
                   *[q(m(f"{k}_ratio", species=s), f"{s} {k}")
                     for s in SPECIES_COLORS for k in ("health", "stamina", "fat")],
                   unit="percentunit", min_=0, w=6),
        timeseries("Activity", "Share of each species that is swimming in a lake, standing "
                   "still to eat, or winded (out of stamina and slowed down). A rising swim "
                   "share in herbivores can mean they use lakes as a refuge.",
                   *[q(m(f"{k}_share", species=s), f"{s} {k.replace('in_', '')}")
                     for s in SPECIES_COLORS for k in ("in_water", "eating", "winded")],
                   unit="percentunit", min_=0, w=6),
        timeseries("Meals abandoned", "Meals broken off per minute by the brain's stop_eating "
                   "output; the rest of the meal is lost. Herbivores that learn to escape a "
                   "carnivore mid-meal raise this.",
                   *[q(per_minute(m("meals_abandoned_total", species=s)), s)
                     for s in SPECIES_COLORS], unit="short", min_=0, w=6),
        timeseries("Signalling", "Mean signal output (0 to 1). Kin within SIGNAL_RADIUS hear "
                   "it, and it costs health, so a rising signal means it pays off: they "
                   "evolved to communicate.",
                   q(m("signal_mean"), "{{species}}"), min_=0, w=6),
    ),
    row(
        "Run health",
        timeseries("Ticks per second", "Simulation speed. It falls as populations grow, since "
                   "perception and brains run for every creature each tick.",
                   q(f"rate({m('ticks_total')}[1m])", "ticks / s"), colors={"ticks": "blue"},
                   min_=0, w=6),
        timeseries("Brain saves", "New Hall of Fame brains written to saved_brains/ per 10 "
                   "minutes, at most one per AUTOSAVE_INTERVAL ticks.",
                   q(f"increase({m('brain_saves_total')}[10m])", "{{species}}"),
                   bars=True, unit="short", min_=0, w=6),
        timeseries("CPU", "CPU used by the simulation process, in cores.",
                   q(f"rate(process_cpu_seconds_total{{{JOB}}}[1m])", "cores"),
                   colors={"cores": "blue"}, min_=0, w=4),
        timeseries("Memory", "Resident memory of the simulation process.",
                   q(f"process_resident_memory_bytes{{{JOB}}}", "resident"),
                   colors={"resident": "blue"}, unit="bytes", min_=0, w=4),
        stat("Brain topology", "Neurons per layer of every creature's brain, from the inputs "
             "(perception) to the outputs (rotation, speed, signal, memory).",
             q(m("brain_layer_neurons"), "layer {{layer}}"), w=4, h=8),
    ),
]


def layout(rows):
    """Flatten rows into Grafana panels with ids and positions, filling 24-wide lines."""
    panels, y, next_id = [], 0, 1
    for r in rows:
        header = {"type": "row", "title": r["title"], "collapsed": r["collapsed"], "id": next_id,
                  "gridPos": {"h": 1, "w": 24, "x": 0, "y": y}, "panels": []}
        next_id += 1
        y += 1
        x, line_height, placed = 0, 0, []
        for p in r["panels"]:
            w, h = p["gridPos"]["w"], p["gridPos"]["h"]
            if x + w > 24:
                x, y, line_height = 0, y + line_height, 0
            placed.append({**p, "id": next_id, "gridPos": {"w": w, "h": h, "x": x, "y": y}})
            next_id += 1
            x += w
            line_height = max(line_height, h)
        panels.append(header)
        if r["collapsed"]:   # its panels show only when expanded, and take no space before
            header["panels"] = placed
            y = header["gridPos"]["y"] + 1
        else:
            panels.extend(placed)
            y += line_height
    return panels


def dashboard():
    return {
        "title": "Evolution simulation",
        "description": "Headless evolution simulation: progress, deaths, diversity, "
                       "predator-prey dynamics, genes and run health. Hover the (i) of a panel "
                       "for what it shows.",
        "uid": DASHBOARD_UID,
        "editable": True,
        "graphTooltip": 1,
        "refresh": "5s",
        "schemaVersion": 41,
        "tags": ["evolution"],
        "time": {"from": "now-3h", "to": "now"},
        "timepicker": {"refresh_intervals": ["5s", "10s", "30s", "1m", "5m", "15m", "1h"]},
        "timezone": "browser",
        "weekStart": "monday",
        "templating": {"list": []},
        "annotations": {"list": [
            {"builtIn": 1, "datasource": {"type": "grafana", "uid": "-- Grafana --"},
             "enable": True, "hide": True, "iconColor": "rgba(0, 211, 255, 1)",
             "name": "Annotations & Alerts", "type": "dashboard"},
            {"name": "Extinctions", "datasource": DATASOURCE, "enable": True,
             "iconColor": "orange", "expr": f"increase({m('extinctions_total')}[30s]) > 0",
             "step": "30s", "titleFormat": "{{species}} extinct",
             "textFormat": "Repopulated from the Hall of Fame"},
        ]},
        "panels": layout(ROWS),
    }


if __name__ == "__main__":
    OUT.write_text(json.dumps(dashboard(), indent=2) + "\n")
    print(f"wrote {OUT}")
