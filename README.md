# Evolution Simulation

A predator/prey neuroevolution simulator: herbivores and carnivores, each driven by its own
neural network, live, eat, hunt, breed and die in a world with lakes, trees, day and night and
seasons. There are no generations to wait for: evolution is continuous, and the best brains
are saved and carried over between runs.

## Features

- **Two species:** herbivores graze food that drifts around the map in patches; carnivores
  chase, bite and scavenge carcasses. Bites take time and aim, prey gets an adrenaline boost,
  and big herbivores fight back. Herds confuse attackers (bites miss more often), while packs
  bite harder and share the credit for a kill.
- **Neural brains:** every creature has a `[41, 16, 8, 6]` tanh network. It sees through
  whiskers, a vision cone and smell, hears kin signals, feels bodies touching it (kin or the
  other species, and from which side), and feels its health, stamina, fat, water, the season,
  a bite (`hurt`) and how much of its meal is left. It steers, sets its
  speed, signals, keeps two memory values and can break off a meal to escape.
- **Genetics:** size, speed, sight, stamina, attractiveness, lifespan, litter size, colour and
  even the brain's own mutation rate are inherited and mutate. Only genetically close pairs
  breed (speciation), and every trait costs upkeep.
- **Life history:** young grow into adults, eating takes time, surplus food becomes fat (slower
  but a reserve), old age costs health, and every dead creature leaves a carcass.
- **World:** solid bodies (creatures collide with momentum: the lighter one bounces off, hard
  impacts hurt, and nobody walks through carcasses),
  walls at the edges, rocks, trees whose crowns hide what's under them, lakes to
  swim through (slow, and tiring), a day/night cycle that shortens sight, and seasons that
  change how much food grows.
- **Records:** a Hall of Fame keeps each species' best brains (scores fade over time), refills
  extinct species from them, and autosaves them to `saved_brains/`. Every start snapshots the
  previous best brains into `saved_brains/history/`.
- **Speed:** vectorised perception and batched brain inference, island workers in background
  mode, and a headless container with Prometheus and Grafana.

## Installation

Needs Python 3.12+ and [Poetry](https://python-poetry.org):

```bash
git clone https://github.com/alex-peresunko/evolution.git
cd evolution
poetry install
```

## Running

```bash
make run ui                  # the graphical simulation (ARGS="--new" passes flags)
make run headless            # headless simulation + Prometheus + Grafana in Podman
make run watch               # read-only UI following the running headless simulation
make logs / make stop        # follow / save and stop the headless simulation
make clean-data              # stop it and delete the Prometheus and Grafana data
```

`make run ui` and `make run headless` refuse to start while another simulation is running,
since both write the best brain files. The same entry points directly:

```bash
poetry run python game.py                # continues from saved_brains/best_brain_*.json
poetry run python game.py --new          # random brains (the current best move to history)
poetry run python game.py --list-brains  # list history snapshots
poetry run python game.py --load 2       # start from snapshot #2 (1 = newest) or a folder name
poetry run python game.py --watch        # read-only: reloads when the brain files change
poetry run python game_headless.py       # no graphics; metrics on http://localhost:8000
poetry run python pretrain.py            # pretrain brains by imitating a hand-written policy
                                         # (--install also makes them the best brains)
```

The container setup is described in `README.Podman.md`.

## Controls

| Key / mouse | Action |
|---|---|
| Space | Pause / resume |
| + / - | Simulation speed: 1/8 to 32 ticks per frame (slow motion is interpolated) |
| S | Save the best living creatures' brains |
| L | Reload the best brains from file |
| Enter | Toggle the 60 FPS cap |
| D | Toggle drawing |
| [ / ] | Shrink / grow the score charts |
| G | Gene-drift chart for the selected creature's species |
| N | Brain panel of the selected creature |
| B | Background mode: island worker processes evolve in parallel |
| Mouse wheel | Zoom around the cursor |
| Left click | On a creature: select it and follow it; elsewhere: drop food |
| Minimap click / drag | Pan the camera (stops following) |

Watch mode only allows the keys that don't write `saved_brains/`.

## Display

- **Top left:** minimap, and the statistics below it.
- **Top right:** the selected creature's details; with N, its brain panel (activations, the
  strongest connections, top input drivers and output history).
- **Bottom right:** Hall of Fame score and longevity charts per species; with G, the gene-drift
  chart beside them.
- **In the world:**
  - creature sprites, hue-shifted by the colour gene, with a glow: red when just hit, yellow
    for the species' best, cyan for the newborn;
  - pulsing rings around signalling creatures;
  - whiskers and the sight cone on the selected creature, whiskers on each species' best;
  - a bubble around carnivores still immune after a respawn;
  - food, carcasses, rocks, trees, lakes, and a night overlay.

## Metrics

`make run headless` starts the headless simulation with Prometheus and a ready Grafana
dashboard (http://localhost:3000; see `README.Podman.md`). Its rows, each panel described
under its (i):
- **Overview:** uptime, generation, speed, populations, extinctions, last brain save
- **Evolution progress:** best / mean / median scores, Hall of Fame vs saved brain, lifespan,
  generations per hour
- **Life and death:** deaths by cause (bitten, starved, old age, ...), births vs deaths,
  extinctions
- **Diversity:** live lineages, genetic diversity against the speciation distance
- **Predators and prey:** populations with food, season and daylight, hunt success, bites,
  group hunting (confused and pack bites), herd density, carcass and food fate
- **Genes:** population-mean gene drift; the best creature's genes (collapsed)
- **Lifestyle:** health, stamina and fat; swimming, eating and winded shares; meals abandoned;
  signalling
- **Run health:** ticks per second, brain saves, CPU, memory, brain topology

The raw metrics are at http://localhost:8000.

## Development

```bash
make test        # poetry run pytest
make golden-record && make golden   # bit-exact check around a refactor: record before, check after
make lint        # poetry run ruff check .
```

Every simulation constant lives in `src/evolution/config.py`. `tests/test_golden.py` replays
seeded runs bit-exactly against a local trace; it is opt-in and only meant for refactors.

## Files

- `game.py`: the graphical simulation
- `game_headless.py`: no graphics, for faster evolution (also the island worker)
- `pretrain.py`: pretrains both species' brains by imitating a hand-written policy
- `src/evolution/`: the simulation library, in layers that only import downward:
  - `ui/`: pygame app, controls, HUD and renderer
  - `app/`: shared startup, command line, headless runner, Prometheus metrics
  - `persistence/`: brain files, history snapshots, autosave, island brain exchange
  - `sim/`: the simulation step, Hall of Fame, statistics, breeding
  - `world/`: world state, creatures, perception, terrain
  - `brain/`: neural network, batched inference, input/output layout, imitation
  - `species.py`, `genetics.py`, `config.py`
- `monitoring/`: Prometheus config, Grafana provisioning, and `dashboard.py`, which generates
  `grafana_dashboard.json`
- `Dockerfile`, `compose.yaml`, `requirements-headless.txt`: the headless container
- `saved_brains/`: `best_brain_herbivore.json` and `best_brain_carnivore.json` (with their
  chart history), and the `history/` snapshots. They autosave when a new Hall of Fame record
  is set, at most once per `AUTOSAVE_INTERVAL` ticks, and on exit.
