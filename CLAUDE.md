# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the Simulation

```bash
# Install dependencies
poetry install

# Graphical (requires display)
poetry run python game.py

# Headless (fastest, auto-loads saved brains on start)
poetry run python game_headless.py

# Headless + multi-core (uses ProcessPoolExecutor)
poetry run python game_headless_multiproc.py

# Profile any script
poetry run python -m cProfile -s time <script.py>
```

**Docker:**
```bash
docker compose up   # builds image, exposes Prometheus metrics on port 8000
```

> Note: The Dockerfile runs `game.py` by default, which needs a display. In containers, prefer `game_headless.py`.

## Architecture

### Three simulation variants

All three files share the same core classes and logic but differ in execution mode:

| File | Mode |
|------|------|
| `game.py` | Graphical via pygame |
| `game_headless.py` | Console-only; replaces pygame with custom `Vector2`/`SimpleRect` |
| `game_headless_multiproc.py` | Console + multiprocessing; uses `LeanCreature`/`LeanFood` frozen dataclasses for pickling |

### Core classes (defined per-file, not shared modules)

- **`NeuralNetwork`** — Feed-forward NN, topology `[16, 8, 2]`, tanh activations. `forward()`, `mutate()`, `crossover()`, `save()`/`load()` (JSON). Load is topology-resilient: weights copy into the overlap region if architecture changes.
- **`Creature`** — Single agent. Has a `NeuralNetwork` brain and a `genes` dict. `see()` builds the 16-input vector; `think()` runs the NN; `act()` applies movement, eating, collision. The `is_carnivore` flag distinguishes species.
- **`Food`** — Passive resource with freshness/lifetime decay.
- **`Obstacle`** — Static rectangle; collision via `clipline`/`colliderect`.
- **`SpatialHashGrid`** — Bucketed spatial lookup, O(1) neighbor queries; critical for populations of 80–160+ creatures.
- **`GameMetrics`** — Prometheus `Counter`/`Gauge` wrappers; updated each tick; served on port 8000.

### Neural network inputs (16 values)

```
whisker[0..2]          obstacle proximity
target_angle, dist     nearest food (herbivore) or prey (carnivore)
pred_angle, dist       nearest predator / smell direction
smell_angle, strength  omnidirectional scent
health_ratio           current health normalized
sin(angle), cos(angle) heading
size, speed, sight_dist, sight_angle, stamina
```

Outputs: `[rotation_delta, speed]` — both tanh-clamped.

### Evolution loop (800 ticks per generation)

Each tick: `see()` → `think()` → `act()`. Eligible nearby creatures reproduce sexually (brain crossover + gene averaging + mutation). End of generation: top 5% survive; new population bred from survivors. A **Hall of Fame** preserves all-time best individuals per species and repopulates after extinction.

### Genetic traits

`size`, `max_speed`, `sight_distance`, `sight_angle`, `max_stamina`, `attractiveness` — all bounded by `GENE_MIN_MAX`. Larger size reduces speed; stamina adds to effective size.

### Saved brains

Best brains auto-save as JSON to `saved_brains/` every 100 generations. Headless mode loads them on startup. The JSON format stores weights as nested lists with topology metadata, enabling cross-architecture loading.

### Prototype / legacy files

`main.py`, `sprite.py`, `sprite_demo.py`, `neural_network.py`, `neural_network_sample.py`, `test.py` are early prototypes not used by the main simulation.

## Metrics

Prometheus metrics are exposed at `http://localhost:8000`. A ready-to-import Grafana dashboard is in `grafana_dashboard.json`.
