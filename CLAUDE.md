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
# Dockerfile uses game_headless.py (no display required) and requirements-headless.txt (no pygame)
```

## Architecture

### Package layout

```
src/evolution/         # shared library
  config.py            # all simulation constants (single source of truth)
  math_types.py        # headless Vector2 / SimpleRect (pygame-free)
  neural_network.py    # NeuralNetwork class
  genetics.py          # combine_brains, combine_genes, mutate_genes
  spatial.py           # SpatialHashGrid
  entities.py          # Creature, Food, Obstacle (with vec2/rect factory injection)
  simulation.py        # update_world, create_initial_population, build_sim_state
  metrics.py           # GameMetrics (Prometheus)
  brain_store.py       # save_brain / load_brain
```

Entry scripts are thin wrappers (~60–150 lines each):

| File | Mode |
|------|------|
| `game.py` | Graphical via pygame; passes `pygame.math.Vector2` factory to entities |
| `game_headless.py` | Console-only; uses headless `Vector2`/`SimpleRect` (the default) |
| `game_headless_multiproc.py` | Console + ProcessPoolExecutor; keeps local `LeanCreature`/`LeanFood` for IPC |

### Vector type injection

`Creature`/`Food`/`Obstacle` accept `vec2_factory=` / `rect_factory=` kwargs. Default: headless `Vector2`/`SimpleRect`. `game.py` passes `pygame.math.Vector2` / `pygame.Rect`. Child creatures inherit the factory from parent 1 during reproduction inside `simulation.py`.

### Core classes

- **`NeuralNetwork`** — Feed-forward NN, topology `[18, 16, 8, 2]`, tanh activations. `forward()`, `mutate()`, `copy()`, `save()`/`load()` (JSON). Load is topology-resilient.
- **`Creature`** — Single agent with `NeuralNetwork` brain and `genes` dict. `see()` builds 18-input vector; `think()` runs the NN; `act()` applies movement, eating, collision.
- **`Food`** — Passive resource with freshness/lifetime decay.
- **`Obstacle`** — Static rectangle; collision via `clipline`/`colliderect`.
- **`SpatialHashGrid`** — Bucketed spatial lookup, O(1) neighbor queries.
- **`GameMetrics`** — Prometheus `Counter`/`Gauge` wrappers; served on port 8000.

### Neural network inputs (18 values)

```
whisker[0..2]          obstacle proximity
target_angle, dist     nearest food (herbivore) or prey (carnivore)
pred_angle, dist       nearest predator / smell direction
smell_angle, strength  omnidirectional scent
prey_vel_x, prey_vel_y prey velocity vector (carnivores)
health_ratio           current health normalized
sin(angle), cos(angle) heading
size, speed, sight_dist, sight_angle, stamina, hunger
```

Outputs: `[rotation_delta, speed]` — both tanh-clamped.

### Evolution

Continuous sexual reproduction: eligible nearby creatures reproduce (brain crossover + gene averaging + mutation). **Hall of Fame** preserves all-time best individuals per species and repopulates after extinction.

### Genetic traits

`size`, `max_speed`, `sight_distance`, `sight_angle`, `max_stamina`, `attractiveness` — all bounded by `GENE_MIN_MAX`. Larger size reduces speed; stamina adds to effective size.

### Saved brains

Best brains auto-save as JSON to `saved_brains/`. Headless mode loads them on startup. The JSON format stores weights as nested lists with topology metadata, enabling cross-architecture loading.

### Tests

```bash
poetry run pytest tests/ -v   # 24 tests, no pygame required
```

### Legacy / prototype files

Early prototypes moved to `archive/` — not used by the main simulation.

## Metrics

Prometheus metrics are exposed at `http://localhost:8000`. A ready-to-import Grafana dashboard is in `grafana_dashboard.json`.
