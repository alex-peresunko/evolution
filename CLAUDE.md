# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
poetry install                                   # deps (Python >=3.12)

poetry run python game.py                        # graphical (requires display)
poetry run python game_headless.py               # headless, auto-loads saved brains, Prometheus on :8000
poetry run python game_headless_multiproc.py     # headless + ProcessPoolExecutor

poetry run pytest                                # all tests (no pygame required)
poetry run pytest tests/test_genetics.py         # one file
poetry run pytest tests/test_spatial.py::test_insert_and_query   # one test

poetry run ruff check .                          # lint (E, F, I, UP; line-length 100)
poetry run python -m cProfile -s time <script.py>
```

**Docker:** `docker compose up` — runs `game_headless.py` with `requirements-headless.txt` (no pygame), exposes metrics on port 8000. See `README.Docker.md`.

## Architecture

Shared library lives in `src/evolution/`; the three `game*.py` entry scripts are thin wrappers around it. `config.py` is the single source of truth for every simulation constant.

### Simulation loop

`simulation.build_sim_state()` returns a plain dict (`sim_state`) holding all world state: creature/food lists, four `SpatialHashGrid`s (general, food, herbivore, carnivore), score histories, tick/generation counters, and `hall_of_fame`. `simulation.update_world(sim_state, ...)` advances one tick and is shared by `game.py` and `game_headless.py`. NN inference for all creatures goes through `batch_forward()`, which stacks the per-creature weights and runs one vectorized `np.einsum` per layer. Any change to `NeuralNetwork`'s weight/bias layout has to stay compatible with it.

Evolution is continuous: creatures die and are "retired" (`_retire_creature` updates the Hall of Fame and saves the brain), and eligible nearby adults reproduce (brain crossover + gene averaging + mutation, in `genetics.py`). After an extinction, the species is repopulated from the Hall of Fame.

**`game_headless_multiproc.py` diverges:** it has its own `update_world` and a *generational* `evolve_population`, and it ships `LeanCreature`/`LeanFood` dataclasses to worker processes for IPC. Changes to `simulation.update_world` do not carry over to it automatically.

### Island workers (game.py)

Pressing **B** (background mode) in `game.py` spawns `cpu_count()-1` subprocesses of `game_headless.py --worker-id N`. Each worker runs an independent world, skips the Prometheus server, logs to `saved_brains/worker_N.log`, and saves its best brains as `saved_brains/worker_N_{herbivore,carnivore}.json`. `_sync_worker_brains` polls those files by mtime and promotes a worker's brain into the main Hall of Fame when its score is higher.

### Vector type injection

`Creature`/`Food`/`Obstacle` accept `vec2_factory=` / `rect_factory=` kwargs. The default is the headless `Vector2`/`SimpleRect` from `math_types.py`, which mirrors the subset of the pygame API that's used (`distance_to`, `clipline`, `colliderect`, …). `game.py` passes `pygame.math.Vector2` / `pygame.Rect` instead. Code inside the library must create vectors via `self._vec2` (or inherit the factory from an existing entity, as children and respawned food do) and never import pygame.

### Rendering (game.py only)

`render.Renderer` draws everything; the entity classes have no draw methods. Sprites are drawn procedurally once (4× supersampled, then smoothscaled), pre-rotated into `config.SPRITE_ANGLE_STEPS` buckets with the drop shadow baked in, and cached by `(is_carnivore, size, health level, walk frame, angle)`, so a frame is mostly `screen.blits`. Obstacles are static and baked into the background surface at startup; if obstacles ever move or change, the background must be rebuilt.

### Neural network

Topology is `config.BRAIN_TOPOLOGY = [18, 16, 8, 2]`, tanh on all layers. The input size is derived from `NUM_WHISKERS`, so changing the inputs in `Creature.see()` means updating that formula too. Inputs, in order:

```
whisker[0..2]                obstacle proximity (1 = touching)
target_angle, target_dist    carnivore: nearest visible prey
                             herbivore: food angle blended away from nearest visible predator, food dist
smell_angle, smell_strength  omnidirectional: prey (carnivore) / predator (herbivore)
prey_vel_x, prey_vel_y       carnivore only, else 0
health_ratio, sin(angle), cos(angle)
size, speed, sight_dist, sight_angle, stamina, hunger   (normalized)
```

Outputs: `[rotation_delta, speed]`. `_apply_mate_seeking_drive` may override them for adults.

### Genes

`size`, `max_speed`, `sight_distance`, `sight_angle`, `max_stamina`, `attractiveness`, all clamped by `config.GENE_MIN_MAX`. Larger size reduces speed, and stamina adds to effective size.

### Saved brains

`brain_store.save_brain`/`load_brain` write JSON to `saved_brains/` (weights as nested lists plus topology, generation, genes, score). `NeuralNetwork.load` handles topology mismatches, so old brains survive architecture changes. Main files: `best_brain_{herbivore,carnivore}.json`. These are tracked in git. Autosaves and exit saves write them through `save_brain_if_better`, which only overwrites a file when the new score is higher. The manual **S** key in `game.py` always overwrites. `game_headless.py` loads them on startup, while `game.py` starts with a random population and loads them only when **L** is pressed.

## game.py controls

Space pause · S save brains · L load brains · Enter toggle FPS cap · D toggle drawing · B background mode + island workers · left-click: spawn food / select creature · minimap drag to pan.

## Metrics

Prometheus metrics are served at `http://localhost:8000` (`metrics.GameMetrics`). An importable Grafana dashboard is in `grafana_dashboard.json`.

## Legacy

`archive/` contains early prototypes that the simulation does not use.
