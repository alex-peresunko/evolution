# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
poetry install                                   # deps (Python >=3.12)

poetry run python game.py                        # graphical (requires display); continues from best_brain_*.json
poetry run python game.py --new                  # random brains; current best brains move to history
poetry run python game.py --list-brains          # list history snapshots (also game_headless.py)
poetry run python game.py --load 2               # start from snapshot #2 (1 = newest) or a folder name
poetry run python game_headless.py               # headless, auto-loads saved brains, Prometheus on :8000
poetry run python game_headless_multiproc.py     # headless + ProcessPoolExecutor
poetry run python pretrain.py                    # behavior-cloning pretrain -> history snapshot *-pretrained (--install: best_brain_*.json)

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

`simulation.build_sim_state()` returns a plain dict (`sim_state`) holding all world state: creature/food lists, four `SpatialHashGrid`s (general, food, herbivore, carnivore), score histories, tick/generation counters, and `hall_of_fame`. `simulation.update_world(sim_state, ...)` advances one tick and is shared by `game.py` and `game_headless.py`. Perception for all creatures is one vectorized numpy pass, `perception.perceive_all`, which returns the (N, `BRAIN_INPUTS`) input matrix and sets `c.whiskers`, `c.target_whiskers`, `c._nearest_target_dist`, `c._nearest_smell_strength` and `c.effective_sight`. `Creature.see()` is a one-creature wrapper around it. `tests/test_perception.py` holds the original scalar implementation as a reference, and the vectorized code must keep matching it. NN inference for all creatures goes through `batch_forward()`, which stacks the per-creature weights and runs one vectorized `np.einsum` per layer. The stacks are cached until the population changes, which assumes a creature's brain never changes in place after it is created. Any change to `NeuralNetwork`'s weight/bias layout has to stay compatible with it.

World mechanics, all tuned through `config.py`:
- **Walls:** the world edges are walls, not a wrap. Crossing one costs `HEALTH_LOST_ON_HIT` and pushes the creature back, like an obstacle. Obstacle whiskers see the edges, and food patches bounce off them.
- **Lakes:** `generate_lakes` places `NUM_LAKES` lakes inside the walls and clear of obstacles. Each lake is 1..`LAKE_LOBES` overlapping `Lake` blobs, and `sim_state["lakes"]` holds the flat list of blobs. A blob's shore radius is its mean radius (`LAKE_RADIUS_RANGE`) plus random angular harmonics (`random_shore`, `LAKE_SHORE_ROUGHNESS`, `LAKE_SHORE_HARMONICS`). Creatures swim through them: speed is multiplied by `WATER_SPEED_FACTOR`, and instead of regenerating, stamina drains by `HERBIVORE_SWIM_STAMINA_COST` / `CARNIVORE_SWIM_STAMINA_COST` per tick (`c.in_water`). Respawned food avoids lakes. The renderer bakes lakes into the background.
- **Chases:** carnivores bite instead of killing on contact. A bite needs contact, the prey within `CARNIVORE_BITE_ANGLE` of the carnivore's heading, and `BITE_COOLDOWN_TICKS` since the last bite (`Creature._try_bite`). Bitten prey gets an adrenaline stamina boost. Herbivores bigger than the carnivore deal retaliation damage. Death is simply `health <= 0`.
- **Carcasses:** every dead creature leaves a `Carcass` in `sim_state["carcasses"]` (with its own grid). Carnivores scavenge it (`_scavenge`) until it's eaten or rots.
- **Meals:** eating makes a creature stand still for `HERBIVORE_EAT_TICKS` / `CARNIVORE_EAT_TICKS` (`eating_ticks`; brain outputs are ignored meanwhile). Herbivores absorb a food item's health gradually (`_digest`); carnivores keep eating from `meal_carcass` until the meal ends, the carcass is gone, or health is full. A bite interrupts a herbivore's meal and forfeits the rest. Setting a duration to 0 restores instant eating.
- **Weight:** food eaten at full health is stored as `c.fat` (`Creature._absorb`, `FAT_STORE_EFFICIENCY`, capped at `FAT_MAX`). Fat grows the body (`c.size`, `FAT_SIZE_FACTOR`) and lowers speed (`FAT_SPEED_PENALTY`). It burns slowly (`FAT_BURN_PER_TICK`) and pays the starvation penalty before health does. Carnivore meals continue past full health until the carnivore is stuffed (`_is_stuffed`).
- **Food patches:** `sim_state["food_patches"]` drift around the map, and most respawned food lands near one.
- **Day/night:** `day_light(tick)` goes into `sim_state["light"]`. At night vision cones and target whiskers shrink (`NIGHT_SIGHT_FACTOR`), while obstacle whiskers and smell don't.
- **Upkeep:** the base per-tick health loss is `Creature.upkeep()`. It is `HEALTH_LOSS_PER_TICK` times a weighted sum (`UPKEEP_WEIGHTS`, summing to 1) of trait/species-default ratios:
  - the lean body, Kleiber-scaled (`KLEIBER_EXPONENT`);
  - sight distance × angle;
  - stamina;
  - lifespan;
  - attractiveness.

  A creature with default genes pays exactly `HEALTH_LOSS_PER_TICK`, and every other trait has a cost.
- **Life history:** past its `lifespan` gene, a creature loses an extra `Creature.senescence()` per tick. A pair has a litter of `stochastic_round(mean litter_size)` young, which split `LITTER_ENDOWMENT` of max health between them.
- **Speciation:** only pairs within `SPECIATION_DISTANCE` (`genetics.genetic_distance`, the RMS of gene differences scaled by `GENE_MIN_MAX` ranges) breed or seek each other. The neutral `color` gene hue-shifts the sprite (`COLOR_BUCKETS`, `COLOR_HUE_RANGE`).
- **Seasons:** `season(tick)` runs over `SEASON_LENGTH_TICKS` and goes into `sim_state["season"]` (-1 is midwinter, 1 is midsummer). It scales the food cap (`food_cap`, `SEASON_FOOD_AMPLITUDE`) and is a brain input.
- **Lineage:** a child inherits `lineage` (the founder id) from its first parent and gets `parent_ids` and `descent`. At each score sample, `sample_genes` records the species' mean genes in `sim_state["gene_history"]` and the number of live lineages in `sim_state["lineages"]`.
- **Signal and memory:** after `batch_forward`, the extra outputs are stored as `c.signal` (heard by kin within `SIGNAL_RADIUS`, with a small health cost) and `c.memory` (fed back as inputs next tick).

Evolution is continuous: creatures die and are "retired" (`_retire_creature` updates the Hall of Fame and saves the brain), and eligible nearby adults reproduce (brain crossover + gene averaging + mutation, in `genetics.py`). Crossover is neuron-wise: each neuron's incoming weights and bias come from one parent. Mates are ranked by attractiveness. Two optional selection boosts are off by default (both 0), because an A/B run showed no gain: `MATE_SCORE_WEIGHT` adds score to mate ranking, and `ELITE_REINJECT_INTERVAL` periodically replaces the weakest `ELITE_REINJECT_COUNT` creatures of each species with the Hall of Fame brain and mutated copies (`islands.inject_immigrants`). The reproduction cooldown is counted in ticks (`REPRODUCTION_COOLDOWN_TICKS`), so behavior doesn't depend on simulation speed. After an extinction, the species is repopulated from the Hall of Fame.

**`game_headless_multiproc.py` diverges:** it has its own `update_world` and a *generational* `evolve_population`, and it ships `LeanCreature`/`LeanFood` dataclasses to worker processes for IPC. Changes to `simulation.update_world` do not carry over to it automatically. It only stores signal/memory: it keeps instant kills and has no carcasses, food patches, lakes, night, seasons, litters or speciation. Upkeep, senescence and the heritable mutation rate do reach it, because they live in `Creature`/`genetics`.

### Island workers (game.py)

Pressing **B** (background mode) in `game.py` spawns `cpu_count()-1` subprocesses of `game_headless.py --worker-id N`. Each worker runs an independent world, skips the Prometheus server, and logs to `saved_brains/worker_N.log`. The brain exchange lives in `islands.py`:
- Workers save their Hall of Fame to `saved_brains/worker_N_{species}.json`.
- Every `WORKER_SYNC_INTERVAL` main ticks, `_sync_worker_brains` imports any worker brain that beats the main Hall of Fame (`import_if_better`). The brain and `ISLAND_IMMIGRANTS - 1` mutated copies replace the weakest creatures of that species, so the visible population evolves from it. The main autosave then persists it to `best_brain_*.json`.
- The main sim publishes its Hall of Fame (the best across all islands) to `worker_main_{species}.json`, and every worker imports that in turn.
- Starting background mode clears the old `worker_*.json` files.
- Brain files are written atomically (temp file plus `os.replace`), so a reader never sees half a file.
- Stopping (B again, closing the window, Ctrl+C or a crash) SIGTERMs the workers. They save unsaved records on SIGTERM, then the main sim does a final sync and saves its Hall of Fame. A worker also exits by itself if its parent `game.py` dies.
- Only the main process writes `best_brain_*.json` while workers are running.

Hall of Fame autosaves happen at most once per `AUTOSAVE_INTERVAL` ticks. A record set inside that window is kept in `hof_unsaved` and written later (`autosave_hall_of_fame`).

### Vector type injection

`Creature`/`Food`/`Obstacle` accept `vec2_factory=` / `rect_factory=` kwargs. The default is the headless `Vector2`/`SimpleRect` from `math_types.py`, which mirrors the subset of the pygame API that's used (`distance_to`, `clipline`, `colliderect`, …). `game.py` passes `pygame.math.Vector2` / `pygame.Rect` instead. Code inside the library must create vectors via `self._vec2` (or inherit the factory from an existing entity, as children and respawned food do) and never import pygame.

### Rendering (game.py only)

`render.Renderer` draws everything; the entity classes have no draw methods. Sprites are drawn procedurally once (4× supersampled, then smoothscaled), pre-rotated into `config.SPRITE_ANGLE_STEPS` buckets with the drop shadow baked in, and cached by `(is_carnivore, size, health level, walk frame, angle)`, so a frame is mostly `screen.blits`. Obstacles are static and baked into the background surface at startup; if obstacles ever move or change, the background must be rebuilt. The renderer also draws:
- carcass sprites;
- a red glow on creatures hit within the last few ticks;
- pulsing rings around creatures that are signalling;
- a night overlay scaled by `sim_state["light"]`, with food glowing through it.

Zoom (`sim_state["zoom"]`, mouse wheel): `draw_world` scales positions and sprite sizes by the zoom, so sprites are built and cached at their on-screen size and stay sharp. The size is part of the cache key, so zoom levels that round to the same size share sprites. When zoomed, the background is a scaled copy of the view plus a `BG_CACHE_MARGIN`, and it is rebuilt only when the camera leaves it or the zoom changes. Avoid any per-frame full-screen `smoothscale`: it roughly doubles the frame time. Screen↔world conversions in `game.py` go through `_screen_to_world` / `_view_size`, and the zoom math (`zoom_at`, `min_zoom`) lives in `camera.py`.

### Neural network

Topology is `config.BRAIN_TOPOLOGY = [BRAIN_INPUTS, 16, 8, BRAIN_OUTPUTS]` = `[35, 16, 8, 5]`, tanh on all layers. Changing the inputs in `perception.perceive_all` means updating the `BRAIN_INPUTS` formula too. Append new inputs/outputs at the end: `NeuralNetwork.load` copies the overlapping top-left block, so saved brains keep their existing wiring. Inputs, in order:

```
whisker[0..2]                obstacle proximity (1 = touching)
target_angle, target_dist    carnivore: nearest visible prey
                             herbivore: food angle blended away from nearest visible predator, food dist
smell_angle, smell_strength  omnidirectional: prey (carnivore) / predator (herbivore)
prey_vel_x, prey_vel_y       carnivore only, else 0
health_ratio, sin(angle), cos(angle)
size, speed, sight_dist, sight_angle, stamina, hunger   (normalized)
kin_angle, kin_dist, kin_density   nearest same-species creature within KIN_SENSE_RADIUS
signal_angle, signal_strength      loudest same-species signal within SIGNAL_RADIUS
carcass_angle, carcass_dist        carnivore only: nearest visible carcass, else (0, 1)
light                              0 = midnight, 1 = noon
memory[0..NUM_MEMORY-1]            own memory outputs from the previous tick
target_whisker[0..2]               food (herbivore) / prey or carcass (carnivore) within
                                   WHISKER_TARGET_WIDTH of each whisker ray, up to the night-
                                   shrunk sight, occluded by obstacles (1 = touching, 0 = none)
season                             -1 = midwinter, 1 = midsummer
carcass_smell_angle, carcass_smell_strength   carnivore only: nearest carcass within
                                   SMELL_DISTANCE, omnidirectional, else 0
in_water                           1 = standing in a lake
```

Outputs: `[rotation_delta, speed, signal, memory...]`. `_apply_mate_seeking_drive` may override them for adults.

### Genes

Genes:
- traits: `size`, `max_speed`, `sight_distance`, `sight_angle`, `max_stamina`, `attractiveness`;
- life history (shared by both species, `_LIFE_HISTORY_GENES`): `mutation_rate` and `mutation_amount` (the offspring's brain mutation, applied via `genetics.mutate_brain`), `lifespan`, `litter_size` and `color`.

Details:
- All genes are clamped by `config.GENE_MIN_MAX`.
- Genes mutate multiplicatively (±`GENE_MUTATION_AMOUNT`), except those listed in `ADDITIVE_GENE_MUTATION` (`color`).
- `Creature` fills in missing genes from the species defaults, so older saves load.
- Larger size reduces speed, and stamina adds to effective size.
- Never call `brain.mutate(MUTATION_RATE, ...)` directly for offspring or copies. Use `mutate_brain(brain, genes)` so the heritable rate applies.

### Pretraining (imitation.py)

`imitation.teacher_actions` is a hand-written policy over perception rows: seek the target, follow carcasses and smell (carnivores), flee strong predator smell (herbivores), and steer off near obstacles. Its column indices must match `perceive_all`'s layout. `train_brain` fits a brain to it with numpy backprop (Adam, MSE). `pretrain.py` collects rows DAgger-style (round 1 uses random brains, later rounds the current student) and disables Hall of Fame autosave in its worlds.

### Saved brains

`brain_store.save_brain`/`load_brain` write JSON to `saved_brains/` (weights as nested lists plus topology, generation, genes, score). `NeuralNetwork.load` handles topology mismatches, so old brains survive architecture changes. Main files: `best_brain_{herbivore,carnivore}.json`. These are tracked in git. Autosaves and exit saves write them through `save_brain_if_better`, which only overwrites a file when the new score is higher. The manual **S** key in `game.py` always overwrites. `game.py` and `game_headless.py` start from them (`create_start_populations`) unless given `--new` or `--load`; **L** reloads them mid-run. Each file also stores that species' chart `history` (score and longevity, capped at `CHART_HISTORY_SAVE_MAX`). After a load, `simulation.restore_progress` restores the history and seeds the Hall of Fame with the saved score, so charts continue instead of restarting. When the brain isn't better, headless exit still updates the history (`update_saved_history`). `game_headless_multiproc.py` doesn't write history.

**Brain history** (`brain_history.py`): every start of `game.py`/`game_headless.py` (not island workers) snapshots the current best files as a pair into `saved_brains/history/<YYYYmmdd-HHMMSS>[-label]/` (git-ignored), after the starting populations were read. A snapshot is skipped when the brains equal the newest one; chart history and uptime are ignored in that comparison because every exit rewrites them. `--new` and `--load` move the best files out instead of copying, so the run's own records save from scratch through `save_brain_if_better`. Only the newest `BRAIN_HISTORY_KEEP` snapshots are kept, plus the one loaded this run.

## game.py controls

Space pause · +/- simulation speed (1–32 ticks per frame) · S save brains · L load brains · Enter toggle FPS cap · [ / ] shrink/grow the score charts · D toggle drawing · G toggle the gene-drift chart (species mean genes ÷ defaults, for the selected creature's species) · N toggle the brain panel of the selected creature (node activations, strongest weight×activation connections, top input drivers, output history) · B background mode + island workers · mouse wheel zoom (`ZOOM_STEP` per notch, up to `ZOOM_MAX`, around the cursor) · left-click: spawn food / select creature (the camera follows it; when it dies, it jumps to the best of that species; a minimap click stops following) · minimap drag to pan.

## Metrics

Prometheus metrics are served at `http://localhost:8000` (`metrics.GameMetrics`). An importable Grafana dashboard is in `grafana_dashboard.json`.

## Legacy

`archive/` contains early prototypes that the simulation does not use.
