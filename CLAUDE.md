# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
poetry install                                   # deps (Python >=3.12)
make run ui                                      # game.py (ARGS="--new" passes flags)
make run watch                                   # game.py --watch: read-only, follows a running headless sim
make run headless                                # podman compose stack, opens Grafana; make logs / make stop
make clean-data                                  # stop the stack and delete the Prometheus + Grafana volumes
                                                 # both refuse to start while any simulation (local or container) runs

poetry run python game.py                        # graphical (requires display); continues from best_brain_*.json
poetry run python game.py --new                  # random brains; current best brains move to history
poetry run python game.py --list-brains          # list history snapshots (also game_headless.py)
poetry run python game.py --watch                # read-only UI: repopulates when best_brain_*.json change (no port 8000, no saves/snapshots)
poetry run python game.py --load 2               # start from snapshot #2 (1 = newest) or a folder name
poetry run python game_headless.py               # headless, auto-loads saved brains, Prometheus on :8000
poetry run python pretrain.py                    # behavior-cloning pretrain -> history snapshot *-pretrained (--install: best_brain_*.json)

poetry run pytest                                # all tests except the opt-in golden trace
make golden-record / make golden                 # golden trace: record before a refactor, check after
poetry run pytest tests/test_genetics.py         # one file
poetry run pytest tests/world/test_spatial.py::test_insert_and_query   # one test

poetry run ruff check .                          # lint (E, F, I, UP; line-length 100)
poetry run python -m cProfile -s time <script.py>
```

**Containers (Podman, not Docker):** `podman compose up --build -d` runs `game_headless.py` plus Prometheus and Grafana (dashboard at http://localhost:3000). The simulation runs with `requirements-headless.txt` (no pygame) and exposes metrics on port 8000. It mounts `./saved_brains`, so it starts from and autosaves into the host's `best_brain_*.json`. See `README.Podman.md`.

## Architecture

The shared library lives in `src/evolution/`, and the `game*.py` / `pretrain.py` entry scripts are thin wrappers around it. `config.py` is the single source of truth for every simulation constant. The packages are layered, and an import may only point down this chain:

```
ui/           pygame only: GameApp, ViewState, key commands, HUD, renderer, island WorkerPool
app/          startup shared by every entry script (bootstrap, cli), HeadlessRunner, metrics
persistence/  brain JSON files, history snapshots, Hall of Fame autosave, island exchange
sim/          Simulation (one tick = ordered phases), records, stats, breeding, populations
world/        World and its entities, perception, terrain, spatial grids, geometry
brain/        NeuralNetwork, BatchBrain, input/output Layout, imitation, brain-panel analysis
species.py    Species (HERBIVORE, CARNIVORE); genetics.py; config.py
```

Nothing outside `ui/` imports pygame. `tests/` mirrors the package layout.

**Bit-exact golden trace (opt-in).** `tests/test_golden.py` (marker `golden`, skipped by plain `pytest`) replays seeded runs (evolution, imitation and rendering) and compares them with `tests/golden/trace.json`, exactly. The trace is local and git-ignored: don't run or re-record it for feature work or rule changes. For a behaviour-preserving refactor, `make golden-record` before and `make golden` after; only the adapter `tests/golden/scenarios.py` may change when APIs move.
- It breaks on any change to the order of `random` / `np.random` calls, to the creature iteration order (`ALL_SPECIES`: herbivores first), to float operand order, or to the global id counters.

### Species

`species.py` turns everything that differs between the species into data. A `Species` (compared by identity) holds:
- `default_genes`, `gene_bounds`, `start_count`, `max_health`, `health_per_food`, `eat_ticks`, `swim_stamina_cost`;
- the naming helpers `brain_file`, `title` and `plural`;
- a `diet` strategy: `world/diet.py`'s `GrazingDiet` (food, digesting) or `PredatorDiet(prey=HERBIVORE)` (bites, scavenging).

Code loops over `ALL_SPECIES` or keys dicts by `Species` instead of branching on a boolean. `Creature.is_carnivore` is a read-only shortcut for `species.is_predator`.

### Simulation loop

`World` (`world/world.py`) holds:
- the populations (`{Species: [Creature]}`, plus `herbivores` / `carnivores` properties);
- food, carcasses, obstacles, lakes and food patches;
- the clock (`tick`, `light`, `season`);
- a `SpatialIndex` (grids for food, carcasses and each species), rebuilt by `reindex()`.

`Simulation` (`sim/simulation.py`) wraps a World together with the `HallOfFame`, the `Statistics`, `generation` and `extinctions`. `step()` advances one tick in phases, and their order is part of the rules:
1. clock
2. food
3. carcasses
4. reindex
5. perceive / think / act
6. separate (solid bodies)
7. retire the dead
8. extinct species recovery
9. elite reinjection
10. fade / sample
11. mark best
12. breed

The simulation does no I/O itself. It notifies `SimulationListener`s (`sim/events.py`: `record_set`, `deaths_processed`, `extinction`). The app layer subscribes:
- `persistence.autosave.HallOfFameAutosaver` writes brain files;
- `app.console.ExtinctionReporter` prints extinctions.

`app/bootstrap.start_simulation(source, geometry=..., listeners=...)` is the one startup path: it loads populations, builds the world and restores saved progress.

**Perception.** Perception for all creatures is one vectorized numpy pass, `world.perception.perceive_all` (via `World.perceive()`). It returns the (N, `BRAIN_INPUTS`) input matrix and sets `c.whiskers`, `c.target_whiskers`, `c.nearest_target_dist`, `c.nearest_smell_strength` and `c.effective_sight`. `tests/world/test_perception.py` holds the original scalar implementation as a reference, and the vectorized code must keep matching it.

**Brains.** NN inference for all creatures goes through `brain.batch.BatchBrain.forward()`, which stacks the per-creature weights and runs one vectorized `np.einsum` per layer. The Simulation owns one instance. The stacks are cached until the population changes, which assumes a creature's brain never changes in place after it is created. Any change to `NeuralNetwork`'s weight/bias layout has to stay compatible with it.

World mechanics, all tuned through `config.py`:
- **Walls:** the world edges are walls, not a wrap. Crossing one costs `HEALTH_LOST_ON_HIT` and pushes the creature back, like an obstacle. Obstacle whiskers see the edges, and food patches bounce off them.
- **Trees:** `world.worldgen.generate_obstacles` places `NUM_OBSTACLES` rocks and up to `NUM_TREES` `Tree`s (crowns inside the walls, off rocks, spaced apart). A `Tree` is an `Obstacle` whose `rect` is only the trunk (`TREE_TRUNK_FRACTION` of `TREE_CROWN_RADIUS_RANGE`), so collision and whiskers treat it like a small rock. The crown blocks sight but not movement: `perceive_all` hides vision-cone targets and target-whisker hits behind or under a crown, unless observer and target are under the same crown. Smell, kin, signals and obstacle whiskers ignore crowns. The renderer bakes the shadow and trunk into the background and draws the crown over creatures (`TREE_CROWN_ALPHA`), with creatures under it redrawn faintly on top.
- **Lakes:** `generate_lakes` places `NUM_LAKES` lakes inside the walls and clear of obstacles. Each lake is 1..`LAKE_LOBES` overlapping `Lake` blobs, and `world.lakes` holds the flat list of blobs. A blob's shore radius is its mean radius (`LAKE_RADIUS_RANGE`) plus random angular harmonics (`random_shore`, `LAKE_SHORE_ROUGHNESS`, `LAKE_SHORE_HARMONICS`). Creatures swim through them: speed is multiplied by `WATER_SPEED_FACTOR`, and instead of regenerating, stamina drains by the species' `swim_stamina_cost` per tick (`c.in_water`). Food (initial and respawned) avoids lakes and obstacles (`World.spawn_food`). The renderer bakes lakes into the background.
- **Bodies:** live creatures and carcasses are solid circles of radius `size × BODY_RADIUS_FACTOR` (below 1, sprites may overlap a little; 0 turns collisions off). After every creature has acted, `world.collision.separate` pushes overlapping creatures apart (`COLLISION_PASSES` relaxation passes over the overlapping pairs, split by mass `size²`) and out of carcasses, which never move. A push into a rock is undone, with no health cost. Food is not solid. Bites and scavenging count bodies within `CONTACT_REACH` as touching, because a separated pair sits exactly at the sum of the radii. Creatures feel touch (`perception._touch`): `INPUTS.touch_kin` / `touch_other` and their angles describe the body of the own / the other species pressing hardest. Strength is 0 from `TOUCH_TOLERANCE` apart, 0.5 at contact, and 1 at `TOUCH_TOLERANCE` deep.
  - **Impulses:** approaching bodies first exchange an impulse (mass `size²`, `RESTITUTION`; carcasses have infinite mass). It is added to the creature's knockback velocity `c.knock`, which moves it each tick on top of the brain's movement and decays by `KNOCK_FRICTION`. A rock or wall cancels the knock. A velocity change above `IMPACT_DAMAGE_THRESHOLD` costs `IMPACT_DAMAGE` per px/tick and sets `last_impact_tick` (death cause `impact`).
- **Chases:** carnivores bite instead of killing on contact. A bite needs contact, the prey within `CARNIVORE_BITE_ANGLE` of the carnivore's heading, and `BITE_COOLDOWN_TICKS` since the last bite (`PredatorDiet._bite`). Bitten prey gets an adrenaline stamina boost. A bite, or retaliation that lands, sets the victim's `hurt` input to 1, fading to 0 over `HURT_FADE_TICKS`. Herbivores bigger than the carnivore deal retaliation damage, split between the carnivores touching the prey. Death is simply `health <= 0`.
  - **Herds:** a bite on prey with other herbivores within `CONFUSION_RADIUS` may miss (`CONFUSION_MAX_MISS`, scaled by herd size / `CONFUSION_HERD_NORM`); the cooldown is still spent (`world.tally.bites_confused`).
  - **Packs:** prey keeps `recent_biters`. A bite on prey another carnivore bit within `PACK_WINDOW_TICKS` does `PACK_BITE_BONUS` damage (`tally.pack_bites`), and a kill gives those earlier biters `PACK_ASSIST_SCORE`.
- **Carcasses:** every dead creature leaves a `Carcass` in `world.carcasses`. Carnivores scavenge it (`PredatorDiet._scavenge`) until it's eaten or rots.
- **Meals:** eating makes a creature stand still for the species' `eat_ticks` (`c.eating_ticks`). The brain keeps running and sees the share of the meal left (`INPUTS.eating`). A `stop_eating` output above `MEAL_BREAK_THRESHOLD` abandons the meal, losing the rest, and the creature moves that tick (counted in `world.tally.meals_abandoned`). Herbivores absorb a food item's health gradually (`Creature.digest`); carnivores keep eating from `meal_carcass` until the meal ends, the carcass is gone, or health is full. A bite interrupts a herbivore's meal and forfeits the rest. Setting a duration to 0 restores instant eating.
- **Weight:** food eaten at full health is stored as `c.fat` (`Creature.absorb`, `FAT_STORE_EFFICIENCY`, capped at `FAT_MAX`). Fat grows the body (`c.size`, `FAT_SIZE_FACTOR`) and lowers speed (`FAT_SPEED_PENALTY`). It burns slowly (`FAT_BURN_PER_TICK`) and pays the starvation penalty before health does. Carnivore meals continue past full health until the carnivore is stuffed (`Creature.is_stuffed`).
- **Food patches:** `world.food_patches` (`FoodPatch`) drift around the map, and most respawned food lands near one.
- **Day/night:** `day_light(tick)` goes into `world.light`. At night vision cones and target whiskers shrink (`NIGHT_SIGHT_FACTOR`), while obstacle whiskers and smell don't.
- **Upkeep:** the base per-tick health loss is `Creature.upkeep()`. It is `HEALTH_LOSS_PER_TICK` times a weighted sum (`UPKEEP_WEIGHTS`, summing to 1) of trait/species-default ratios:
  - the lean body, Kleiber-scaled (`KLEIBER_EXPONENT`);
  - sight distance × angle;
  - stamina;
  - lifespan;
  - attractiveness.

  A creature with default genes pays exactly `HEALTH_LOSS_PER_TICK`, and every other trait has a cost.
- **Life history:** past its `lifespan` gene, a creature loses an extra `Creature.senescence()` per tick. A pair has a litter of `stochastic_round(mean litter_size)` young, which split `LITTER_ENDOWMENT` of max health between them.
- **Speciation:** only pairs within `SPECIATION_DISTANCE` (`genetics.genetic_distance`, the RMS of gene differences scaled by the species' `gene_bounds`) breed or seek each other (`genetics.compatible`). The neutral `color` gene hue-shifts the sprite (`COLOR_BUCKETS`, `COLOR_HUE_RANGE`).
- **Seasons:** `season(tick)` runs over `SEASON_LENGTH_TICKS` and goes into `world.season` (-1 is midwinter, 1 is midsummer). It scales the food cap (`food_cap`, `SEASON_FOOD_AMPLITUDE`) and is a brain input.
- **Lineage:** a child inherits `lineage` (the founder id) from its first parent and gets `parent_ids` and `descent`. At each score sample, `Statistics.sample` records each species' mean genes in `sim.stats[species].gene_history` and its number of live lineages in `.lineages`.
- **Signal and memory:** after the brain runs, the extra outputs are stored as `c.signal` (heard by kin within `SIGNAL_RADIUS`, with a small health cost) and `c.memory` (fed back as inputs next tick).

**Evolution is continuous.** Creatures die and are retired by `Simulation.retire`, which enters them into the statistics and the `HallOfFame` (`sim/records.py`).
- Eligible nearby adults reproduce (`sim/breeding.py`: brain crossover + gene averaging + mutation, in `genetics.py`).
  - Crossover is neuron-wise: each neuron's incoming weights and bias come from one parent.
  - Mates are ranked by attractiveness (`mate_preference`).
- Two optional selection boosts are off by default (both 0), because an A/B run showed no gain:
  - `MATE_SCORE_WEIGHT` adds score to mate ranking;
  - `ELITE_REINJECT_INTERVAL` periodically replaces the weakest `ELITE_REINJECT_COUNT` creatures of each species with the Hall of Fame brain and mutated copies (`Simulation.replace_weakest`, `populations.immigrants`).
- The reproduction cooldown is counted in ticks (`REPRODUCTION_COOLDOWN_TICKS`), so behavior doesn't depend on simulation speed.
- `HallOfFame[species]` is a `SpeciesRecords`:
  - `best` (a `Record`);
  - `elites` (the `ELITE_POOL_SIZE` best retired creatures);
  - `saved` (the score last written to disk).
- Scores fade with a `HOF_SCORE_HALF_LIFE` (`HallOfFame.fade`), so an old lucky record can't block newer brains forever.
- After an extinction (`sim.extinctions`), `populations.repopulate` refills the species: young of random elite pairs (the best record if the pool is empty), plus `REPOPULATE_RANDOM_SHARE` random brains.
- `sim.generation` is the deepest `Creature.generation` born (parents' max + 1). Creatures loaded from a brain file take the file's generation.

### Island workers (game.py)

Pressing **B** (background mode) in `game.py` starts `ui.workers.WorkerPool`, which spawns `cpu_count()-1` subprocesses of `game_headless.py --worker-id N`. Each worker runs an independent world, skips the Prometheus server, and logs to `saved_brains/worker_N.log`. The brain exchange lives in `persistence/islands.py` (`IslandExchange`):
- Workers save their Hall of Fame to `saved_brains/worker_N_{species}.json`.
- Every `WORKER_SYNC_INTERVAL` main ticks, `WorkerPool.sync` imports any worker brain that beats the main Hall of Fame (`IslandExchange.import_if_better`). The brain and `ISLAND_IMMIGRANTS - 1` mutated copies replace the weakest creatures of that species, so the visible population evolves from it. The main autosave then persists it to `best_brain_*.json`.
- The main sim publishes its Hall of Fame (the best across all islands) whenever it changes to `worker_main_{species}.json` (`IslandExchange.publish`), and every worker imports that in turn.
- Starting background mode clears the old `worker_*.json` files.
- Brain files are written atomically (`brain_store.write_json_atomic`: temp file plus `os.replace`), so a reader never sees half a file.
- Stopping (B again, closing the window, Ctrl+C or a crash) SIGTERMs the workers. They save unsaved records on SIGTERM, then the main sim does a final sync and saves its Hall of Fame. A worker also exits by itself if its parent `game.py` dies.
- Only the main process writes `best_brain_*.json` while workers are running.

Hall of Fame autosaves (`HallOfFameAutosaver`) happen at most once per `AUTOSAVE_INTERVAL` ticks. A record set inside that window is kept in `unsaved` and written by a later save.

### Geometry injection

Entities take their vector and rect types from a `world.geometry.Geometry(vec2, rect)` pair (`geometry=` kwarg, and `World.geometry`). The default `HEADLESS` pair is `Vector2`/`SimpleRect`, which mirrors the subset of the pygame API that's used (`distance_to`, `clipline`, `colliderect`, …). `ui/app.py` passes `PYGAME_GEOMETRY` (`pygame.math.Vector2` / `pygame.Rect`) for speed. Code inside the library must create vectors via an entity's `geometry` (children and respawned food inherit it) and never import pygame.

### Rendering (game.py only)

`ui/render/world_renderer.WorldRenderer` draws the world, and `ui/hud` draws panels, charts, the minimap and the brain panel. The entity classes have no draw methods.
- **Sprites** are drawn procedurally once (`ui/render/*_art.py`, 4× supersampled, then smoothscaled), pre-rotated into `config.SPRITE_ANGLE_STEPS` buckets with the drop shadow baked in, and cached in `SpriteCache` by `(is_carnivore, size, health level, walk frame, angle)`. A frame is mostly `screen.blits`.
- **Obstacles** are static and baked into the background surface at startup. If obstacles ever move or change, the background must be rebuilt.
- **Also drawn:**
  - carcass sprites;
  - a red glow on creatures hit within the last few ticks;
  - pulsing rings around creatures that are signalling;
  - a night overlay scaled by `world.light`, with food glowing through it.
- **Zoom** (`ViewState.zoom`, mouse wheel): `WorldRenderer.draw` scales positions and sprite sizes by the zoom, so sprites are built and cached at their on-screen size and stay sharp. The size is part of the cache key, so zoom levels that round to the same size share sprites.
  - When zoomed, the background is a scaled copy of the view plus a `BG_CACHE_MARGIN`. It is rebuilt only when the camera leaves it or the zoom changes.
  - Avoid any per-frame full-screen `smoothscale`: it roughly doubles the frame time.
  - Screen↔world conversions go through `ViewState.screen_to_world` / `view_size`, and the zoom math (`zoom_at`, `min_zoom`) lives in `ui/camera.py`.

`ui/view_state.ViewState` holds everything the player controls that isn't simulation state: camera, zoom, selection, toggles, speed and brain history. `ui/commands.KEY_COMMANDS` is the whole keyboard map.

### Neural network

Topology is `config.BRAIN_TOPOLOGY = [BRAIN_INPUTS, 16, 8, BRAIN_OUTPUTS]` = `[41, 16, 8, 6]`, tanh on all layers. `NeuralNetwork` (`brain/network.py`) is pure math; `to_data` / `from_data` convert it for `persistence.brain_store`.

`brain/layout.py` names every column: `INPUTS.target_whiskers`, `OUTPUTS.speed` and so on (an int for scalars, a slice for groups), plus `labels` for the brain panel.
- `perceive_all`, `imitation`, the simulation and the brain panel all index through it.
- To add an input, write it in `perceive_all`, append a field to `INPUTS`, and update the `BRAIN_INPUTS` formula. A module-level assert checks the widths.
- New inputs and outputs go at the end: `NeuralNetwork.from_data` copies the overlapping top-left block and zeroes the rest, so saved brains keep their existing wiring and behave as before until mutation connects the new columns.

Outputs: `[rotation, speed, signal, memory..., stop_eating]`. `sim.breeding.mate_seeking_drive` may override the first two for adults.

### Genes

Genes:
- traits: `size`, `max_speed`, `sight_distance`, `sight_angle`, `max_stamina`, `attractiveness`;
- life history (shared by both species, `_LIFE_HISTORY_GENES`): `mutation_rate` and `mutation_amount` (the offspring's brain mutation, applied via `genetics.mutate_brain`), `lifespan`, `litter_size` and `color`.

Details:
- All genes are clamped by the species' `gene_bounds` (`genetics.clamp_genes`). The size ranges don't overlap (`HERBIVORE_GENE_MIN_MAX`, `CARNIVORE_GENE_MIN_MAX`), so herbivores are always the bigger species (defaults 16 vs 8). Saved brains' genes are clamped on load.
- Genes mutate multiplicatively (±`GENE_MUTATION_AMOUNT`), except those listed in `ADDITIVE_GENE_MUTATION` (`color`).
- `Creature` fills in missing genes from the species defaults, so older saves load.
- Larger size reduces speed, and stamina adds to effective size.
- Never call `brain.mutate(MUTATION_RATE, ...)` directly for offspring or copies. Use `mutate_brain(brain, genes)` so the heritable rate applies.

### Pretraining (brain/imitation.py)

`imitation.teacher_actions` is a hand-written policy over perception rows (columns via `INPUTS`): seek the target, follow carcasses and smell (carnivores), flee strong predator smell or a predator's touch (herbivores), step away from kin pressing against a wanderer, and steer off near obstacles. `train_brain` fits a brain to it with numpy backprop (Adam, MSE). `pretrain.py` collects rows DAgger-style (round 1 uses random brains, later rounds the current student) and runs its worlds without listeners, so nothing autosaves.

### Saved brains

`persistence/brain_store.save_brain` / `load_brain` write JSON to `saved_brains/` (weights as nested lists plus topology, generation, genes, score). `NeuralNetwork.from_data` handles topology mismatches, so old brains survive architecture changes. The main files are `best_brain_{herbivore,carnivore}.json` (`Species.brain_file`), and they are tracked in git.

When they are written:
- Autosaves and exit saves (`autosave.save_on_exit`) overwrite them only with a Hall of Fame record set during this run (`Record.fresh`; a faded record may score below the file).
- The manual **S** key in `game.py` always overwrites.

`game.py` and `game_headless.py` start from them (`bootstrap.start_simulation`) unless given `--new` or `--load`. **L** reloads them mid-run (`bootstrap.reload_population`).

Each file also stores that species' chart `history` (score and longevity, capped at `CHART_HISTORY_SAVE_MAX`):
- After a load, `bootstrap.restore_progress` restores the history and seeds the Hall of Fame with the saved score, so charts continue instead of restarting.
- Without a new record, exit still updates the history (`update_saved_history`).

**Brain history** (`persistence/brain_history.py`):
- Every start of `game.py`/`game_headless.py` (not island workers) snapshots the current best files as a pair into `saved_brains/history/<YYYYmmdd-HHMMSS>[-label]/` (git-ignored), after the starting populations were read.
- A snapshot is skipped when the brains equal the newest one. Chart history and uptime are ignored in that comparison, because every exit rewrites them.
- `--new` and `--load` move the best files out instead of copying, so the run's own records save from scratch.
- Only the newest `BRAIN_HISTORY_KEEP` snapshots are kept, plus the one loaded this run.

## game.py controls

Space pause · +/- simulation speed (1/8–32 ticks per frame; below 1 a tick runs every few frames) · S save brains · L load brains · Enter toggle FPS cap · [ / ] shrink/grow the score charts · D toggle drawing · G toggle the gene-drift chart (species mean genes ÷ defaults, for the selected creature's species) · N toggle the brain panel of the selected creature (node activations, strongest weight×activation connections, top input drivers, output history) · B background mode + island workers · mouse wheel zoom (`ZOOM_STEP` per notch, up to `ZOOM_MAX`, around the cursor) · left-click: spawn food / select creature (the camera follows it; when it dies, it jumps to the best of that species; a minimap click stops following) · minimap drag to pan.

## Metrics

Prometheus metrics are served at `http://localhost:8000` by `app/metrics.GameMetrics`, a collector that computes them from the watched `Simulation` when Prometheus scrapes, so the loop pays nothing per tick. Event counts come from `world.tally` (`WorldTally`: bites, food and carcass fates), `SpeciesStats` (births, deaths by `Creature.death_cause`, extinctions) and the autosaver (brain saves).

`grafana_dashboard.json` is generated: edit `monitoring/dashboard.py` (panels, layout, descriptions) and run `poetry run python monitoring/dashboard.py`. Its queries use the metric and label names from `GameMetrics`. `podman compose up` also starts Prometheus (`:9090`, config in `monitoring/prometheus.yml`) and Grafana (`:3000`, no login, the dashboard as the home page, provisioned from `monitoring/grafana/`). These couple to the dashboard in two ways:
- its panels reference the data source uid `fev2gbax7os8we`;
- its queries filter on `job="evolution"`.

Renaming either breaks the panels.
