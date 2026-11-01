from evolution import config
from evolution.app import bootstrap
from evolution.brain.network import NeuralNetwork
from evolution.persistence.brain_store import save_brain
from evolution.species import ALL_SPECIES, CARNIVORE, HERBIVORE

SMALL = (400, 300)


def test_no_source_starts_random_populations(capsys):
    populations, generation = bootstrap.start_populations(None, SMALL)
    assert generation == 1
    for species in ALL_SPECIES:
        assert len(populations[species]) == species.start_count
    assert "Creating a new random population" in capsys.readouterr().out


def test_saved_brain_seeds_the_hall_of_fame_and_charts(tmp_path):
    brain = NeuralNetwork(config.BRAIN_TOPOLOGY)
    save_brain(brain, HERBIVORE.brain_file, generation=6, genes=dict(HERBIVORE.default_genes),
               score=12.5, history={"score": [1.0, 2.0], "longevity": [3, 4]}, save_dir=tmp_path)
    sim = bootstrap.start_simulation(str(tmp_path), world_bounds=SMALL)
    assert sim.generation == 6
    records = sim.hall_of_fame[HERBIVORE]
    assert records.best.score == 12.5 and not records.best.fresh
    assert not sim.hall_of_fame[CARNIVORE].has_brain
    assert sim.stats[HERBIVORE].score_history == [1.0, 2.0]
    assert sim.stats[CARNIVORE].score_history == [0, 0]
    progenitor = sim.world.herbivores[0]
    assert all((a == b).all() for a, b in zip(progenitor.brain.weights, brain.weights))


def test_reload_without_a_file_changes_nothing(tmp_path):
    sim = bootstrap.start_simulation(None, world_bounds=SMALL)
    before = list(sim.world.carnivores)
    assert bootstrap.reload_population(sim, CARNIVORE, str(tmp_path)) is None
    assert sim.world.carnivores == before


def test_reload_from_parsed_data_ignores_the_file_on_disk(tmp_path):
    save_brain(NeuralNetwork(config.BRAIN_TOPOLOGY), HERBIVORE.brain_file, generation=3,
               score=1.0, save_dir=tmp_path)
    sim = bootstrap.start_simulation(None, world_bounds=SMALL)
    data = NeuralNetwork(config.BRAIN_TOPOLOGY).to_data()
    data.update(generation=9, score=7.0, history={"score": [5.0], "longevity": [1]})
    progenitor = bootstrap.reload_population(sim, HERBIVORE, str(tmp_path), data=data)
    assert progenitor.generation == 9
    assert sim.hall_of_fame[HERBIVORE].best.score == 7.0
    assert sim.stats[HERBIVORE].score_history == [5.0]


def _brain_data(generation, score=None, genes=None):
    data = NeuralNetwork(config.BRAIN_TOPOLOGY).to_data()
    data["generation"] = generation
    if score is not None:
        data["score"] = score
    if genes is not None:
        data["genes"] = genes
    return data


def test_adopt_keeps_the_creatures_and_swaps_in_exact_copies():
    sim = bootstrap.start_simulation(None, world_bounds=(400, 300))
    before = [(c.id, c.pos.x, c.pos.y) for c in sim.world.herbivores]
    genes = {**HERBIVORE.default_genes, "size": 20}
    data = _brain_data(77, score=12.5, genes=genes)
    bootstrap.adopt_population(sim, HERBIVORE, data)
    herbivores = sim.world.herbivores
    assert [(c.id, c.pos.x, c.pos.y) for c in herbivores] == before
    want = NeuralNetwork.from_data(data, config.BRAIN_TOPOLOGY)
    assert len({id(c.brain) for c in herbivores}) == len(herbivores)
    for c in herbivores:
        assert all((a == b).all() for a, b in zip(c.brain.weights, want.weights))
        assert c.genes["size"] == 20 and c.generation == 77
    assert sim.hall_of_fame[HERBIVORE].best.score == 12.5


def test_adopt_into_an_empty_species_starts_founders():
    sim = bootstrap.start_simulation(None, world_bounds=(400, 300))
    sim.world.populations[CARNIVORE] = []
    bootstrap.adopt_population(sim, CARNIVORE, _brain_data(5))
    assert len(sim.world.carnivores) == CARNIVORE.start_count
