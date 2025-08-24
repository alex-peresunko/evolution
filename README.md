# Evolution Simulation

A neural network-driven evolution simulator where herbivores and carnivores compete, evolve, and adapt in a dynamic environment.

## Features

- **Two Species Ecosystem**:
  - Herbivores (green) that eat plant food
  - Carnivores (red) that hunt herbivores
  - Realistic predator-prey dynamics

- **Neural Network AI**:
  - Each creature has its own neural network brain
  - Brains evolve and improve over generations
  - Inputs include vision, smell, health, and stamina
  - Outputs control movement and rotation

- **Advanced Genetics**:
  - Inherited traits: size, speed, sight distance, stamina, etc.
  - Mutation system for genetic diversity
  - Sexual reproduction between compatible creatures
  - Hall of Fame system preserves best specimens

- **Realistic Features**:
  - Stamina system affects speed and exhaustion
  - Life stages (juvenile to adult growth)
  - Food decay and nutrition system
  - Spatial optimization for large populations
  - Collision detection with obstacles

- **Performance Features**:
  - Background mode for maximum simulation speed
  - FPS limiting toggle
  - Drawing toggle for performance
  - Prometheus metrics integration

## Installation

1. Clone the repository:
```bash
git clone https://github.com/alex-peresunko/evolution.git
cd evolution
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
.\venv\Scripts\activate  # Windows
source venv/bin/activate # Linux/Mac
```

3. Install dependencies:
```bash
pip install pygame numpy prometheus_client
```

4. Run the simulation:
```bash
python game.py
```

## Controls

- **Space**: Pause/Resume simulation
- **S**: Save best brains to file
- **L**: Load saved brains from file
- **Enter**: Toggle FPS limit (60 FPS vs unlimited)
- **D**: Toggle drawing (visual updates)
- **B**: Toggle background mode (max speed)
- **Left Click**: 
  - On empty space: Create new food
  - On creature: Select/deselect creature
- **Ctrl+C**: Quit (when running in terminal)

## Display Elements

- **Top Left**: Statistics (generation, population counts, scores)
- **Top Right**: Selected creature details
- **Bottom Right**: Score history charts
- **Main View**:
  - Green circles: Herbivores
  - Red triangles: Carnivores
  - Green dots: Food
  - Gray rectangles: Obstacles
  - Cyan flashing: Newly born creatures
  - Yellow outline: Best performer of species
  - Whiskers: Visible on selected or best creatures

## Metrics

The simulation exposes Prometheus metrics on port 8000. Metrics include:
- Population counts
- Generation numbers
- Best scores
- Genetic traits
- Neural network topology
- Performance metrics

Access metrics at: http://localhost:8000

## Headless Mode

For running without graphics (faster evolution):
```bash
python game_headless.py
```

## Files

- `game.py`: Main simulation with graphics
- `game_headless.py`: Console-only version for faster evolution
- `saved_brains/`: Directory containing saved neural networks
- Best brains are automatically saved as:
  - `best_brain_herbivore.json`
  - `best_brain_carnivore.json`
- Autosaves occur every 100 generations

## Notes

- The simulation uses wrapped boundaries (creatures/food loop around edges)
- Population caps prevent exponential growth
- Extinct species can recover from Hall of Fame specimens
- Performance varies based on population size and environment complexity
