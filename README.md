# MAX-SC-QBF Genetic Algorithm Solver

This project implements a Genetic Algorithm-based metaheuristic for solving the MAX-QBF problem with Set Cover constraints (MAX-SC-QBF).  you can change the condigs to your preferences and all the data from each instance will be sabe in the results folder


## Configuration

Edit `src/config.py` to customize the Genetic Algorithm parameters:

```python
GENERATIONS = 10000        # Number of generations to run
POP_SIZE = 100             # Population size
MUTATION_RATE = 0.1        # Mutation probability
TIMEOUT_MINUTES = 30       # Maximum runtime per instance

# Instance files to process
INSTANCE_FILES = [
    "instances/qbf/qbf020",
    "instances/qbf/qbf040", 
    "instances/qbf/qbf060",
]
```

## Rquirements:
- Python3.11

## How to Run:

Create a python environment

```bash
python3.11 -m venv venv
```

Activate you python environment 
```bash
# On Linux/Mac:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

Install dependencies
```bash
pip install -r requirements.txt
```

Run the main script
```bash
python3.11 src/main.py
```

## Results

All instance data and computational results are automatically saved in the `results/` for later analysis.



