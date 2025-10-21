# MAX-SC-QBF Genetic Algorithm Solver

This project implements a Genetic Algorithm-based metaheuristic for solving the MAX-QBF problem with Set Cover constraints (MAX-SC-QBF).  you can change the condigs to your preferences and all the data from each instance will be saved in the results folder.

## Configuration

Add your instance files on the get_files() function in main.py
Add configurations on the get_intances() function in main.py

a configuration is a tupple with the following:
    - a dict with the specification of any extra strategy (ommiting a strategy results in the default one being applied, see the example in main.py)
    - the maximum number of generations
    - the population size
    - the mutation rate

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