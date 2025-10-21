from solver.GA import GA
from multiprocessing import Pool

def get_instances():
    return [
        ({}, 1000, 100, 0.1),
        ({}, 1000, 200, 0.1),
        ({}, 1000, 100, 0.2),
        ({"mutation": "ADAPTATIVE"}, 1000, 100, 0.1),
        ({"initialization": "latin"}, 1000, 100, 0.1),
        ({"parent_selection": "SUS"}, 1000, 100, 0.1),
        ({"crossover": "UNIFORM"}, 1000, 100, 0.1),
        ({"population_selection": "READY_STATE"}, 1000, 100, 0.1),
        ({"mutation": "ADAPTATIVE",
          "initialization": "latin",
          "parent_selection": "SUS",
          "crossover": "UNIFORM",
          "population_selection": "READY_STATE"}, 1000, 100, 0.1)
    ]

def get_files():
    return [
        "instances/scqbf/exact_n25.txt",
        "instances/scqbf/exact_n50.txt",
        "instances/scqbf/exact_n100.txt",
        "instances/scqbf/exact_n200.txt",
        "instances/scqbf/exact_n400.txt",
        "instances/scqbf/exp_n25.txt",
        "instances/scqbf/exp_n50.txt",
        "instances/scqbf/exp_n100.txt",
        "instances/scqbf/exp_n200.txt",
        "instances/scqbf/exp_n400.txt",
        "instances/scqbf/normal_n25.txt",
        "instances/scqbf/normal_n50.txt",
        "instances/scqbf/normal_n100.txt",
        "instances/scqbf/normal_n200.txt",
        "instances/scqbf/normal_n400.txt",
    ]

def run_instance(args):
    config, generations, pop_size, mutation_rate, filename, config_ind = args
    print(f"[START] Config {config_ind} - File: {filename}")

    instance = GA(config, generations, pop_size, mutation_rate, filename)
    instance.solve(config_ind=config_ind)

    print(f"[DONE] Config {config_ind} - File: {filename}")
    return f"Finished config {config_ind} on {filename}"

def main():
    files = get_files()
    input = get_instances()
    config_ind = 1

    tasks = []
    for config_ind, (config, generations, pop_size, mutation_rate) in enumerate(input, start=1):
        for filename in files:
            tasks.append((config, generations, pop_size, mutation_rate, filename, config_ind))

    print(f"Submitting {len(tasks)} tasks to the pool...")

    with Pool() as pool:
        results = pool.map(run_instance, tasks)


if __name__ == "__main__":
    main() 