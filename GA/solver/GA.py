import random
from .solution import *
from .evaluator import *
from .instance import *
import signal
import time
import pandas as pd
from typing import List
import os
import numpy as np


class TimeoutError(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutError("Execution time exceeded!")


class GA():
    rng = random.Random(0)
    verbose = True

    def __init__(self, config: dict, generations: int, pop_size: int, mutation_rate: float, filename: str):
        self.instance = read_max_sc_qbf_instance(filename)
        self.ObjFunction: ScQbfEvaluator = ScQbfEvaluator(self.instance)
        self.generations: int = generations
        self.pop_size: int = pop_size
        self.chromosome_size = self.instance.n
        self.mutation_rate = mutation_rate
        self.best_cost = float('-inf')
        self.best_sol = ScQbfSolution(n = self.instance.n)
        self.config = config
        # Initialize DataFrame to store generation data
        self.generation_data = pd.DataFrame(columns=['generation', 'best_cost', 'timestamp'])
        self.timed_out = False
        self.start_time = None
        self.filename = filename

    #Runtime control functions

    def _record_generation_data(self, generation: int, best_cost: float):
        new_row = pd.DataFrame({
            'generation': [generation],
            'best_cost': [best_cost],
            'timestamp': [time.time()]
        })
        self.generation_data = pd.concat([self.generation_data, new_row], ignore_index=True)

    def save_generation_data(self, filename: str = "generation_data.parquet"):
        try:
            instance_folder = "results/"
            os.makedirs(instance_folder, exist_ok=True)

            filepath = os.path.join(instance_folder, filename)
            self.generation_data.to_parquet(filepath, index=False)
            print(f"Generation data saved to {filename}")
        except Exception as e:
            print(f"Error saving to Parquet: {e}")
            csv_filename = filename.replace('.parquet', '.csv')
            self.generation_data.to_csv(csv_filename, index=False)
            print(f"Data saved to {csv_filename} instead")

    def check_timeout(self, timeout_minutes: int = 30):
        if self.start_time and (time.time() - self.start_time) > (timeout_minutes * 60):
            self.timed_out = True
            raise TimeoutError(f"Execution stopped: exceeded {timeout_minutes} minutes")

    #Auxiliary functions

    def get_best_chromosome(self, population: List[ScQbfSolution]) -> ScQbfSolution:
        best_fitness = float('-inf')
        best_chromosome = None
        for c in population:
            fit = self.ObjFunction.evaluate_objfun(c)
            if fit > best_fitness:
                best_fitness = fit
                best_chromosome = c
        return best_chromosome

    def get_worse_chromosome(self, population: List[ScQbfSolution]) -> ScQbfSolution:
        worse_fitness = float('inf')
        worse_chromosome = None
        for c in population:
            fit = self.ObjFunction.evaluate_objfun(c)
            if fit < worse_fitness:
                worse_fitness = fit
                worse_chromosome = c
        return worse_chromosome

    def _fix_solution_greedy(self, sol: ScQbfSolution) -> ScQbfSolution:
        """
        This function is called when the constructed solution is not feasible.
        It'll add the most covering elements until the solution is feasible.
        """
        while not self.ObjFunction.is_solution_valid(sol):
            cl = [
                i for i in range(0, self.instance.n) 
                if sol.elements[i] == 0
                and self.ObjFunction.evaluate_insertion_delta_coverage(i, sol) > 0
                ]
            best_cand = None
            best_coverage = -1
            
            for cand in cl:
                coverage = self.ObjFunction.evaluate_insertion_delta_coverage(cand, sol)
                if coverage > best_coverage:
                    best_coverage = coverage
                    best_cand = cand
            
            if best_cand is not None:
                sol.elements[best_cand] = 1
            else:
                break
        
        if not self.ObjFunction.is_solution_valid(sol):
            raise ValueError("Could not fix the solution to be feasible")
        
        return sol

    def _fix_solution_random(self, sol: ScQbfSolution) -> ScQbfSolution:
        """
        This function is called when the constructed solution is not feasible.
        It'll add the random elements that increase coverage until the solution is feasible.
        """
        while not self.ObjFunction.is_solution_valid(sol):
            cl = [
                i for i in range(0, self.instance.n) 
                if sol.elements[i] == 0
                and self.ObjFunction.evaluate_insertion_delta_coverage(i, sol) > 0
                ]

            sol.elements[self.rng.choice(cl)] = 1
        
        return sol

    #Population initialization functions

    def generate_random_chromosome(self) -> List[int]:
        """
        Sets a random array o variables, possibly creating an unfeasable solution
        """
        chromosome = []
        for _ in range(self.chromosome_size):
            chromosome.append(self.rng.randint(0, 1))
        return chromosome

    def default_initialization(self) -> List[ScQbfSolution]:
        """
        This function is the default population initialization, creating random feasible solutions. 
        """
        population = []
        while len(population) < self.pop_size:
            constructed_sol = ScQbfSolution(el=self.generate_random_chromosome())
            constructed_sol = self._fix_solution_random(constructed_sol)
            population.append(constructed_sol)

        return population

    def latin_hypercube(self) -> List[ScQbfSolution]:
        """
        This function uses the latin hyper cube for population initialization. 
        As it may create infeasable solutions, we fix them randomly.
        """
        genes = [
            self.rng.sample([i % 2 for i in range(self.pop_size)], self.pop_size)
            for _ in range(self.chromosome_size)
            ]
        solutions = population = [ScQbfSolution(el = [genes[i][e] for i in range(self.chromosome_size)])
                      for e in range(self.pop_size)
                    ]
        population = [self._fix_solution_random(sol)
                      for sol in solutions
                    ]
        return population
    
    def initialize_population(self) -> List[ScQbfSolution]:
        initialization = self.config.get("initialization", "default")
        if (initialization == "latin"):
            return self.latin_hypercube()
        else:
            return self.default_initialization()

    #Parent selection functions

    def default_parents_selection(self, population: List[ScQbfSolution]) -> List[ScQbfSolution]:
        parents = []
        while len(parents) < self.pop_size:
            index1 = self.rng.randint(0, self.pop_size - 1)
            index2 = self.rng.randint(0, self.pop_size - 1)
            parent1 = population[index1]
            parent2 = population[index2]
            if self.ObjFunction.evaluate_objfun(parent1) > self.ObjFunction.evaluate_objfun(parent2):
                parents.append(parent1)
            else:
                parents.append(parent2)
        return parents
    
    def sthocastic_uniform_selection(self, population: List[ScQbfSolution]) -> List[ScQbfSolution]:
        fitnesses = [self.ObjFunction.evaluate_objfun(i) for i in population]
        fitnesses = np.array(fitnesses)
        fitnesses = fitnesses 
        worst = self.ObjFunction.evaluate_objfun(self.get_worse_chromosome(population))
        if worst < 0:
            fitnesses = fitnesses - worst
        cumulative_fitness = []
        cumulative_sum = 0

        for f in fitnesses:
            cumulative_sum += f
            cumulative_fitness.append(cumulative_sum)
        
        step = np.sum(fitnesses)/self.pop_size
        current = self.rng.uniform(0, step)
        pointer = 0
        selected = []
        for _ in range(self.pop_size):
            while current > cumulative_fitness[pointer]:
                pointer += 1
            selected.append(population[pointer])
            current += step

        return selected

    def select_parents(self, population: List[ScQbfSolution]) -> List[ScQbfSolution]:
        parent_selection = self.config.get("parent_selection", "default")
        if (parent_selection == "SUS"):
            return self.sthocastic_uniform_selection(population)
        else:
            return self.default_parents_selection(population)

    #Crossover functions

    def fix_solution_from_parents(self, solution: ScQbfSolution, parent1: ScQbfSolution, parent2: ScQbfSolution) -> ScQbfSolution:
        """
        This function is called when the constructed solution is not feasible.
        It'll add the most covering elements until the solution is feasible.
        """
        while not self.ObjFunction.is_solution_valid(solution):
            cl = [
                i for i in range(0, self.instance.n) 
                if solution.elements[i] == 0
                and (parent1.elements[i] == 1 or parent2.elements[i] == 1)
                and self.ObjFunction.evaluate_insertion_delta_coverage(i, solution) > 0
                ]
            best_cand = None
            best_coverage = -1
            
            for cand in cl:
                coverage = self.ObjFunction.evaluate_insertion_delta_coverage(cand, solution)
                if coverage > best_coverage:
                    best_coverage = coverage
                    best_cand = cand
            
            if best_cand is not None:
                solution.elements[best_cand] = 1
            else:
                break
        
        if not self.ObjFunction.is_solution_valid(solution):
            raise ValueError("Could not fix the solution to be feasible")
        
        return solution


    def uniform_crossover(self, parents: List[ScQbfSolution]) -> List[ScQbfSolution]:
        offsprings = []
        
        for i in range(0, self.pop_size, 2):
            parent1 = parents[i]
            parent2 = parents[i + 1]
            
            mask = [self.rng.randint(0, 1) for _ in range(self.chromosome_size)]
            offspring1 = [parent1.elements[e] if mask[e] else parent2.elements[e] for e in range(self.chromosome_size)]
            offspring2 = [parent2.elements[e] if mask[e] else parent1.elements[e] for e in range(self.chromosome_size)]
            offspring1 = self.fix_solution_from_parents(ScQbfSolution(el=offspring1), parent1, parent2)
            offspring2 = self.fix_solution_from_parents(ScQbfSolution(el=offspring2), parent1, parent2)
            offsprings.append(offspring1)
            offsprings.append(offspring2)
        
        return offsprings

    def default_crossover(self, parents: List[ScQbfSolution]) -> List[ScQbfSolution]:
        offsprings = []
        
        for i in range(0, self.pop_size, 2):
            parent1 = parents[i]
            parent2 = parents[i + 1]
            
            crosspoint1 = self.rng.randint(0, self.chromosome_size)
            crosspoint2 = crosspoint1 + self.rng.randint(0, self.chromosome_size - crosspoint1)
            
            offspring1 = []
            offspring2 = []
            
            for j in range(self.chromosome_size):
                if crosspoint1 <= j < crosspoint2:
                    offspring1.append(parent2.elements[j])
                    offspring2.append(parent1.elements[j])
                else:
                    offspring1.append(parent1.elements[j])
                    offspring2.append(parent2.elements[j])

            offspring1 = self.fix_solution_from_parents(ScQbfSolution(el=offspring1), parent1, parent2)
            offspring2 = self.fix_solution_from_parents(ScQbfSolution(el=offspring2), parent1, parent2)

            offsprings.append(offspring1)
            offsprings.append(offspring2)
        
        return offsprings

    def crossover(self, parents: List[ScQbfSolution]) -> List[ScQbfSolution]:
        crossover = self.config.get("crossover", "default")
        if (crossover == "UNIFORM"):
            return self.uniform_crossover(parents)
        else:
            return self.default_crossover(parents)

    #Mutation functions
    def mutate_gene(self, chromosome: List[int], locus: int):
        chromosome[locus] = 1 - chromosome[locus]

    def default_mutation(self, offsprings: List[ScQbfSolution]) -> List[ScQbfSolution]:
        mutation_genes = [[l for l in gene.elements] for gene in offsprings]
        for c in mutation_genes:
            for locus in range(self.chromosome_size):
                if self.rng.random() < self.mutation_rate:
                    self.mutate_gene(c, locus)
        
        mutations = [self._fix_solution_random(ScQbfSolution(el=s)) for s in mutation_genes]

        return mutations

    def adaptative_mutation(self, offsprings: List[ScQbfSolution]) -> List[ScQbfSolution]:
        mutation_genes = [s.elements for s in offsprings]
        sums = np.sum(np.array(mutation_genes), axis=0)
        probabilities = []
        for s in sums:
            aux = ((s / (self.pop_size / 2)) - 1) / 4 #Max mutation rate is 0.5, min is 0 (when theres enough variation)
            if aux >= 0:
                probabilities.append(aux)
            else:
                probabilities.append(-aux)

        for c in mutation_genes:
            for locus in range(self.chromosome_size):
                if self.rng.random() < probabilities[locus]:
                    self.mutate_gene(c, locus)

        mutations = [self._fix_solution_random(ScQbfSolution(el =s)) for s in mutation_genes]


        return mutations

    def mutate(self, parents: List[ScQbfSolution]) -> List[ScQbfSolution]:
        mutation = self.config.get("mutation", "default")
        if (mutation == "ADAPTATIVE"):
            return self.adaptative_mutation(parents)
        else:
            return self.default_mutation(parents)

    #Population selection functions

    def ready_state_selection(self, offsprings: List[ScQbfSolution], parents: List[ScQbfSolution]) -> List[ScQbfSolution]:
        worst = self.get_worse_chromosome(parents)
        best = self.get_best_chromosome(offsprings)
        if (self.ObjFunction.evaluate_objfun(best) > self.ObjFunction.evaluate_objfun(worst)):
            parents.remove(worst)
            parents.append(best)
        
        return parents

    def default_select_population(self, mutants: List[ScQbfSolution]) -> List[ScQbfSolution]:
        worse = self.get_worse_chromosome(mutants)
        if self.ObjFunction.evaluate_objfun(worse) < self.ObjFunction.evaluate_objfun(self.best_chromosome):
            mutants.remove(worse)
            mutants.append(self.best_chromosome)
        return mutants

    def select_population(self, mutants: List[ScQbfSolution], parents: List[ScQbfSolution]) -> ScQbfSolution:
        population_selection = self.config.get("population_selection", "default")
        if (population_selection == "READY_STATE"):
            return self.ready_state_selection(mutants, parents)
        else:
            return self.default_select_population(mutants)

    #Core functions

    def evolve_population(self, population: List[ScQbfSolution], generation: int) -> List[ScQbfSolution]:
        parents = self.select_parents(population)
        offsprings = self.crossover(parents)
        mutants = self.mutate(offsprings)
        new_population = self.select_population(mutants, parents)
        
        self.best_chromosome = self.get_best_chromosome(new_population)
        
        if self.ObjFunction.evaluate_objfun(self.best_chromosome) > self.best_sol._last_objfun_val:
            self.best_sol = self.best_chromosome
            if self.verbose:
                print(f"(Gen. {generation}) BestSol = {self.best_sol}")
        
        # Record generation data
        self._record_generation_data(generation, self.best_sol._last_objfun_val)
        
        return new_population

    def solve(self, timeout_minutes: int = 30, config_ind: str = "") -> ScQbfSolution:
        # Set up timeout signal (Unix-like systems)
        try:
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout_minutes * 60)
        except AttributeError:
            # Windows doesn't support SIGALRM, we'll use manual checking
            print("Warning: SIGALRM not supported on this system, using manual timeout checking")
        
        self.start_time = time.time()
        self.timed_out = False
        
        try:
            population = self.initialize_population()
            
            self.best_sol = self.get_best_chromosome(population)
            self.best_chromosome = self.best_sol
            # Record generation 0 data
            self._record_generation_data(0, self.best_sol._last_objfun_val)
            print(f"(Gen. 0) BestSol = {self.best_sol._last_objfun_val}")
            
            for g in range(1, self.generations + 1):
                # Check timeout manually (for Windows compatibility)
                self.check_timeout(timeout_minutes)
                
                population = self.evolve_population(population, g)
            
        except TimeoutError as e:
            print(f"\n TIMEOUT: {e}")
            self.timed_out = True
            
        except Exception as e:
            print(f"\n ERROR: {e}")
            raise e
            
        finally:
            # Cancel alarm if it was set
            try:
                signal.alarm(0)
            except AttributeError:
                pass
            
            execution_time = time.time() - self.start_time
            print(f"Execution time: {execution_time:.2f} seconds")
            
            # Add execution metadata to the DataFrame
            self.generation_data['total_execution_time'] = execution_time
            self.generation_data['timed_out'] = self.timed_out
            self.generation_data['instance_name'] = os.path.basename(self.filename)
            self.generation_data['pop_size'] = self.pop_size
            self.generation_data['mutation_rate'] = self.mutation_rate
            
            # Save the data
            output_filename = f"ga_results_config{config_ind}_{os.path.basename(self.filename)}_{int(time.time())}.parquet"
            self.save_generation_data(output_filename)
            
            if self.timed_out:
                print(f"Run timed out after {timeout_minutes} minutes")
            else:
                print("Run completed successfully")
        
        return self.best_sol