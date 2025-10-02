import os
import random
from abc import ABC, abstractmethod
from typing import List, TypeVar, Generic, Optional
from problems.Evaluator import Evaluator
from solutions.Solution import Solution
import pandas as pd
import time

G = TypeVar('G')
F = TypeVar('F')

class AbstractGA(Generic[G, F], ABC):
    verbose = True
    rng = random.Random(0)
    
    def __init__(self, obj_function: Evaluator[F], generations: int, pop_size: int, mutation_rate: float):
        self.ObjFunction = obj_function
        self.generations = generations
        self.pop_size = pop_size
        self.chromosome_size = self.ObjFunction.get_domain_size()
        self.mutation_rate = mutation_rate
        self.best_cost = float('-inf')
        self.best_sol: Optional[Solution[F]] = None
        self.best_chromosome: Optional[List[G]] = None
        # Initialize DataFrame to store generation data
        self.generation_data = pd.DataFrame(columns=['generation', 'best_cost', 'timestamp'])
    
    @abstractmethod
    def create_empty_sol(self) -> Solution[F]:
        pass
    
    @abstractmethod
    def decode(self, chromosome: List[G]) -> Solution[F]:
        pass
    
    @abstractmethod
    def generate_random_chromosome(self) -> List[G]:
        pass
    
    @abstractmethod
    def fitness(self, chromosome: List[G]) -> float:
        pass
    
    @abstractmethod
    def mutate_gene(self, chromosome: List[G], locus: int):
        pass
    
    def solve(self) -> Solution[F]:
        population = self.initialize_population()
        
        self.best_chromosome = self.get_best_chromosome(population)
        self.best_sol = self.decode(self.best_chromosome)
        
        # Record generation 0 data
        self._record_generation_data(0, self.best_sol.cost)
        print(f"(Gen. 0) BestSol = {self.best_sol}")
        
        for g in range(1, self.generations + 1):
            population = self.evolve_population(population, g)
        
        return self.best_sol
    
    def evolve_population(self, population: List[List[G]], generation: int) -> List[List[G]]:
        parents = self.select_parents(population)
        offsprings = self.crossover(parents)
        mutants = self.mutate(offsprings)
        new_population = self.select_population(mutants)
        
        self.best_chromosome = self.get_best_chromosome(new_population)
        
        if self.fitness(self.best_chromosome) > self.best_sol.cost:
            self.best_sol = self.decode(self.best_chromosome)
            if self.verbose:
                print(f"(Gen. {generation}) BestSol = {self.best_sol}")
        
        # Record generation data
        self._record_generation_data(generation, self.best_sol.cost)
        
        return new_population
    
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
    
    def initialize_population(self) -> List[List[G]]:
        population = []
        while len(population) < self.pop_size:
            population.append(self.generate_random_chromosome())
        return population
    
    def get_best_chromosome(self, population: List[List[G]]) -> List[G]:
        best_fitness = float('-inf')
        best_chromosome = None
        for c in population:
            fit = self.fitness(c)
            if fit > best_fitness:
                best_fitness = fit
                best_chromosome = c
        return best_chromosome
    
    def get_worse_chromosome(self, population: List[List[G]]) -> List[G]:
        worse_fitness = float('inf')
        worse_chromosome = None
        for c in population:
            fit = self.fitness(c)
            if fit < worse_fitness:
                worse_fitness = fit
                worse_chromosome = c
        return worse_chromosome
    
    def select_parents(self, population: List[List[G]]) -> List[List[G]]:
        parents = []
        while len(parents) < self.pop_size:
            index1 = self.rng.randint(0, self.pop_size - 1)
            index2 = self.rng.randint(0, self.pop_size - 1)
            parent1 = population[index1]
            parent2 = population[index2]
            if self.fitness(parent1) > self.fitness(parent2):
                parents.append(parent1)
            else:
                parents.append(parent2)
        return parents
    
    def crossover(self, parents: List[List[G]]) -> List[List[G]]:
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
                    offspring1.append(parent2[j])
                    offspring2.append(parent1[j])
                else:
                    offspring1.append(parent1[j])
                    offspring2.append(parent2[j])
            
            offsprings.append(offspring1)
            offsprings.append(offspring2)
        
        return offsprings
    
    def mutate(self, offsprings: List[List[G]]) -> List[List[G]]:
        for c in offsprings:
            for locus in range(self.chromosome_size):
                if self.rng.random() < self.mutation_rate:
                    self.mutate_gene(c, locus)
        return offsprings
    
    def select_population(self, offsprings: List[List[G]]) -> List[List[G]]:
        worse = self.get_worse_chromosome(offsprings)
        if self.fitness(worse) < self.fitness(self.best_chromosome):
            offsprings.remove(worse)
            offsprings.append(self.best_chromosome)
        return offsprings