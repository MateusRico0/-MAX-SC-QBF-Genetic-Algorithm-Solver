import time
from typing import List
import signal
import os

from metaheuristics.ga.AbstractGA import AbstractGA
from problems.qbf.QBF import QBF
from solutions.Solution import Solution
import config

class TimeoutError(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutError("Execution time exceeded!")


class GA_QBF(AbstractGA[int, int]):
    def __init__(self, generations: int, pop_size: int, mutation_rate: float, filename: str):
        super().__init__(QBF(filename), generations, pop_size, mutation_rate)
        self.timed_out = False
        self.start_time = None
        self.filename = filename
    
    def create_empty_sol(self) -> Solution[int]:
        sol = Solution[int]()
        sol.cost = 0.0
        return sol
    
    def decode(self, chromosome: List[int]) -> Solution[int]:
        solution = self.create_empty_sol()
        for locus in range(len(chromosome)):
            if chromosome[locus] == 1:
                solution.append(locus)
        
        self.ObjFunction.evaluate(solution)
        return solution
    
    def generate_random_chromosome(self) -> List[int]:
        chromosome = []
        for _ in range(self.chromosome_size):
            chromosome.append(self.rng.randint(0, 1))
        return chromosome
    
    def fitness(self, chromosome: List[int]) -> float:
        return self.decode(chromosome).cost
    
    def mutate_gene(self, chromosome: List[int], locus: int):
        chromosome[locus] = 1 - chromosome[locus]

    def evolve_population(self, population: List[List[int]], generation: int) -> List[List[int]]:
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

    def check_timeout(self, timeout_minutes: int = 30):
        if self.start_time and (time.time() - self.start_time) > (timeout_minutes * 60):
            self.timed_out = True
            raise TimeoutError(f"Execution stopped: exceeded {timeout_minutes} minutes")
    
    def solve(self, timeout_minutes: int = 30) -> Solution[int]:
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
            
            self.best_chromosome = self.get_best_chromosome(population)
            self.best_sol = self.decode(self.best_chromosome)
            
            # Record generation 0 data
            self._record_generation_data(0, self.best_sol.cost)
            print(f"(Gen. 0) BestSol = {self.best_sol}")
            
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
            output_filename = f"ga_results_{os.path.basename(self.filename)}_{int(time.time())}.parquet"
            self.save_generation_data(output_filename)
            
            if self.timed_out:
                print(f"Run timed out after {timeout_minutes} minutes")
            else:
                print("Run completed successfully")
        
        return self.best_sol

    @staticmethod
    def main():
        try:
            for filename in config.INSTANCE_FILES:
                start_time = time.time()

                ga = GA_QBF(config.GENERATIONS, config.POP_SIZE, config.MUTATION_RATE, filename)
                best_sol = ga.solve(timeout_minutes=config.TIMEOUT_MINUTES)
                print(f"Final best cost: {best_sol.cost}")

                end_time = time.time()
                total_time = end_time - start_time
                print(f"Total execution time = {total_time} seconds")

        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    GA_QBF.main()