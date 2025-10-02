from abc import ABC, abstractmethod
from typing import TypeVar, Generic
from solutions.Solution import Solution

F = TypeVar('F')

class Evaluator(Generic[F], ABC):
    @abstractmethod
    def get_domain_size(self) -> int:
        pass
    
    @abstractmethod
    def evaluate(self, sol: Solution[F]) -> float:
        pass
    
    @abstractmethod
    def evaluate_insertion_cost(self, elem: F, sol: Solution[F]) -> float:
        pass
    
    @abstractmethod
    def evaluate_removal_cost(self, elem: F, sol: Solution[F]) -> float:
        pass
    
    @abstractmethod
    def evaluate_exchange_cost(self, elem_in: F, elem_out: F, sol: Solution[F]) -> float:
        pass