from dataclasses import dataclass
from typing import List, Set

@dataclass
class ScQbfSolution:
    elements: List[int]
    _last_objfun_val: float = 0

    def __init__(self, n: int = None, el: List[int] = None):
        if n != None:
            self.elements = [0 for _ in range(n)]
        else:
            self.elements = el
