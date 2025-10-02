from typing import List, TypeVar

F = TypeVar('F')

class Solution(List[F]):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cost = float('inf')
    
    def __str__(self):
        return f"Solution: cost=[{self.cost}], size=[{len(self)}], elements={super().__str__()}"