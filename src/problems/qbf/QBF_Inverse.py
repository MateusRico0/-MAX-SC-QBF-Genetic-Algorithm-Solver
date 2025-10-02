from .QBF import QBF

class QBF_Inverse(QBF):
    def evaluate_qbf(self) -> float:
        return -super().evaluate_qbf()
    
    def evaluate_insertion_qbf(self, i: int) -> float:
        return -super().evaluate_insertion_qbf(i)
    
    def evaluate_removal_qbf(self, i: int) -> float:
        return -super().evaluate_removal_qbf(i)
    
    def evaluate_exchange_qbf(self, in_idx: int, out_idx: int) -> float:
        return -super().evaluate_exchange_qbf(in_idx, out_idx)