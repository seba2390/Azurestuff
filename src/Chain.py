from typing import List, Tuple
import numpy as np


class Chain:
    def __init__(self,
                 N_qubits: int = None) -> None:

        self.N_qubits = N_qubits
        self.initialization_strategy = None

    def get_chain_indexing(self) -> np.ndarray:
        return np.array([i for i in range(self.N_qubits)])

    def get_NN_indices(self) -> List[Tuple[int, int]]:
        """ Returns pairs of indices corresponding to
        Nearest Neighbor interactions in the 1D chain structure """
        return [(q_1, q_1+1) for q_1 in range(0,self.N_qubits-1)]

    def get_NNN_indices(self) -> List[Tuple[int, int]]:
        """ Returns pairs of indices corresponding to both Nearest Neighbor
        and Next Nearest Neighbor interactions in the 1D chain structure """
        return [(q_1, q_1+1+i) for q_1 in range(0,self.N_qubits-2) for i in range(2)]+[(self.N_qubits-2,self.N_qubits-1)]

    def set_initialization_strategy(self, strategy: np.ndarray) -> None:
        if len(strategy) != self.N_qubits:
            raise ValueError('Size of strategy does not match number of qubits.')
        if np.any((strategy != 0) & (strategy != 1)):
            raise ValueError('Strategy should binary 1d array.')
        self.initialization_strategy = strategy

    def get_initialization_strategy(self) -> np.ndarray:
        if self.initialization_strategy is None:
            raise RuntimeError('Initialization strategy not yet defined.')
        return self.initialization_strategy

    def get_initialization_indices(self) -> List[int]:
        if self.initialization_strategy is None:
            raise RuntimeError('Initialization strategy not yet defined.')
        return self.get_chain_indexing()[np.where(self.initialization_strategy == 1)].flatten().tolist()


