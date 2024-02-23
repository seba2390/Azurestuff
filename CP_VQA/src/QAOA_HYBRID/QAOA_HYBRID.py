from typing import Union, List
from abc import ABC, abstractmethod
from itertools import combinations

import numpy as np

from src.Qubo import Qubo
from src.Ising import get_ising
from src.Grid import Grid
from src.Chain import Chain


class QAOA_HYBRID(ABC):
    def __init__(self, N_qubits: int,
                 cardinality: int,
                 layers: int,
                 qubo: Qubo,
                 topology: Union[Grid, Chain],
                 with_next_nearest_neighbors: bool = False):
        self.n_qubits = N_qubits
        self.layers = layers
        self.k = cardinality
        self.QUBO = qubo
        self.J_list, self.h_list = get_ising(qubo=self.QUBO)
        self.topology = topology
        if topology.N_qubits != self.n_qubits:
            raise ValueError(f'provided topology consists of different number of qubits that provided for this ansatz.')
        self.with_next_nearest_neighbors = with_next_nearest_neighbors
        # Nearest Neighbors
        self.nearest_neighbor_pairs = topology.get_NN_indices()
        # Nearest + Next Nearest Neighbors
        self.next_nearest_neighbor_pairs = topology.get_NNN_indices()
        # Strategy for which qubits to set:
        self.initialization_strategy = topology.get_initialization_indices()
        # Indices to iterate over
        self.qubit_indices = self.next_nearest_neighbor_pairs if self.with_next_nearest_neighbors \
            else self.nearest_neighbor_pairs
        # For storing probability <-> state dict during opt. to avoid extra call for callback function
        self.counts = None
        self.normalized_costs = []
        self.opt_state_probabilities = []

    @abstractmethod
    def set_circuit(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_cost(self, angles: Union[np.ndarray[float], List[float]]) -> float:
        pass

    @abstractmethod
    def get_state_vector(self, angles: Union[np.ndarray[float], List[float]]) -> np.ndarray[complex]:
        pass

    @abstractmethod
    def callback(self, *args, **kwargs) -> None:
        pass

    @staticmethod
    def filter_small_probabilities(counts: dict[str, float], eps: float = 9e-15) -> dict[str, float]:
        return {state: prob for state, prob in counts.items() if prob >= eps}

    @staticmethod
    def _int_to_fixed_length_binary_array_(number: int, num_bits: int) -> str:
        # Convert the number to binary and remove the '0b' prefix
        binary_str = bin(number)[2:]
        # Pad the binary string with zeros if necessary
        return binary_str.zfill(num_bits)

    def get_counts(self, state_vector: np.ndarray) -> dict[str, float]:
        n_qubits = int(np.log2(len(state_vector)))
        return {self._int_to_fixed_length_binary_array_(number=idx, num_bits=n_qubits): np.abs(state_vector[idx]) ** 2
                for idx in range(len(state_vector))}

    @staticmethod
    def generate_bit_strings(N, k) -> List[str]:
        """
        Generate all bit strings of length N with k ones.

        Parameters:
        N (int): The length of the bit strings.
        k (int): The number of ones in each bit string.

        Returns:
        List[str]: A list of all bit strings of length N with k ones.
        """
        bit_strings = []
        for positions in combinations(range(N), k):
            bit_string = ['0'] * N
            for pos in positions:
                bit_string[pos] = '1'
            bit_strings.append(''.join(bit_string)[::-1])
        return bit_strings


