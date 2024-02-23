from typing import Union, List
from abc import ABC, abstractmethod

import numpy as np

from src.Qubo import Qubo
from src.Ising import get_ising


class QAOA(ABC):
    def __init__(self, N_qubits: int,
                 cardinality: int,
                 layers: int,
                 qubo: Qubo):
        self.n_qubits = N_qubits
        self.layers = layers
        self.k = cardinality
        self.QUBO = qubo
        self.J_list, self.h_list = get_ising(qubo=self.QUBO)
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


