import os
from time import time

from spynoza import QuantumCircuit, QuantumRegister, PyState
import numpy as np

from src.Tools import get_ising, qubo_cost, string_to_array


class Spynoza_QAOA:
    def __init__(self,
                 N_qubits,
                 layers,
                 QUBO_matrix,
                 QUBO_offset,
                 normalize_cost: bool = False):

        self.n_qubits = N_qubits
        self.layers = layers
        self.QUBO_matrix = QUBO_matrix
        self.J_list, self.h_list = get_ising(Q=QUBO_matrix, offset=QUBO_offset)

        self.normalize_cost = normalize_cost

        self.counts = None
        self.cost_time, self.circuit_time = 0.0, 0.0

    @staticmethod
    def _int_to_fixed_length_binary_array_(number: int, num_bits: int) -> str:
        # Convert the number to binary and remove the '0b' prefix
        binary_str = bin(number)[2:]
        # Pad the binary string with zeros if necessary
        return binary_str.zfill(num_bits)

    @staticmethod
    def PyState_to_NumpyArray(state: PyState) -> np.ndarray:
        return np.array([state.__getitem__(i)[0] + 1j * state.__getitem__(i)[1] for i in range(state.__len__())],
                        dtype=np.complex64)

    def get_counts(self, state_vector: np.ndarray) -> dict[str, float]:
        n_qubits = int(np.log2(len(state_vector)))
        return {self._int_to_fixed_length_binary_array_(number=idx, num_bits=n_qubits): np.abs(state_vector[idx]) ** 2
                for idx in range(len(state_vector))}

    def set_circuit(self, angles):

        gamma = angles[self.layers:]
        beta = angles[:self.layers]

        register = QuantumRegister(self.n_qubits)
        circuit = QuantumCircuit(register)

        # Initial state: Hadamard gate on each qubit
        for q_i in range(self.n_qubits):
            circuit.h(q_i)

        for layer in range(self.layers):
            # Cost
            for q_i, h_i in self.h_list:
                circuit.rz(2*h_i*gamma[layer], q_i)
            for q_i, q_j, j_ij in self.J_list:
                circuit.cx(q_i, q_j)
                circuit.rz(2*j_ij*gamma[layer], q_i)
                circuit.cx(q_i, q_j)
            # Mixer
            for q_i in range(self.n_qubits):
                circuit.rx(2*beta[layer], q_i)

        return circuit

    def get_cost(self, angles) -> float:
        __start__ = time()
        circuit = self.set_circuit(angles)
        circuit.execute()
        state_vector = self.PyState_to_NumpyArray(circuit.state_vector)
        self.counts = self.get_counts(state_vector=state_vector)
        __end__ = time()
        self.circuit_time += __end__ - __start__
        __start__ = time()
        cost = np.mean([probability * qubo_cost(state=string_to_array(bitstring), QUBO_matrix=self.QUBO_matrix) for
                        bitstring, probability in self.counts.items()])
        __end__ = time()
        self.cost_time += __end__ - __start__
        return cost

    def get_state_probabilities(self, flip_states: bool = True) -> dict:
        counts = self.counts
        if flip_states:
            return {bitstring[::-1]: probability for bitstring, probability in counts.items()}
        return {bitstring[::-1]: probability for bitstring, probability in counts.items()}
