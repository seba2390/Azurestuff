from typing import List, Tuple, Union
from time import time
import os

import numpy as np

from src.Tools import get_ising, qubo_cost, string_to_array

from qulacs import QuantumCircuit
from qulacs.gate import RZ, RX, H, CNOT
from qulacs import Observable
from qulacs.state import inner_product
from qulacs import QuantumState


class Qulacs_QAOA:
    def __init__(self, N_qubits, layers, QUBO_matrix, QUBO_offset):
        self.n_qubits = N_qubits
        self.layers = layers
        self.QUBO_matrix = QUBO_matrix
        self.J_list, self.h_list = get_ising(Q=QUBO_matrix, offset=QUBO_offset)

        self.counts = None
        self.cost_time, self.circuit_time = 0.0, 0.0

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

    def set_circuit(self, angles):
        gamma = angles[self.layers:]
        beta = angles[:self.layers]

        qcircuit = QuantumCircuit(self.n_qubits)

        # Initial state: Hadamard gate on each qubit
        for qubit_index in range(self.n_qubits):
            qcircuit.add_gate(H(index=qubit_index))

        # For each Cost, Mixer repetition
        for layer in range(self.layers):

            # Cost unitary:
            for qubit_i, qubit_j, J_ij in self.J_list:
                qcircuit.add_gate(CNOT(control=qubit_i, target=qubit_j))
                qcircuit.add_gate(RZ(index=qubit_j, angle=2 * gamma[layer] * J_ij))
                qcircuit.add_gate(CNOT(control=qubit_i, target=qubit_j))

            for qubit_i, h_i in self.h_list:
                qcircuit.add_gate(RZ(index=qubit_i, angle=2 * gamma[layer] * h_i))

            # Mixer unitary:
            for qubit_i in range(self.n_qubits):
                qcircuit.add_gate(RX(index=qubit_i, angle=2 * beta[layer]))

        return qcircuit

    def get_cost(self, angles):
        circuit = self.set_circuit(angles=angles)
        state = QuantumState(self.n_qubits)
        circuit.update_quantum_state(state)
        state_vector = state.get_vector()
        self.counts = self.get_counts(state_vector=np.array(state_vector))
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
