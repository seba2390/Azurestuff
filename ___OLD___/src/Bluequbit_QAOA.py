from typing import List, Tuple, Union
from time import time

import bluequbit
from qiskit import QuantumCircuit, Aer, execute
import numpy as np

from src.Tools import get_ising, qubo_cost, string_to_array


class Bluequbit_QAOA:
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

        self.bq_client = bluequbit.init(api_token="y7wmv229WOmpT4f4eyVmdPhBg2sGYZCv")

        self.normalize_cost = normalize_cost

        self.counts = None
        self.cost_time, self.circuit_time = 0.0, 0.0

    def set_circuit(self, angles):

        gamma = angles[self.layers:]
        beta = angles[:self.layers]

        qcircuit = QuantumCircuit(self.n_qubits)

        # Initial state: Hadamard gate on each qubit
        for qubit_index in range(self.n_qubits):
            qcircuit.h(qubit_index)

        # For each Cost, Mixer repetition
        for layer in range(self.layers):
            # ------ Cost unitary: ------ #
            # Weighted RZZ gate for each edge
            for qubit_i, qubit_j, J_ij in self.J_list:
                qcircuit.rzz(2 * gamma[layer] * J_ij, qubit_i, qubit_j)

            # Weighted RZ gate for each qubit
            for qubit_i, h_i in self.h_list:
                qcircuit.rz(2 * gamma[layer] * h_i, qubit_i)

            # ------ Mixer unitary: ------ #
            for qubit_i in range(self.n_qubits):
                qcircuit.rx(2 * beta[layer], qubit_i)
        return qcircuit

    def get_cost(self, angles) -> float:
        __start__ = time()
        circuit = self.set_circuit(angles=angles)
        self.counts = self.bq_client.run(circuit, device='cpu').get_counts()
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
