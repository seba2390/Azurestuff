from typing import Sequence, Tuple, List
from time import time

import cirq
import sympy
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

        self.normalize_cost = normalize_cost

        self.counts = None
        self.cost_time, self.circuit_time = 0.0, 0.0

    def set_circuit(self, angles):

        gamma_vals = angles[self.layers:]
        beta_vals = angles[:self.layers]
        # Use sympy.Symbols for the 𝛾 and β parameters.
        gammas = [sympy.Symbol(f"𝛄_{layer}") for layer in range(layers)]
        betas = [sympy.Symbol(f"β_{layer}") for layer in range(layers)]

        qubits = [cirq.NamedQubit(f'q_{i}') for i in range(self.n_qubits)]
        h_list = [(qubits[i], h_i) for i, h_i in self.h_list]
        J_list = [(qubits[i], qubits[j], j_ij) for i, j, j_ij in self.J_list]

        def single_rotation_gamma_layer(gamma_value: float,
                                        h_list: List[Tuple[int, float]]) -> Sequence[cirq.Operation]:
            """ Generator for R_z in U(gamma, C) layer of QAOA """
            for qubit_i, h_i in h_list:
                yield cirq.Z(qubit_i) ** (gamma_value * h_i)

        def multi_rotation_gamma_layer(gamma_value: float,
                                       J_list: List[Tuple[int, float]]) -> Sequence[cirq.Operation]:
            """ Generator for R_zz in U(gamma, C) layer of QAOA """
            for qubit_i, qubit_j, j_i in J_list:
                yield cirq.ZZ(qubit_i, qubit_j) ** (gamma_value * j_i)

        def beta_layer(beta_value: float, indices: list[int]) -> Sequence[cirq.Operation]:
            """Generator for U(beta, B) layer (mixing layer) of QAOA"""
            for qubit_i in indices:
                yield cirq.X(qubit_i) ** beta_value


        # Initial state: Hadamard gate on each qubit
        circuit = cirq.Circuit(cirq.H.on_each(qubits))

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
