from typing import Sequence, Tuple, List
from time import time

import cirq
from cirq.ops.named_qubit import NamedQubit
import sympy
import numpy as np

from src.Tools import get_ising, qubo_cost, string_to_array


class Cirq_QAOA:
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
        self.simulator = cirq.Simulator()

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

    def set_circuit(self):

        gammas = [sympy.Symbol(f"𝛄_{layer}") for layer in range(self.layers)]
        betas = [sympy.Symbol(f"β_{layer}") for layer in range(self.layers)]

        qubits = [cirq.NamedQubit(f'q_{i}') for i in range(self.n_qubits)]
        h_list = [(qubits[i], h_i) for i, h_i in self.h_list]
        J_list = [(qubits[i], qubits[j], j_ij) for i, j, j_ij in self.J_list]

        def single_rotation_gamma_layer(gamma_value: float,
                                        h_list: List[Tuple[NamedQubit, float]]) -> Sequence[cirq.Operation]:
            """ Generator for R_z in U(gamma, C) layer of QAOA """
            for qubit_i, h_i in h_list:
                yield cirq.Z(qubit_i) ** (gamma_value * h_i)

        def multi_rotation_gamma_layer(gamma_value: float,
                                       J_list: List[Tuple[NamedQubit, NamedQubit, float]]) -> Sequence[cirq.Operation]:
            """ Generator for R_zz in U(gamma, C) layer of QAOA """
            for qubit_i, qubit_j, j_i in J_list:
                yield cirq.ZZ(qubit_i, qubit_j) ** (gamma_value * j_i)

        def beta_layer(beta_value: float, indices: list[NamedQubit]) -> Sequence[cirq.Operation]:
            """Generator for U(beta, B) layer (mixing layer) of QAOA"""
            for qubit_i in indices:
                yield cirq.X(qubit_i) ** beta_value

        # Initial state: Hadamard gate on each qubit
        circuit = cirq.Circuit(cirq.H.on_each(qubits))

        for layer in range(self.layers):
            # Implement the U(gamma, C) operator.
            circuit.append(single_rotation_gamma_layer(gamma_value=gammas[layer], h_list=h_list))
            circuit.append(multi_rotation_gamma_layer(gamma_value=gammas[layer], J_list=J_list))

            # Implement the U(beta, B) operator.
            circuit.append(beta_layer(betas[layer], indices=qubits))

        return circuit

    def get_cost(self, angles) -> float:
        __start__ = time()
        circuit = self.set_circuit()
        gamma_values = angles[self.layers:]
        beta_values = angles[:self.layers]
        total_dict = {**{f"𝛄_{i}": gamma_values[i] for i in range(len(gamma_values))},
                      **{f"β_{i}": beta_values[i] for i in range(len(beta_values))}}
        params = cirq.ParamResolver(total_dict)
        state_vector = self.simulator.simulate(circuit, param_resolver=params).final_state_vector
        self.counts = self.get_counts(state_vector=np.array(state_vector))
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
