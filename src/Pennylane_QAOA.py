from typing import *
from time import time

import numpy as np
import pennylane as qml
from pennylane.measurements import StateMP

from src.Tools import get_ising, qubo_cost, string_to_array


class Pennylane_QAOA:
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

        self.circuit_time, self.cost_time = 0.0, 0.0

    @staticmethod
    def _int_to_fixed_length_binary_array_(number, num_bits):
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
        # TODO: Check out the 'lightning.gpu' plugin, which is a fast state-vector simulator offloading
        # to the NVIDIA cuQuantum SDK for GPU accelerated circuit simulation. (not supported on windows...)
        __device__ = qml.device('lightning.qubit', wires=self.n_qubits)

        def RZ(angle: float, qubit: int) -> None:
            qml.RZ(phi=angle, wires=qubit)

        def RZZ(angle: float, qubit_1: int, qubit_2: int) -> None:
            qml.CNOT(wires=[qubit_1, qubit_2])
            qml.RZ(phi=angle, wires=qubit_2)
            qml.CNOT(wires=[qubit_1, qubit_2])

        @qml.qnode(__device__)
        def circuit(gamma: Union[List[float], np.ndarray[float]],
                    beta: Union[List[float], np.ndarray[float]]) -> StateMP:

            # apply Hadamard gates to get the n qubit |+> state
            for wire in range(self.n_qubits):
                qml.Hadamard(wires=wire)

            # p instances of unitary operators
            for layer in range(self.layers):
                # ------ Cost unitary: ------ #
                # Weighted RZZ gate for each edge
                for qubit_i, qubit_j, J_ij in self.J_list:
                    RZZ(angle=2 * gamma[layer] * J_ij, qubit_1=qubit_i, qubit_2=qubit_j)

                # Weighted RZ gate for each qubit
                for qubit_i, h_i in self.h_list:
                    RZ(angle=2 * gamma[layer] * h_i, qubit=qubit_i)

                # ------ Mixer unitary: ------ #
                # Mixer unitary: Weighted X rotation on each qubit
                for qubit_i in range(self.n_qubits):
                    qml.RX(2 * beta[layer], wires=qubit_i)

            # TODO: investigate what it means that returned state-vector is in lexicographic order
            # See: https://docs.pennylane.ai/en/stable/code/api/pennylane.state.html
            return qml.state()

        return circuit(gamma=gamma, beta=beta)

    def get_cost(self, angles) -> float:
        __start__ = time()
        state_vector = self.set_circuit(angles=angles)
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