from typing import List, Union

import numpy as np

import pennylane as qml
from pennylane.measurements import StateMP

from src.QAOA.QAOA import QAOA
from src.custom_gates.custom_pennylane_gates import RZZ, RZ, RX
from src.Tools import qubo_cost, string_to_array, array_to_string, normalized_cost
from src.Qubo import Qubo


class Pennylane_QAOA(QAOA):
    def __init__(self, N_qubits: int,
                 cardinality: int,
                 layers: int,
                 qubo: Qubo):
        super().__init__(N_qubits, cardinality, layers, qubo)

        # TODO: Check out the 'lightning.gpu' plugin, which is a fast state-vector simulator offloading
        # to the NVIDIA cuQuantum SDK for GPU accelerated circuit simulation. (not supported on windows...)
        self.device = qml.device('lightning.qubit', wires=self.n_qubits)
        self.circuit = self.compile_circuit()

    def compile_circuit(self):
        @qml.qnode(self.device)
        def circuit(angles) -> StateMP:
            gamma = angles[self.layers:]
            beta = angles[:self.layers]
            # apply Hadamard gates to get the n qubit |+> state
            for wire in range(self.n_qubits):
                qml.Hadamard(wires=wire)

            # p instances of unitary operators
            for layer in range(self.layers):
                # ------ Cost unitary: ------ #
                # Weighted RZZ gate for each edge
                for qubit_i, qubit_j, J_ij in self.J_list:
                    # Counting backwards to match Qiskit convention
                    qubit_i, qubit_j = self.n_qubits - qubit_i - 1, self.n_qubits - qubit_j - 1
                    RZZ(angle=2 * gamma[layer] * J_ij, qubit_1=qubit_i, qubit_2=qubit_j)

                # Weighted RZ gate for each qubit
                for qubit_i, h_i in self.h_list:
                    # Counting backwards to match Qiskit convention
                    qubit_i = self.n_qubits - qubit_i - 1
                    RZ(angle=2 * gamma[layer] * h_i, qubit=qubit_i)

                # ------ Mixer unitary: ------ #
                # Mixer unitary: Weighted X rotation on each qubit
                for qubit_i in range(self.n_qubits):
                    # Counting backwards to match Qiskit convention
                    qubit_i = self.n_qubits - qubit_i - 1
                    RX(angle=2 * beta[layer], qubit=qubit_i)

            return qml.state()

        return circuit

    def set_circuit(self, angles):
        return self.circuit(angles=angles)

    def get_cost(self, angles) -> float:
        state_vector = self.get_state_vector(angles=angles)
        self.counts = self.get_counts(state_vector=state_vector)
        cost = np.mean([probability * qubo_cost(state=string_to_array(bitstring), QUBO_matrix=self.QUBO.Q) for
                        bitstring, probability in self.counts.items()])
        return cost

    def get_state_vector(self, angles):
        return np.array(self.set_circuit(angles=angles))

    def callback(self, x):
        probability_dict = self.counts
        most_probable_state = string_to_array(list(probability_dict.keys())[np.argmax(list(probability_dict.values()))])
        if np.sum(most_probable_state) == self.k:
            normalized_c = normalized_cost(state=most_probable_state,
                                           QUBO_matrix=self.QUBO.Q,
                                           QUBO_offset=self.QUBO.offset,
                                           max_cost=self.QUBO.subspace_c_max,
                                           min_cost=self.QUBO.subspace_c_min)
            self.normalized_costs.append(normalized_c)
        else:
            self.normalized_costs.append(1)
        x_min_str = array_to_string(array=self.QUBO.subspace_x_min)
        if x_min_str in list(probability_dict.keys()):
            self.opt_state_probabilities.append(probability_dict[x_min_str])
        else:
            self.opt_state_probabilities.append(0)
