from typing import Union

import numpy as np

import pennylane as qml
from pennylane.measurements import StateMP

from src.CP_VQA.CP_VQA import CP_VQA
from src.custom_gates.custom_pennylane_gates import RXX, RYY
from src.Tools import (qubo_cost,
                       string_to_array,
                       array_to_string,
                       normalized_cost)
from src.Grid import Grid
from src.Chain import Chain
from src.Qubo import Qubo


class Pennylane_CP_VQA(CP_VQA):
    def __init__(self,
                 N_qubits,
                 cardinality,
                 layers,
                 qubo: Qubo,
                 topology: Union[Grid, Chain],
                 with_next_nearest_neighbors: bool = False,
                 seed: int = 0):
        super().__init__(N_qubits, cardinality, layers, qubo, topology, with_next_nearest_neighbors)
        np.random.seed(seed)

        self.states_strings = self.generate_bit_strings(N=self.n_qubits, k=self.k)
        self.states_ints = [int(string, 2) for string in self.states_strings]

        # TODO: Check out the 'lightning.gpu' plugin, which is a fast state-vector simulator offloading
        # to the NVIDIA cuQuantum SDK for GPU accelerated circuit simulation. (not supported on windows...)
        self.device = qml.device('lightning.qubit', wires=self.n_qubits)
        self.circuit = self.compile_circuit()

    def compile_circuit(self):
        @qml.qnode(self.device)
        def circuit(angles) -> StateMP:
            __angles__ = iter(angles)

            # Initial state: 'k' excitations
            for qubit_idx in self.initialization_strategy:
                # Counting backwards to match Qiskit convention
                qubit_idx = self.n_qubits - qubit_idx - 1
                qml.PauliX(wires=qubit_idx)

            # p instances of unitary operators
            for layer in range(self.layers):
                for qubit_i, qubit_j in self.qubit_indices:
                    # Counting backwards to match Qiskit convention
                    qubit_i, qubit_j = self.n_qubits - qubit_i - 1, self.n_qubits - qubit_j - 1
                    theta_ij = next(__angles__)
                    RXX(angle=theta_ij, qubit_1=qubit_i, qubit_2=qubit_j)
                    RYY(angle=theta_ij, qubit_1=qubit_i, qubit_2=qubit_j)

            return qml.state()

        return circuit

    def set_circuit(self, angles):
        return self.circuit(angles=angles)

    def get_cost(self, angles) -> float:
        state_vector = self.get_state_vector(angles=angles)
        self.counts = self.get_counts(state_vector=np.array(state_vector))
        probabilities = np.array([np.abs(state_vector[s]) ** 2 for s in self.states_ints],
                                 dtype=np.float32)
        self.counts = self.filter_small_probabilities(
            {self.states_strings[i]: np.float32(probabilities[i]) for i in range(len(probabilities))})
        cost = np.mean([probability * qubo_cost(state=string_to_array(bitstring), QUBO_matrix=self.QUBO.Q) for
                        bitstring, probability in self.counts.items()])
        return cost

    def get_state_vector(self, angles):
        return np.array(self.set_circuit(angles=angles))

    def callback(self, x):
        eps = 1e-5
        probability_dict = self.counts
        most_probable_state = string_to_array(list(probability_dict.keys())[np.argmax(list(probability_dict.values()))])
        normalized_c = normalized_cost(state=most_probable_state,
                                       QUBO_matrix=self.QUBO.Q,
                                       QUBO_offset=self.QUBO.offset,
                                       max_cost=self.QUBO.subspace_c_max,
                                       min_cost=self.QUBO.subspace_c_min)
        if 0 - eps > normalized_c or 1 + eps < normalized_c:
            raise ValueError(
                f'Not a valid normalized cost for Qulacs_CPVQA. Specifically, the normalized cost is: {normalized_c}'
                f'and this is given for most probable state: {most_probable_state}')
        self.normalized_costs.append(normalized_c)
        x_min_str = array_to_string(array=self.QUBO.subspace_x_min)
        if x_min_str in list(probability_dict.keys()):
            self.opt_state_probabilities.append(probability_dict[x_min_str])
        else:
            self.opt_state_probabilities.append(0)
