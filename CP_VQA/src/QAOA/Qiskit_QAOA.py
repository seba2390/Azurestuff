from typing import List, Union

from qiskit import QuantumCircuit, Aer, execute
import numpy as np

from src.Tools import qubo_cost, string_to_array, array_to_string, normalized_cost
from src.Qubo import Qubo
from src.QAOA.QAOA import QAOA


class Qiskit_QAOA(QAOA):
    def __init__(self, N_qubits: int,
                 cardinality: int,
                 layers: int,
                 qubo: Qubo):
        super().__init__(N_qubits, cardinality, layers, qubo)
        self.simulator = Aer.get_backend('statevector_simulator')

    def set_circuit(self, angles: Union[np.ndarray[float], List[float]]):

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
            # Weighted X rotation on each qubit
            for qubit_i in range(self.n_qubits):
                qcircuit.rx(2 * beta[layer], qubit_i)

        return qcircuit

    def get_cost(self, angles: Union[np.ndarray[float], List[float]]) -> float:
        circuit = self.set_circuit(angles=angles)
        self.counts = execute(circuit, self.simulator).result().get_counts()
        cost = np.mean([probability * qubo_cost(state=string_to_array(bitstring), QUBO_matrix=self.QUBO.Q)
                        for bitstring, probability in self.counts.items()])
        return cost

    def get_state_vector(self, angles: Union[np.ndarray[float], List[float]]) -> np.ndarray:
        circuit = self.set_circuit(angles=angles)
        return np.array(execute(circuit, self.simulator).result().get_statevector())

    def callback(self, x):
        probability_dict = self.counts
        most_probable_state = string_to_array(list(probability_dict.keys())[np.argmax(list(probability_dict.values()))])
        """if np.sum(most_probable_state) == self.k:
            normalized_c = normalized_cost(state=most_probable_state,
                                           QUBO_matrix=self.QUBO.Q,
                                           QUBO_offset=0.0,
                                           max_cost=self.QUBO.subspace_c_max,
                                           min_cost=self.QUBO.subspace_c_min)
            self.normalized_costs.append(normalized_c)
        else:
            self.normalized_costs.append(1)"""
        normalized_c = normalized_cost(state=most_probable_state,
                                       QUBO_matrix=self.QUBO.Q,
                                       QUBO_offset=self.QUBO.offset if np.sum(most_probable_state) == self.k else 0.0,
                                       max_cost=self.QUBO.subspace_c_max,
                                       min_cost=self.QUBO.full_space_c_min)
        self.normalized_costs.append(normalized_c)
        x_min_str = array_to_string(array=self.QUBO.subspace_x_min)
        if x_min_str in list(probability_dict.keys()):
            self.opt_state_probabilities.append(probability_dict[x_min_str])
        else:
            self.opt_state_probabilities.append(0)
