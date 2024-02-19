from typing import Union, List

import numpy as np

from src.Tools import (qubo_cost,
                       string_to_array,
                       array_to_string,
                       normalized_cost)
from spynoza import QuantumCircuit, QuantumRegister, PyState

from src.CP_VQA.CP_VQA import CP_VQA
from src.Grid import Grid
from src.Chain import Chain
from src.Qubo import Qubo
from src.custom_spynoza_gates import RXX, RYY


class Spynoza_CP_VQA(CP_VQA):
    def __init__(self,
                 N_qubits,
                 cardinality,
                 layers,
                 qubo: Qubo,
                 topology: Union[Grid, Chain],
                 with_next_nearest_neighbors: bool = False):
        super().__init__(N_qubits, cardinality, layers, qubo, topology, with_next_nearest_neighbors)

    def set_circuit(self, angles):

        __angles__ = iter(angles)
        register = QuantumRegister(self.n_qubits)
        circuit = QuantumCircuit(register)

        # Setting 'k' qubits to |1>
        for qubit_index in self.initialization_strategy:
            circuit.x(qubit_index)

        for layer in range(self.layers):
            # XX+YY terms
            for (qubit_i, qubit_j) in self.qubit_indices:
                theta_ij = next(__angles__)
                RXX(circuit=circuit, qubit_1 = qubit_i, qubit_2 = qubit_j, angle=theta_ij)
                RYY(circuit=circuit, qubit_1 = qubit_i, qubit_2 = qubit_j, angle=theta_ij)

        return circuit

    @staticmethod
    def PyState_to_NumpyArray(state: PyState) -> np.ndarray:
        return np.array([state.__getitem__(i)[0] + 1j * state.__getitem__(i)[1] for i in range(state.__len__())],
                        dtype=np.complex64)

    def get_cost(self, angles) -> float:
        circuit = self.set_circuit(angles)
        circuit.execute()
        state_vector = self.PyState_to_NumpyArray(circuit.state_vector)
        self.counts = self.get_counts(state_vector=state_vector)
        cost = np.mean([probability * qubo_cost(state=string_to_array(bitstring), QUBO_matrix=self.QUBO.Q) for
                        bitstring, probability in self.counts.items()])
        return cost

    def get_state_vector(self, angles: Union[np.ndarray[float], List[float]]) -> np.ndarray:
        circuit = self.set_circuit(angles)
        circuit.execute()
        return self.PyState_to_NumpyArray(circuit.state_vector)

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

