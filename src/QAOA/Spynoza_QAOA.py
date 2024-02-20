from typing import Union, List

from spynoza import QuantumCircuit, QuantumRegister, PyState
import numpy as np

from src.Tools import qubo_cost, string_to_array, normalized_cost, array_to_string
from src.QAOA.QAOA import QAOA
from src.Qubo import Qubo
from src.custom_gates.custom_spynoza_gates import RZZ, RZ, RX


class Spynoza_QAOA(QAOA):
    def __init__(self, N_qubits: int,
                 cardinality: int,
                 layers: int,
                 qubo: Qubo):
        super().__init__(N_qubits, cardinality, layers, qubo)


    @staticmethod
    def PyState_to_NumpyArray(state: PyState) -> np.ndarray:
        return np.array([state.__getitem__(i)[0] + 1j * state.__getitem__(i)[1] for i in range(state.__len__())],
                        dtype=np.complex64)

    def set_circuit(self, angles):

        gamma, beta = angles[self.layers:], angles[:self.layers]

        register = QuantumRegister(self.n_qubits)
        circuit = QuantumCircuit(register)

        # Initial state: Hadamard gate on each qubit
        for q_i in range(self.n_qubits):
            circuit.h(q_i)
        for layer in range(self.layers):
            # Cost
            for q_i, h_i in self.h_list:
                RZ(circuit=circuit, qubit=q_i, angle=2 * h_i * gamma[layer])
            for q_i, q_j, j_ij in self.J_list:
                RZZ(circuit=circuit, qubit_1=q_i, qubit_2=q_j, angle=2 * j_ij * gamma[layer])
            # Mixer
            for q_i in range(self.n_qubits):
                RX(circuit=circuit, qubit=q_i, angle=2 * beta[layer])
        return circuit

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
        probability_dict = self.counts
        most_probable_state = string_to_array(list(probability_dict.keys())[np.argmax(list(probability_dict.values()))])
        if np.sum(most_probable_state) == self.k:
            normalized_c = normalized_cost(state=most_probable_state,
                                           QUBO_matrix=self.QUBO.Q,
                                           QUBO_offset=0.0,
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
