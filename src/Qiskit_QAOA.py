from typing import List, Tuple, Union
from time import time
import os

from qiskit import QuantumCircuit, Aer, execute
from qiskit.quantum_info import Operator
from scipy.linalg import expm
import numpy as np

from src.Tools import qubo_cost, string_to_array, array_to_string, normalized_cost
from src.Qubo import Qubo
from src.Ising import get_ising
from src.Grid import Grid
from src.Chain import Chain


class Qiskit_QAOA:
    def __init__(self,
                 N_qubits: int,
                 cardinality: int,
                 layers: int,
                 qubo: Qubo):
        self.n_qubits = N_qubits
        self.layers = layers
        self.k = cardinality
        self.QUBO = qubo
        self.J_list, self.h_list = get_ising(qubo=self.QUBO)
        self.simulator = Aer.get_backend('statevector_simulator')

        # For storing probability <-> state dict during opt. to avoid extra call for callback function
        self.counts = None
        self.normalized_costs = []
        self.opt_state_probabilities = []

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
            # Weighted X rotation on each qubit
            for qubit_i in range(self.n_qubits):
                qcircuit.rx(2 * beta[layer], qubit_i)

        return qcircuit

    def get_cost(self, angles) -> float:
        circuit = self.set_circuit(angles=angles)
        self.counts = execute(circuit, self.simulator).result().get_counts()
        cost = np.mean([probability * qubo_cost(state=string_to_array(bitstring), QUBO_matrix=self.QUBO.Q)
                        for bitstring, probability in self.counts.items()])
        return cost

    def get_state_probabilities(self, flip_states: bool = True) -> dict:
        counts = self.counts
        if flip_states:
            return {bitstring[::-1]: probability for bitstring, probability in counts.items()}
        return {bitstring: probability for bitstring, probability in counts.items()}

    def callback(self, x):
        probability_dict = self.get_state_probabilities(flip_states=False)
        most_probable_state = string_to_array(list(probability_dict.keys())[np.argmax(list(probability_dict.values()))])
        normalized_c = normalized_cost(state=most_probable_state,
                                       QUBO_matrix=self.QUBO.Q,
                                       QUBO_offset=0.0 if np.sum(most_probable_state) == self.k else self.QUBO.offset,
                                       max_cost=self.QUBO.full_space_c_max,
                                       min_cost=self.QUBO.full_space_c_min)
        self.normalized_costs.append(normalized_c)
        x_min_str = array_to_string(array=self.QUBO.subspace_x_min)
        if x_min_str in list(probability_dict.keys()):
            self.opt_state_probabilities.append(probability_dict[x_min_str])
        else:
            self.opt_state_probabilities.append(0)
