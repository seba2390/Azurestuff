from itertools import combinations
from typing import List

import numpy as np

from qulacs import QuantumCircuit, ParametricQuantumCircuit
from qulacs.circuit import QuantumCircuitOptimizer
from qulacs.gate import H
from qulacs import QuantumState

from src.QAOA.QAOA import QAOA
from src.custom_qulacs_gates import RZZ, RZ, RX
from src.custom_qulacs_gates import parametric_RZZ, parametric_RZ, parametric_RX
from src.Tools import qubo_cost, string_to_array, array_to_string, normalized_cost
from src.Qubo import Qubo


class Qulacs_QAOA(QAOA):
    def __init__(self, N_qubits: int,
                 cardinality: int,
                 layers: int,
                 qubo: Qubo,
                 use_parametric_circuit_opt: bool = True):
        super().__init__(N_qubits, cardinality, layers, qubo)

        self.use_parametric_circuit_opt = use_parametric_circuit_opt
        self.block_size = 2
        self.optimizer = QuantumCircuitOptimizer()

        __dummy_angles__ = np.random.uniform(-2 * np.pi, 2 * np.pi, 2 * self.layers)
        self.circuit = self.set_circuit(angles=__dummy_angles__)

    @staticmethod
    def generate_bit_strings(N, k) -> List[str]:
        """
        Generate all bit strings of length N with k ones.

        Parameters:
        N (int): The length of the bit strings.
        k (int): The number of ones in each bit string.

        Returns:
        List[str]: A list of all bit strings of length N with k ones.
        """
        bit_strings = []
        for positions in combinations(range(N), k):
            bit_string = ['0'] * N
            for pos in positions:
                bit_string[pos] = '1'
            bit_strings.append(''.join(bit_string)[::-1])
        return bit_strings

    def set_circuit(self, angles):

        gamma = angles[self.layers:]
        beta = angles[:self.layers]

        if self.use_parametric_circuit_opt:
            qcircuit = ParametricQuantumCircuit(self.n_qubits)
        else:
            qcircuit = QuantumCircuit(self.n_qubits)

        # Initial state: initialize in |+>
        for qubit_idx in range(self.n_qubits):
            qcircuit.add_gate(H(index=qubit_idx))

        # Layered Ansatz
        for layer in range(self.layers):
            if self.use_parametric_circuit_opt:
                # ------ Cost unitary: ------ #
                # Weighted RZZ gate for each edge
                for qubit_i, qubit_j, J_ij in self.J_list:
                    parametric_RZZ(circuit=qcircuit, angle=2 * gamma[layer] * J_ij, qubit_1=qubit_i, qubit_2=qubit_j)
                # Weighted RZ gate for each qubit
                for qubit_i, h_i in self.h_list:
                    parametric_RZ(circuit=qcircuit, qubit=qubit_i, angle=2 * gamma[layer] * h_i)
                # ------ Mixer unitary: ------ #
                # Weighted X rotation on each qubit
                for qubit_i in range(self.n_qubits):
                    parametric_RX(circuit=qcircuit, qubit=qubit_i, angle=2 * beta[layer])
            else:
                # ------ Cost unitary: ------ #
                # Weighted RZZ gate for each edge
                for qubit_i, qubit_j, J_ij in self.J_list:
                    RZZ(circuit=qcircuit, angle=2 * gamma[layer] * J_ij, qubit_1=qubit_i, qubit_2=qubit_j)
                # Weighted RZ gate for each qubit
                for qubit_i, h_i in self.h_list:
                    RZ(circuit=qcircuit, qubit=qubit_i, angle=2 * gamma[layer] * h_i)
                # ------ Mixer unitary: ------ #
                # Weighted X rotation on each qubit
                for qubit_i in range(self.n_qubits):
                    RX(circuit=qcircuit, qubit=qubit_i, angle=2 * beta[layer])

        if self.use_parametric_circuit_opt:
            # Optimize the circuit (reduce nr. of gates)
            self.optimizer.optimize(circuit=qcircuit, block_size=self.block_size)
        return qcircuit

    def get_cost(self, angles):
        if self.use_parametric_circuit_opt:
            gamma, beta = angles[self.layers:], angles[:self.layers]
            gate_counter = 0
            for layer in range(self.layers):
                # ------ Cost unitary: ------ #
                # Weighted RZZ gate for each edge
                for qubit_i, qubit_j, J_ij in self.J_list:
                    self.circuit.set_parameter(index=gate_counter, parameter=2 * gamma[layer] * J_ij)
                    gate_counter += 1
                # Weighted RZ gate for each qubit
                for qubit_i, h_i in self.h_list:
                    self.circuit.set_parameter(index=gate_counter, parameter=2 * gamma[layer] * h_i)
                    gate_counter += 1
                # ------ Mixer unitary: ------ #
                # Weighted X rotation on each qubit
                for qubit_i in range(self.n_qubits):
                    self.circuit.set_parameter(index=gate_counter, parameter=2 * beta[layer])
                    gate_counter += 1
        else:
            self.circuit = self.set_circuit(angles)
        state = QuantumState(self.n_qubits)
        self.circuit.update_quantum_state(state)
        self.counts = self.filter_small_probabilities(self.get_counts(state_vector=np.array(state.get_vector())))
        cost = np.mean([probability * qubo_cost(state=string_to_array(bitstring), QUBO_matrix=self.QUBO.Q) for
                        bitstring, probability in self.counts.items()])
        return cost

    def get_state_vector(self, angles):
        if self.use_parametric_circuit_opt:
            gamma, beta = angles[self.layers:], angles[:self.layers]
            gate_counter = 0
            for layer in range(self.layers):
                # ------ Cost unitary: ------ #
                # Weighted RZZ gate for each edge
                for qubit_i, qubit_j, J_ij in self.J_list:
                    self.circuit.set_parameter(index=gate_counter, parameter=2 * gamma[layer] * J_ij)
                    gate_counter += 1
                # Weighted RZ gate for each qubit
                for qubit_i, h_i in self.h_list:
                    self.circuit.set_parameter(index=gate_counter, parameter=2 * gamma[layer] * h_i)
                    gate_counter += 1
                # ------ Mixer unitary: ------ #
                # Weighted X rotation on each qubit
                for qubit_i in range(self.n_qubits):
                    self.circuit.set_parameter(index=gate_counter, parameter=2 * beta[layer])
                    gate_counter += 1
        else:
            self.circuit = self.set_circuit(angles)
        state = QuantumState(self.n_qubits)
        self.circuit.update_quantum_state(state)
        return np.array(state.get_vector())

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


