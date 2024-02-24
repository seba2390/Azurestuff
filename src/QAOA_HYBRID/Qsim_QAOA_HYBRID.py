from typing import Union
import os

import cirq
#import qsimcirq
import sympy
import numpy as np

from src.QAOA_HYBRID.QAOA_HYBRID import QAOA_HYBRID
from src.custom_gates.custom_cirq_gates import RXX, RYY, RZZ, RZ
from src.Tools import qubo_cost, string_to_array, array_to_string, normalized_cost
from src.Grid import Grid
from src.Chain import Chain
from src.Qubo import Qubo


class Qsim_QAOA_HYBRID(QAOA_HYBRID):
    def __init__(self, N_qubits: int,
                 cardinality: int,
                 layers: int,
                 qubo: Qubo,
                 topology: Union[Grid, Chain],
                 with_next_nearest_neighbors: bool = False,
                 get_full_state_vector: bool = False):
        super().__init__(N_qubits, cardinality, layers, qubo, topology, with_next_nearest_neighbors)

        self.get_full_state_vector = get_full_state_vector
        self.states_strings = self.generate_bit_strings(N=self.n_qubits, k=self.k)
        self.states_ints = [int(string, 2) for string in self.states_strings]
        #options = qsimcirq.QSimOptions(max_fused_gate_size=3, cpu_threads=os.cpu_count())
        self.simulator = cirq.Simulator()
        #self.simulator = qsimcirq.QSimSimulator(options)
        self.circuit = self.set_circuit()

    def set_circuit(self):

        N_angles = len(self.qubit_indices) * self.layers
        N_angles += len(self.J_list) * self.layers
        N_angles += len(self.h_list) * self.layers
        thetas = [sympy.Symbol(f"theta_{i}") for i in range(N_angles)]
        qubits = [cirq.NamedQubit(f'q_{i}') for i in range(self.n_qubits)]

        # Initial state: "k" excitations
        circuit = cirq.Circuit()
        for qubit_idx in self.initialization_strategy:
            # Counting backwards to match Qiskit convention
            qubit_idx = self.n_qubits - qubit_idx - 1
            circuit.append(cirq.X(qubits[qubit_idx]))

        # Layered Ansatz
        angle_counter = 0
        for layer in range(self.layers):
            # ------ Cost unitary: ------ #
            # Weighted RZZ gate for each edge
            for qubit_i, qubit_j, J_ij in self.J_list:
                qubit_i, qubit_j = self.n_qubits - qubit_i - 1, self.n_qubits - qubit_j - 1
                q_i, q_j = qubits[qubit_i], qubits[qubit_j]
                RZZ(circuit=circuit, angle=thetas[angle_counter], qubit_1=q_i, qubit_2=q_j)
                angle_counter += 1
            # Weighted RZ gate for each qubit
            for qubit_i, h_i in self.h_list:
                qubit_i = self.n_qubits - qubit_i - 1
                q_i = qubits[qubit_i]
                RZ(circuit=circuit, angle=thetas[angle_counter], qubit=q_i)
                angle_counter += 1

            # ------ Mixer unitary: ------ #
            for qubit_i, qubit_j in self.qubit_indices:
                qubit_i, qubit_j = self.n_qubits - qubit_i - 1, self.n_qubits - qubit_j - 1
                q_i, q_j = qubits[qubit_i], qubits[qubit_j]
                RXX(circuit=circuit, angle=thetas[angle_counter], qubit_1=q_i, qubit_2=q_j)
                RYY(circuit=circuit, angle=thetas[angle_counter], qubit_1=q_i, qubit_2=q_j)
                angle_counter += 1

        return circuit

    def get_cost(self, angles) -> float:
        if self.get_full_state_vector:
            state_vector = self.get_state_vector(angles=angles)
            self.counts = self.filter_small_probabilities(self.get_counts(state_vector=np.array(state_vector)))
        else:
            cost_angles, mixer_angles = angles[:self.layers], angles[self.layers:]
            param_dict = {}
            internal_angle_counter = 0
            mixer_external_angle_counter = 0
            for layer in range(self.layers):
                # ------ Cost unitary: ------ #
                gamma = cost_angles[layer]
                # Weighted RZZ gate for each edge
                for qubit_i, qubit_j, J_ij in self.J_list:
                    param_dict[f'theta_{internal_angle_counter}'] = 2 * gamma * J_ij
                    internal_angle_counter += 1
                # Weighted RZ gate for each qubit
                for qubit_i, h_i in self.h_list:
                    param_dict[f'theta_{internal_angle_counter}'] = 2 * gamma * h_i
                    internal_angle_counter += 1
                # ------ Mixer unitary: ------ #
                beta = mixer_angles[mixer_external_angle_counter]
                for qubit_i, qubit_j in self.qubit_indices:
                    beta = mixer_angles[mixer_external_angle_counter]
                    param_dict[f'theta_{internal_angle_counter}'] = beta
                    internal_angle_counter += 1
                mixer_external_angle_counter += 1
            params = cirq.ParamResolver(param_dict=param_dict)
            probabilities = np.power(np.abs(self.simulator.compute_amplitudes(program=self.circuit,
                                                                              param_resolver=params,
                                                                              bitstrings=self.states_ints)), 2)
            self.counts = self.filter_small_probabilities(
                {self.states_strings[i]: probabilities[i] for i in range(len(probabilities))})
        cost = np.mean([probability * qubo_cost(state=string_to_array(bitstring), QUBO_matrix=self.QUBO.Q) for
                        bitstring, probability in self.counts.items()])
        return cost

    def get_state_vector(self, angles):
        params = cirq.ParamResolver(param_dict={f"theta_{i}": angles[i] for i in range(len(angles))})
        return np.array(self.simulator.simulate(program=self.circuit, param_resolver=params).final_state_vector)

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
