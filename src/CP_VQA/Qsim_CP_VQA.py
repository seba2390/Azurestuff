from typing import Union
import os

import cirq
import qsimcirq
import sympy
import numpy as np

from src.CP_VQA.CP_VQA import CP_VQA
from src.custom_cirq_gates import RXX, RYY
from src.Tools import qubo_cost, string_to_array, array_to_string, normalized_cost
from src.Grid import Grid
from src.Chain import Chain
from src.Qubo import Qubo


class Qsim_CP_VQA(CP_VQA):
    def __init__(self,
                 N_qubits,
                 cardinality,
                 layers,
                 qubo: Qubo,
                 topology: Union[Grid, Chain],
                 get_full_state_vector: bool = False,
                 with_next_nearest_neighbors: bool = False,
                 seed: int = 0):
        super().__init__(N_qubits, cardinality, layers, qubo, topology, with_next_nearest_neighbors)
        np.random.seed(seed)

        self.get_full_state_vector = get_full_state_vector
        self.states_strings = self.generate_bit_strings(N=self.n_qubits, k=self.k)
        self.states_ints = [int(string, 2) for string in self.states_strings]
        options = qsimcirq.QSimOptions(max_fused_gate_size=3, cpu_threads=os.cpu_count())
        self.simulator = qsimcirq.QSimSimulator(options)
        self.circuit = self.set_circuit()

    def set_circuit(self):

        N_angles = len(self.qubit_indices) * self.layers
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
            for qubit_i, qubit_j in self.qubit_indices:
                # Counting backwards to match Qiskit convention
                qubit_i, qubit_j = self.n_qubits - qubit_i - 1, self.n_qubits - qubit_j - 1
                q_i, q_j = qubits[qubit_i], qubits[qubit_j]
                theta = thetas[angle_counter]
                RXX(circuit=circuit, angle=theta, qubit_1=q_i, qubit_2=q_j)
                RYY(circuit=circuit, angle=theta, qubit_1=q_i, qubit_2=q_j)
                angle_counter += 1

        return circuit

    def get_cost(self, angles) -> float:
        params = cirq.ParamResolver(param_dict={f"theta_{i}": angles[i] for i in range(len(angles))})
        if self.get_full_state_vector:
            state_vector = self.simulator.simulate(program=self.circuit, param_resolver=params).final_state_vector
            self.counts = self.filter_small_probabilities(self.get_counts(state_vector=np.array(state_vector)))
        else:
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
