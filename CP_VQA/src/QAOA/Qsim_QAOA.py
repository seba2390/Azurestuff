import os

import cirq
import qsimcirq
import sympy
import numpy as np

from src.Tools import (qubo_cost,
                       string_to_array,
                       array_to_string,
                       normalized_cost)

from src.QAOA.QAOA import QAOA
from src.Qubo import Qubo
from src.custom_gates.custom_cirq_gates import RZZ, RZ, RX


class Qsim_QAOA(QAOA):
    def __init__(self, N_qubits: int,
                 cardinality: int,
                 layers: int,
                 qubo: Qubo):
        super().__init__(N_qubits, cardinality, layers, qubo)

        # Read bottom of page: https://quantumai.google/cirq/simulate/simulation
        # and consider: https://github.com/quantumlib/qsim/blob/master/docs/tutorials/qsimcirq.ipynb
        # and consider https://developer.nvidia.com/blog/accelerating-quantum-circuit-simulation-with-nvidia-custatevec/
        options = qsimcirq.QSimOptions(max_fused_gate_size=3, cpu_threads=os.cpu_count())
        self.simulator = qsimcirq.QSimSimulator(options)
        self.circuit = self.set_circuit()

    def set_circuit(self):

        gamma = [sympy.Symbol(f"gamma_{i}") for i in range(self.layers)]
        beta = [sympy.Symbol(f"beta_{i}") for i in range(self.layers)]
        qubits = [cirq.NamedQubit(f'q_{i}') for i in range(self.n_qubits)]

        # Initial state: |+>
        circuit = cirq.Circuit()
        for qubit_idx in range(self.n_qubits):
            circuit.append(cirq.H(qubits[qubit_idx]))

        # Layered Ansatz
        for layer in range(self.layers):
            # ------ Cost unitary: ------ #
            # Weighted RZZ gate for each edge
            for qubit_i, qubit_j, J_ij in self.J_list:
                # Counting backwards to match Qiskit convention
                qubit_i, qubit_j = self.n_qubits - qubit_i - 1, self.n_qubits - qubit_j - 1
                RZZ(circuit=circuit, angle=2 * gamma[layer] * J_ij, qubit_1=qubits[qubit_i], qubit_2=qubits[qubit_j])
            # Weighted RZ gate for each qubit
            for qubit_i, h_i in self.h_list:
                # Counting backwards to match Qiskit convention
                qubit_i = self.n_qubits - qubit_i - 1
                RZ(circuit=circuit, qubit=qubits[qubit_i], angle=2 * gamma[layer] * h_i)
            # ------ Mixer unitary: ------ #
            # Weighted X rotation on each qubit
            for qubit_i in range(self.n_qubits):
                RX(circuit=circuit, qubit=qubits[qubit_i], angle=2 * beta[layer])
        return circuit

    def get_cost(self, angles) -> float:
        gamma_values, beta_values = angles[self.layers:], angles[:self.layers]
        total_dict = {**{f"gamma_{i}": gamma_values[i] for i in range(len(gamma_values))},
                      **{f"beta_{i}": beta_values[i] for i in range(len(beta_values))}}
        state_vector = self.simulator.simulate(program=self.circuit,
                                               param_resolver=cirq.ParamResolver(total_dict)).final_state_vector
        self.counts = self.get_counts(state_vector=np.array(state_vector))
        cost = np.mean([probability * qubo_cost(state=string_to_array(bitstring), QUBO_matrix=self.QUBO.Q) for
                        bitstring, probability in self.counts.items()])
        return cost

    def get_state_vector(self, angles):
        gamma_values = angles[self.layers:]
        beta_values = angles[:self.layers]
        total_dict = {**{f"gamma_{i}": gamma_values[i] for i in range(len(gamma_values))},
                      **{f"beta_{i}": beta_values[i] for i in range(len(beta_values))}}
        params = cirq.ParamResolver(total_dict)
        state_vector = self.simulator.simulate(self.circuit, param_resolver=params).final_state_vector
        return state_vector

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

