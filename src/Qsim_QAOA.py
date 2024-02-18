import os

import cirq
import qsimcirq
import sympy
import numpy as np

from src.Tools import (qubo_cost,
                       string_to_array,
                       array_to_string,
                       normalized_cost)

from src.Qubo import Qubo
from src.Ising import get_ising
from src.custom_cirq_gates import RZZ, RZ, RX


class Qsim_QAOA:
    def __init__(self,
                 N_qubits: int,
                 cardinality: int,
                 layers: int,
                 qubo: Qubo,
                 normalize_cost: bool = False):
        self.n_qubits = N_qubits
        self.k = cardinality
        self.layers = layers
        self.QUBO = qubo
        self.J_list, self.h_list = get_ising(qubo=qubo)

        self.normalize_cost = normalize_cost
        # Read bottom of page: https://quantumai.google/cirq/simulate/simulation
        # and consider: https://github.com/quantumlib/qsim/blob/master/docs/tutorials/qsimcirq.ipynb
        # and consider https://developer.nvidia.com/blog/accelerating-quantum-circuit-simulation-with-nvidia-custatevec/
        options = qsimcirq.QSimOptions(max_fused_gate_size=3, cpu_threads=os.cpu_count())
        self.simulator = qsimcirq.QSimSimulator(options)

        self.circuit = self.set_circuit()
        # For storing probability <-> state dict during opt. to avoid extra call for callback function
        self.counts = None
        self.normalized_costs = []
        self.opt_state_probabilities = []

    @staticmethod
    def _int_to_fixed_length_binary_array_(number: int, num_bits: int) -> str:
        # Convert the number to binary and remove the '0b' prefix
        binary_str = bin(number)[2:]
        # Pad the binary string with zeros if necessary
        return binary_str.zfill(num_bits)

    def get_counts(self, state_vector: np.ndarray) -> dict[str, float]:
        n_qubits = int(np.log2(len(state_vector)))
        return {self._int_to_fixed_length_binary_array_(number=idx, num_bits=n_qubits): np.abs(state_vector[idx]) ** 2
                for idx in range(len(state_vector))}

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

    def get_statevector(self, angles):
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

    def get_state_probabilities(self, flip_states: bool = True) -> dict:
        counts = self.counts
        if flip_states:
            return {bitstring[::-1]: probability for bitstring, probability in counts.items()}
        return {bitstring[::-1]: probability for bitstring, probability in counts.items()}
