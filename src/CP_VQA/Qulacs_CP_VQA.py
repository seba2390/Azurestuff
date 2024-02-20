from typing import Union

import numpy as np
import qulacs.circuit

from qulacs import QuantumCircuit, ParametricQuantumCircuit
from qulacs.gate import X
from qulacs import QuantumState

from src.CP_VQA.CP_VQA import CP_VQA
from src.Tools import qubo_cost, string_to_array, array_to_string, normalized_cost
from src.custom_gates.custom_qulacs_gates import RXX, RYY, parametric_RXX, parametric_RYY
from src.Chain import Chain
from src.Qubo import Qubo
from src.Grid import Grid


class Qulacs_CP_VQA(CP_VQA):
    def __init__(self,
                 N_qubits: int,
                 cardinality: int,
                 layers: int,
                 qubo: Qubo,
                 topology: Union[Grid, Chain],
                 with_next_nearest_neighbors: bool = False,
                 get_full_state_vector: bool = False,
                 use_parametric_circuit_opt: bool = True):
        super().__init__(N_qubits, cardinality, layers, qubo, topology, with_next_nearest_neighbors)

        self.get_full_state_vector = get_full_state_vector
        self.use_parametric_circuit_opt = use_parametric_circuit_opt

        self.block_size = 2
        self.optimizer = qulacs.circuit.QuantumCircuitOptimizer()

        __dummy_angles__ = np.random.uniform(-2 * np.pi, 2 * np.pi, self.layers * len(self.qubit_indices))
        self.circuit = self.set_circuit(angles=__dummy_angles__)

        self.states_strings = self.generate_bit_strings(N=self.n_qubits, k=self.k)
        self.states_ints = [int(string, 2) for string in self.states_strings]

    def set_circuit(self, angles):

        if self.use_parametric_circuit_opt:
            qcircuit = ParametricQuantumCircuit(self.n_qubits)
        else:
            qcircuit = QuantumCircuit(self.n_qubits)

        # Initial state: 'k' excitations
        for qubit_idx in self.initialization_strategy:
            qcircuit.add_gate(X(index=qubit_idx))

        # Layered Ansatz
        angle_counter = 0
        for layer in range(self.layers):
            for qubit_i, qubit_j in self.qubit_indices:
                theta = angles[angle_counter]
                if self.use_parametric_circuit_opt:
                    parametric_RXX(circuit=qcircuit, angle=theta, qubit_1=qubit_i, qubit_2=qubit_j, use_native=True)
                    parametric_RYY(circuit=qcircuit, angle=theta, qubit_1=qubit_i, qubit_2=qubit_j, use_native=True)
                else:
                    RXX(circuit=qcircuit, angle=theta, qubit_1=qubit_i, qubit_2=qubit_j, use_native=True)
                    RYY(circuit=qcircuit, angle=theta, qubit_1=qubit_i, qubit_2=qubit_j, use_native=True)
                angle_counter += 1

        if self.use_parametric_circuit_opt:
            # Optimize the circuit (reduce nr. of gates)
            self.optimizer.optimize(circuit=qcircuit, block_size=self.block_size)
        return qcircuit

    def get_state_vector(self, angles):
        if self.use_parametric_circuit_opt:
            idx_counter = 0
            for theta_i in angles:
                # Same angle for both Rxx and Ryy
                self.circuit.set_parameter(index=idx_counter, parameter=theta_i)
                self.circuit.set_parameter(index=idx_counter + 1, parameter=theta_i)
                idx_counter += 2
        else:
            self.circuit = self.set_circuit(angles)
        state = QuantumState(self.n_qubits)
        self.circuit.update_quantum_state(state)
        return np.array(state.get_vector())

    def get_cost(self, angles):
        if self.use_parametric_circuit_opt:
            idx_counter = 0
            for theta_i in angles:
                # Same angle for both Rxx and Ryy
                self.circuit.set_parameter(index=idx_counter, parameter=theta_i)
                self.circuit.set_parameter(index=idx_counter + 1, parameter=theta_i)
                idx_counter += 2
        else:
            self.circuit = self.set_circuit(angles)
        state = QuantumState(self.n_qubits)
        self.circuit.update_quantum_state(state)
        if self.get_full_state_vector:
            state_vector = state.get_vector()
            self.counts = self.filter_small_probabilities(self.get_counts(state_vector=np.array(state_vector)))
        else:
            probabilities = np.array([np.abs(state.get_amplitude(comp_basis=s)) ** 2 for s in self.states_ints],
                                     dtype=np.float32)
            self.counts = self.filter_small_probabilities(
                {self.states_strings[i]: np.float32(probabilities[i]) for i in range(len(probabilities))})
        cost = np.mean([probability * qubo_cost(state=string_to_array(bitstring), QUBO_matrix=self.QUBO.Q) for
                        bitstring, probability in self.counts.items()])
        return cost

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
