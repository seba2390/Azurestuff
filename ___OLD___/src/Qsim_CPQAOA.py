from typing import List, Union
from itertools import combinations
import os

import cirq
import qsimcirq
import sympy
import numpy as np

from src.custom_gates.custom_cirq_gates import RXX, RYY
from src.Tools import qubo_cost, string_to_array
from src.Grid import Grid
from src.Chain import Chain


class Qsim_CPQAOA:
    def __init__(self,
                 N_qubits,
                 cardinality,
                 layers,
                 QUBO_matrix,
                 topology: Union[Grid, Chain],
                 get_full_state_vector: bool = True,
                 with_z_phase: bool = False,
                 with_next_nearest_neighbors: bool = False,
                 approximate_hamiltonian: bool = True):

        if not approximate_hamiltonian:
            raise ValueError('Exact Hamiltonian not implemented yet...')
        self.approximate_hamiltonian = approximate_hamiltonian
        if with_z_phase:
            raise ValueError('with z-phase not implemented yet...')
        self.with_z_phase = with_z_phase
        self.get_full_state_vector = get_full_state_vector
        self.n_qubits = N_qubits
        self.cardinality = cardinality
        self.layers = layers
        self.Q = QUBO_matrix.astype(np.float32)
        self.with_next_nearest_neighbors = with_next_nearest_neighbors

        if topology.N_qubits != self.n_qubits:
            raise ValueError(f'provided topology consists of different number of qubits that provided for this ansatz.')

        # Nearest Neighbors
        self.nearest_neighbor_pairs = topology.get_NN_indices()
        # Nearest + Next Nearest Neighbors
        self.next_nearest_neighbor_pairs = topology.get_NNN_indices()
        # Strategy for which qubits to set:
        self.initialization_strategy = topology.get_initialization_indices()
        # Indices to iterate over
        self.qubit_indices = self.next_nearest_neighbor_pairs if self.with_next_nearest_neighbors else self.nearest_neighbor_pairs

        self.states_strings = self.generate_bit_strings(N=self.n_qubits, k=self.cardinality)
        self.states_ints = [int(string, 2) for string in self.states_strings]

        # For storing probability <-> state dict during opt. to avoid extra call for callback function
        self.counts = None
        options = qsimcirq.QSimOptions(max_fused_gate_size=3, cpu_threads=os.cpu_count())
        self.simulator = qsimcirq.QSimSimulator(options)
        self.circuit = self.set_circuit()
        self.cost_time, self.circuit_time = 0.0, 0.0

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
            bit_strings.append(''.join(bit_string))
        return bit_strings

    @staticmethod
    def filter_small_probabilities(counts: dict[str, float], eps: float = 9e-15) -> dict[str, float]:
        return {state: prob for state, prob in counts.items() if prob >= eps}

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
                if self.approximate_hamiltonian:
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
            self.counts = self.filter_small_probabilities({self.states_strings[i]: probabilities[i] for i in range(len(probabilities))})
        cost = np.mean([probability * qubo_cost(state=string_to_array(bitstring), QUBO_matrix=self.Q) for
                        bitstring, probability in self.counts.items()])
        return cost

    def get_state_probabilities(self, flip_states: bool = True) -> dict:
        counts = self.counts
        if flip_states:
            return {bitstring[::-1]: probability for bitstring, probability in counts.items()}
        return {bitstring[::-1]: probability for bitstring, probability in counts.items()}
