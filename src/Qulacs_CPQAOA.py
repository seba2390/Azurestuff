from typing import Union, List
from itertools import combinations

import numpy as np
import qulacs.circuit

from qulacs import QuantumCircuit, ParametricQuantumCircuit
from qulacs.gate import X
from qulacs import QuantumState

from src.Tools import get_ising, qubo_cost, string_to_array
from src.custom_qulacs_gates import RXX, RYY, parametric_RXX, parametric_RYY
from src.Chain import Chain
from src.Grid import Grid


class Qulacs_CPQAOA:
    def __init__(self,
                 N_qubits,
                 cardinality,
                 layers,
                 QUBO_matrix,
                 topology: Union[Grid, Chain],
                 get_full_state_vector: bool = True,
                 use_parametric_circuit_opt: bool = True,
                 with_next_nearest_neighbors: bool = False,
                 approximate_hamiltonian: bool = True):

        if not approximate_hamiltonian:
            raise ValueError('Exact Hamiltonian not implemented yet...')
        self.approximate_hamiltonian = approximate_hamiltonian
        self.n_qubits = N_qubits
        self.cardinality = cardinality
        self.layers = layers
        self.Q = QUBO_matrix.astype(np.float32)
        self.with_next_nearest_neighbors = with_next_nearest_neighbors
        self.use_parametric_circuit_opt = use_parametric_circuit_opt
        self.get_full_state_vector = get_full_state_vector

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

        self.block_size = 2
        self.optimizer = qulacs.circuit.QuantumCircuitOptimizer()
        __dummy_angles__ = np.random.uniform(-2*np.pi, 2*np.pi, self.layers*len(self.qubit_indices))
        self.circuit = self.set_circuit(angles=__dummy_angles__)
        # For storing probability <-> state dict during opt. to avoid extra call for callback function
        self.counts = None
        self.states_strings = self.generate_bit_strings(N=self.n_qubits, k=self.cardinality)
        self.states_ints = [int(string, 2) for string in self.states_strings]

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
                if self.approximate_hamiltonian:
                    theta = angles[angle_counter]
                    if self.use_parametric_circuit_opt:
                        parametric_RXX(circuit=qcircuit, angle=theta, qubit_1=qubit_i, qubit_2=qubit_j)
                        parametric_RYY(circuit=qcircuit, angle=theta, qubit_1=qubit_i, qubit_2=qubit_j)
                    else:
                        RXX(circuit=qcircuit, angle=theta, qubit_1=qubit_i, qubit_2=qubit_j)
                        RYY(circuit=qcircuit, angle=theta, qubit_1=qubit_i, qubit_2=qubit_j)
                    angle_counter += 1

        if self.use_parametric_circuit_opt:
            # Optimize the circuit (reduce nr. of gates)
            self.optimizer.optimize(circuit=qcircuit, block_size=self.block_size)
        return qcircuit

    def get_cost(self, angles):
        if self.use_parametric_circuit_opt:
            idx_counter = 0
            for theta_i in angles:
                # Same angle for both Rxx and Ryy
                self.circuit.set_parameter(index=idx_counter, parameter=theta_i)
                self.circuit.set_parameter(index=idx_counter+1, parameter=theta_i)
                idx_counter += 2
        else:
            self.circuit = self.set_circuit(angles)
        state = QuantumState(self.n_qubits)
        self.circuit.update_quantum_state(state)
        if self.get_full_state_vector:
            state_vector = state.get_vector()
            self.counts = self.filter_small_probabilities(self.get_counts(state_vector=np.array(state_vector)))
        else:
            probabilities = np.array([np.abs(state.get_amplitude(comp_basis=s))**2 for s in self.states_ints], dtype=np.float32)
            self.counts = self.filter_small_probabilities({self.states_strings[i]: probabilities[i] for i in range(len(probabilities))})
        cost = np.mean([probability * qubo_cost(state=string_to_array(bitstring), QUBO_matrix=self.Q) for
                        bitstring, probability in self.counts.items()])
        return cost
