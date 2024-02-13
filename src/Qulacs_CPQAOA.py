from typing import Union

import numpy as np

from qulacs import QuantumCircuit
from qulacs.gate import X
from qulacs import QuantumState

from src.Tools import get_ising, qubo_cost, string_to_array
from src.custom_qulacs_gates import RXX, RYY
from src.Chain import Chain
from src.Grid import Grid


class Qulacs_CPQAOA:
    def __init__(self,
                 N_qubits,
                 cardinality,
                 layers,
                 QUBO_matrix,
                 topology: Union[Grid, Chain],
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

        # For storing probability <-> state dict during opt. to avoid extra call for callback function
        self.counts = None

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
                    RXX(circuit=qcircuit, angle=theta, qubit_1=qubit_i, qubit_2=qubit_j)
                    RYY(circuit=qcircuit, angle=theta, qubit_1=qubit_i, qubit_2=qubit_j)
                    angle_counter += 1

        # Optimize the circuit
        #self.optimizer.optimize(qcircuit, self.block_size)
        return qcircuit

    def get_cost(self, angles):
        circuit = self.set_circuit(angles=angles)
        state = QuantumState(self.n_qubits)
        circuit.update_quantum_state(state)
        state_vector = state.get_vector()
        self.counts = self.filter_small_probabilities(self.get_counts(state_vector=np.array(state_vector)))
        cost = np.mean([probability * qubo_cost(state=string_to_array(bitstring), QUBO_matrix=self.Q) for
                        bitstring, probability in self.counts.items()])
        return cost


