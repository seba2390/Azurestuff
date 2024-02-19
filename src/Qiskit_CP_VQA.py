from typing import Dict, Union
import random
from collections import Counter

from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.synthesis import MatrixExponential, LieTrotter
import numpy as np

from src.Tools import (qubo_cost,
                       string_to_array,
                       array_to_string,
                       normalized_cost)
from src.Grid import Grid
from src.Tools import get_full_hamiltonian_matrix, get_qiskit_hamiltonian
from src.Chain import Chain
from src.Qubo import Qubo


class CP_VQA:
    def __init__(self,
                 N_qubits,
                 cardinality,
                 layers,
                 qubo: Qubo,
                 topology: Union[Grid, Chain],
                 with_z_phase: bool = False,
                 with_next_nearest_neighbors: bool = False,
                 with_gradient: bool = False,
                 approximate_hamiltonian: bool = True,
                 normalize_cost: bool = False,
                 backend: str = 'state_vector',
                 N_samples: int = 1000,
                 seed: int = 0):
        random.seed(seed)

        self.n_qubits = N_qubits
        self.cardinality = cardinality
        self.layers = layers
        self.QUBO = qubo

        # self.O = create_operator(Q=self.Q)
        self.with_next_nearest_neighbors = with_next_nearest_neighbors
        self.with_z_phase = with_z_phase
        self.approximate_hamiltonian = approximate_hamiltonian
        self.normalize_cost = normalize_cost
        self.with_gradient = with_gradient

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
        self.normalized_costs = []
        self.opt_state_probabilities = []

        if backend not in ['state_vector', 'sample']:
            raise ValueError(f'provided backend should be either "state_vector" or "sample"')
        self.backend = backend
        self.N_samples = N_samples

        # Using state-vector sim. for theoretical accuracy
        self.simulator = Aer.get_backend('statevector_simulator')

    def set_circuit(self, angles):
        __angles__ = iter(angles)
        # Defining circuit
        qcircuit = QuantumCircuit(self.n_qubits)

        # Setting 'k' qubits to |1>
        for qubit_index in self.initialization_strategy:
            qcircuit.x(qubit_index)

        for layer in range(self.layers):
            if self.approximate_hamiltonian:
                # XX+YY terms
                for (qubit_i, qubit_j) in self.qubit_indices:
                    theta_ij = next(__angles__)
                    qcircuit.rxx(theta=theta_ij, qubit1=qubit_i, qubit2=qubit_j)
                    qcircuit.ryy(theta=theta_ij, qubit1=qubit_i, qubit2=qubit_j)
                # Z terms
                if self.with_z_phase:
                    for qubit_i in range(self.n_qubits):
                        qcircuit.rz(phi=next(__angles__), qubit=qubit_i)
            else:
                time = 1.0
                H = get_qiskit_hamiltonian(indices=self.qubit_indices,
                                           angles=angles[layer * len(angles) // self.layers:(layer + 1) * len(
                                               angles) // self.layers],
                                           N_qubits=self.n_qubits,
                                           with_z_phase=self.with_z_phase)
                # MatrixExponential(): Exact operator evolution via matrix exponentiation and unitary synthesis
                # LieTrotter(reps=M): Approximates the exponential of two non-commuting operators with products of
                # their exponential up to a second order error, using "M" time steps
                U_H = PauliEvolutionGate(operator=H, time=time, synthesis=MatrixExponential())
                qcircuit.append(U_H, list(range(self.n_qubits)))
        return qcircuit

    def get_cost(self, angles) -> float:
        circuit = self.set_circuit(angles=angles)
        self.counts = execute(circuit, self.simulator).result().get_counts()
        if self.backend == 'sample':
            # Extract states and corresponding probabilities
            state_strings = list(self.counts.keys())
            probabilities = [self.counts[key] for key in state_strings]
            # Sample M times according to the probabilities
            samples = random.choices(state_strings, weights=probabilities, k=self.N_samples)
            # Count occurrences of each state in the samples
            sample_counts = Counter(samples)
            # Convert counts to probabilities
            self.counts = {key: count / self.N_samples for key, count in sample_counts.items()}
        return np.mean([probability * qubo_cost(state=string_to_array(bitstring), QUBO_matrix=self.QUBO.Q) for
                        bitstring, probability in self.counts.items()])

    def callback(self, x):
        eps = 1e-5

        probability_dict = self.get_state_probabilities(flip_states=False)
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

    def get_state_probabilities(self, flip_states: bool = True) -> Dict:
        counts = self.counts
        if flip_states:
            return {bitstring[::-1]: probability for bitstring, probability in counts.items()}
        return {bitstring: probability for bitstring, probability in counts.items()}
