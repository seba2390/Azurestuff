from typing import Union, List
import random
from collections import Counter

from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.synthesis import MatrixExponential
import numpy as np

from src.Tools import (qubo_cost,
                       string_to_array,
                       array_to_string,
                       normalized_cost)

from src.CP_VQA.CP_VQA import CP_VQA
from src.Grid import Grid
from src.Tools import get_qiskit_hamiltonian
from src.Chain import Chain
from src.Qubo import Qubo


class Qiskit_CP_VQA(CP_VQA):
    def __init__(self,
                 N_qubits,
                 cardinality,
                 layers,
                 qubo: Qubo,
                 topology: Union[Grid, Chain],
                 with_next_nearest_neighbors: bool = False,
                 approximate_hamiltonian: bool = True,
                 backend: str = 'state_vector',
                 N_samples: int = 1000,
                 seed: int = 0):
        super().__init__(N_qubits, cardinality, layers, qubo, topology, with_next_nearest_neighbors)
        random.seed(seed)

        self.approximate_hamiltonian = approximate_hamiltonian
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
            else:
                time = 1.0
                H = get_qiskit_hamiltonian(indices=self.qubit_indices,
                                           angles=angles[layer * len(angles) // self.layers:(layer + 1) * len(
                                               angles) // self.layers],
                                           N_qubits=self.n_qubits,
                                           with_z_phase=False)
                # MatrixExponential(): Exact operator evolution via matrix exponentiation and unitary synthesis
                # LieTrotter(reps=M): Approximates the exponential of two non-commuting operators with products of
                # their exponential up to a second order error, using "M" time steps
                U_H = PauliEvolutionGate(operator=H, time=time, synthesis=MatrixExponential())
                qcircuit.append(U_H, list(range(self.n_qubits)))
        return qcircuit

    def get_state_vector(self, angles: Union[np.ndarray[float], List[float]]) -> np.ndarray:
        circuit = self.set_circuit(angles=angles)
        return np.array(execute(circuit, self.simulator).result().get_statevector())

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

