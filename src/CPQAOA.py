from typing import List, Dict, Union
import random
from collections import Counter

from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import Parameter
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import Operator
from qiskit.opflow import X, Y
from qiskit.primitives import Estimator
from qiskit_algorithms.gradients import ParamShiftEstimatorGradient
import numpy as np
from scipy.linalg import expm
from Tools import get_qiskit_H

from src.Tools import qubo_cost, string_to_array, create_operator, operator_expectation, get_generator
from src.Grid import Grid
from src.Chain import Chain


class CP_QAOA:
    def __init__(self,
                 N_qubits,
                 cardinality,
                 layers,
                 QUBO_matrix,
                 topology: Union[Grid, Chain],
                 with_z_phase: bool = False,
                 with_next_nearest_neighbors: bool = False,
                 with_gradient: bool = False,
                 backend: str = 'state_vector',
                 N_samples: int = 1000,
                 seed: int = 0,
                 debug_verbose: bool = False):
        random.seed(seed)

        self.debug_verbose = debug_verbose

        self.n_qubits = N_qubits
        self.cardinality = cardinality
        self.layers = layers
        self.Q = QUBO_matrix
        self.O = create_operator(Q=self.Q)
        self.with_next_nearest_neighbors = with_next_nearest_neighbors
        self.with_z_phase = with_z_phase
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

        if backend not in ['state_vector', 'sample']:
            raise ValueError(f'provided backend should be either "state_vector" or "sample"')
        self.backend = backend
        self.N_samples = N_samples

        # Using state-vector sim. for theoretical accuracy
        self.simulator = Aer.get_backend('statevector_simulator')

    def set_circuit(self, angles):
        qcircuit = QuantumCircuit(self.n_qubits)

        # Setting 'k' qubits to |1>
        for qubit_index in self.initialization_strategy:
            qcircuit.x(qubit_index)

        # Setting aside first (N-1)*L angles for NN-interactions
        NN_angles_per_layer = len(self.nearest_neighbor_pairs)
        NN_angles = angles[:NN_angles_per_layer * self.layers]

        XX_YY_angles = list(NN_angles)
        if self.with_next_nearest_neighbors:
            # Setting aside next (N-2)*L angles for NNN-interactions
            NNN_angles_per_layer = len(self.next_nearest_neighbor_pairs)
            NNN_angles = angles[NN_angles_per_layer * self.layers:][:NNN_angles_per_layer * self.layers]
            XX_YY_angles += list(NNN_angles)

        Z_angles = None
        if self.with_z_phase:
            # Setting aside last N*L angles for z-phase
            Z_angles = angles[-self.n_qubits * self.layers:]

        XX_YY_counter, Z_counter = 0, 0
        for layer in range(self.layers):
            # XX+YY terms
            for (qubit_i, qubit_j) in self.qubit_indices:
                theta_ij = XX_YY_angles[XX_YY_counter]

                """# Define the Hamiltonian for XX and YY interactions
                xx_term = theta_ij * (X ^ X)
                yy_term = theta_ij * (Y ^ Y)
                hamiltonian = xx_term + yy_term
                # Create the time-evolved operator & add to circuit
                time_evolved_operator = PauliEvolutionGate(hamiltonian, time=1.0)
                qcircuit.append(time_evolved_operator, [qubit_i, qubit_j])"""
                qcircuit.rxx(theta=2*theta_ij, qubit1=qubit_i, qubit2=qubit_j)
                qcircuit.ryy(theta=2*theta_ij, qubit1=qubit_i, qubit2=qubit_j)
                XX_YY_counter += 1

            # Z terms
            if self.with_z_phase:
                for qubit_i in range(self.n_qubits):
                    theta_i = Z_angles[Z_counter]
                    qcircuit.rz(phi=2 * theta_i, qubit=qubit_i)
                    Z_counter += 1

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
        return np.mean([probability * qubo_cost(state=string_to_array(bitstring), QUBO_matrix=self.Q) for
                        bitstring, probability in self.counts.items()])

    def get_gradient(self, angles) -> np.ndarray:
        """ Using parameter shift rule to calculate exact derivatives"""

        params = [Parameter(f'theta_{i}') for i in range(len(angles))]

        qcircuit = QuantumCircuit(self.n_qubits)

        # Setting 'k' qubits to |1>
        for qubit_index in self.initialization_strategy:
            qcircuit.x(qubit_index)

        # Setting aside first (N-1)*L angles for NN-interactions
        NN_angles_per_layer = len(self.nearest_neighbor_pairs)
        NN_angles = params[:NN_angles_per_layer * self.layers]

        XX_YY_angles = list(NN_angles)
        if self.with_next_nearest_neighbors:
            # Setting aside next (N-2)*L angles for NNN-interactions
            NNN_angles_per_layer = len(self.next_nearest_neighbor_pairs)
            NNN_angles = params[NN_angles_per_layer * self.layers:][:NNN_angles_per_layer * self.layers]
            XX_YY_angles += list(NNN_angles)

        Z_angles = None
        if self.with_z_phase:
            # Setting aside last N*L angles for z-phase
            Z_angles = params[-self.n_qubits * self.layers:]

        XX_YY_counter = 0
        Z_counter = 0
        for layer in range(self.layers):
            # XX+YY terms
            for (qubit_i, qubit_j) in self.qubit_indices:
                theta_ij = XX_YY_angles[XX_YY_counter]

                qcircuit.rxx(theta=theta_ij, qubit1=qubit_i, qubit2=qubit_j)
                qcircuit.ryy(theta=theta_ij, qubit1=qubit_i, qubit2=qubit_j)
                XX_YY_counter += 1

            # Z-terms
            if self.with_z_phase:
                for qubit_i in range(self.n_qubits):
                    theta_i = Z_angles[Z_counter]
                    qcircuit.rz(phi=theta_i, qubit=qubit_i)
                    Z_counter += 1

        # Get cost hamiltonian
        H_c = get_qiskit_H(Q=self.Q)

        # Parameter values list (multiplying w. 2 as values in set circuit = 2*angle)
        param_values = [[2*theta for theta in angles]]

        # Define the gradient
        gradient = ParamShiftEstimatorGradient(Estimator())

        # Evaluate the gradient of the circuits using parameter shift gradients
        pse_grad_result = gradient.run(circuits=[qcircuit],
                                       observables=H_c,
                                       parameter_values=param_values).result().gradients

        return pse_grad_result

    def get_state_probabilities(self, flip_states: bool = True) -> Dict:
        counts = self.counts
        if flip_states:
            return {bitstring[::-1]: probability for bitstring, probability in counts.items()}
        return {bitstring: probability for bitstring, probability in counts.items()}

