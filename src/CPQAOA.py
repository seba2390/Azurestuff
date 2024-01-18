from typing import List, Dict, Union
import random
from collections import Counter

from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import Operator
from qiskit.opflow import X, Y
import numpy as np
from scipy.linalg import expm

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

        self.mid_circuit_states, self.mid_circuit_unitaries = None, None
        if self.with_gradient:
            self.mid_circuit_states = []
            self.mid_circuit_indices = []

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

    def reset(self):
        self.mid_circuit_states, self.mid_circuit_indices = [], []

    def set_circuit(self, angles):
        self.reset()
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

        XX_YY_counter = 0
        for layer in range(self.layers):
            # XX+YY terms
            for (qubit_i, qubit_j) in self.qubit_indices:
                theta_ij = XX_YY_angles[XX_YY_counter]
                # For gradient calculation
                if self.with_gradient:
                    psi_i = np.array(execute(qcircuit, self.simulator).result().get_statevector())
                    self.mid_circuit_states.append(psi_i)
                    self.mid_circuit_indices.append((qubit_i, qubit_j))
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

            # Z-terms
            if self.with_z_phase:
                # Setting aside last N*L angles for z-phase
                Z_Phase_angles = angles[-self.n_qubits * self.layers:]
                for qubit_i, theta_i in zip(list(range(self.n_qubits)), Z_Phase_angles):
                    # For gradient calculation
                    """if self.with_gradient:
                        psi_i = np.array(execute(qcircuit, self.simulator).result().get_statevector())
                        self.mid_circuit_states.append(psi_i)
                        self.mid_circuit_indices.append((qubit_i, qubit_j))"""
                    qcircuit.rz(phi=2 * theta_i, qubit=qubit_i)

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
        assert not self.with_z_phase, 'not implemented for "with z-phase" yet...'

        # Populating stuff
        __ = self.set_circuit(angles=angles)

        def CTP(A: np.ndarray, B: np.ndarray) -> np.ndarray:
            """CTP: Conjugate Transpose Project"""
            return A.T.conj() @ (B @ A)

        def commutator(A: np.ndarray, B: np.ndarray) -> np.ndarray:
            return A @ B - B @ A

        derivatives = []
        for psi_i, (i, j), theta_i in zip(self.mid_circuit_states, self.mid_circuit_indices, angles):
            qcircuit = QuantumCircuit(self.n_qubits)
            hamiltonian = theta_i * (X ^ X) + theta_i * (Y ^ Y)
            time_evolved_operator = PauliEvolutionGate(hamiltonian, time=1.0)
            qcircuit.append(time_evolved_operator, [i, j])
            U_G_theta = np.array(Operator(qcircuit))
            d_c_d_theta_i = 1j / 2.0 * CTP(psi_i, CTP(U_G_theta,
                                                      commutator(get_generator(i, j, theta_i, self.n_qubits), self.O)))

            # This below should work but sometimes is wrong??
            """H_theta = get_generator(i, j, theta_i, self.n_qubits, flip=True)
            U_theta = expm(-1j*H_theta)
            d_c_d_theta_i = 1j / 2.0 * CTP(psi_i, CTP(U_theta, commutator(H_theta, self.O)))"""

            derivatives.append(d_c_d_theta_i)

        return np.real(np.array(derivatives))

    def get_state_probabilities(self, flip_states: bool = True) -> Dict:
        counts = self.counts
        if flip_states:
            return {bitstring[::-1]: probability for bitstring, probability in counts.items()}
        return {bitstring: probability for bitstring, probability in counts.items()}

