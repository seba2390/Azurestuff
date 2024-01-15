from typing import List, Dict, Union
import random
from collections import Counter

from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import Operator
from qiskit.opflow import X, Y
import numpy as np

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
                 seed: int = 0):
        random.seed(seed)

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
        NN_counter = 0

        if self.with_next_nearest_neighbors:
            # Setting aside next (N-2)*L angles for NNN-interactions
            NNN_angles_per_layer = len(self.next_nearest_neighbor_pairs)
            NNN_angles = angles[NN_angles_per_layer * self.layers:][:NNN_angles_per_layer * self.layers]
            NNN_counter = 0

        if self.with_z_phase:
            # Setting aside last N*L angles for z-phase
            Z_Phase_angles_per_layer = self.n_qubits
            Z_Phase_angles = angles[-Z_Phase_angles_per_layer * self.layers:]
            Z_Phase_counter = 0

        for layer in range(self.layers):
            # Nearest Neighbor
            for (qubit_i, qubit_j) in self.nearest_neighbor_pairs:
                theta_ij = NN_angles[NN_counter]

                # Define the Hamiltonian for XX and YY interactions
                xx_term = theta_ij * (X ^ X)
                yy_term = theta_ij * (Y ^ Y)
                hamiltonian = xx_term + yy_term

                # Create the time-evolved operator
                time_evolved_operator = PauliEvolutionGate(hamiltonian, time=1.0)

                # For gradient calculation
                if self.with_gradient:
                    psi_i = np.array(execute(qcircuit, self.simulator).result().get_statevector())
                    self.mid_circuit_states.append(psi_i)
                    self.mid_circuit_indices.append((qubit_i, qubit_j))
                qcircuit.append(time_evolved_operator, [qubit_i, qubit_j])

                # Increment counter for angles
                NN_counter += 1

            # Next Nearest Neighbor
            if self.with_next_nearest_neighbors:
                for qubit_i in range(self.n_qubits - 2):
                    theta_ij = NNN_angles[NNN_counter]
                    qubit_j = qubit_i + 2

                    # Define the Hamiltonian for XX and YY interactions
                    xx_term = theta_ij * (X ^ X)
                    yy_term = theta_ij * (Y ^ Y)
                    hamiltonian = xx_term + yy_term

                    # Create the time-evolved operator
                    time_evolved_operator = PauliEvolutionGate(hamiltonian, time=1.0)
                    # For gradient calculation
                    if self.with_gradient:
                        psi_i = np.array(execute(qcircuit, self.simulator).result().get_statevector())
                        self.mid_circuit_states.append(psi_i)
                        self.mid_circuit_indices.append((qubit_i, qubit_j))
                    qcircuit.append(time_evolved_operator, [qubit_i, qubit_j])
                    # Increment counter for angles
                    NNN_counter += 1

            if self.with_z_phase:
                for qubit_i in range(self.n_qubits):
                    theta_i = Z_Phase_angles[Z_Phase_counter]
                    # For gradient calculation
                    """if self.with_gradient:
                        psi_i = np.array(execute(qcircuit, self.simulator).result().get_statevector())
                        self.mid_circuit_states.append(psi_i)
                        self.mid_circuit_indices.append((qubit_i, qubit_j))"""
                    qcircuit.rz(phi=2 * theta_i, qubit=qubit_i)
                    # Increment counter for angles
                    Z_Phase_counter += 1

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

        def U_yy(theta, i, j) -> np.ndarray:
            QC = QuantumCircuit(self.n_qubits)
            QC.ryy(theta=theta, qubit1=i, qubit2=j)
            return np.array(Operator(QC))

        def U_xx(theta, i, j) -> np.ndarray:
            QC = QuantumCircuit(self.n_qubits)
            QC.rxx(theta=theta, qubit1=i, qubit2=j)
            return np.array(Operator(QC))

        def CTP(A: np.ndarray, B: np.ndarray) -> np.ndarray:
            """CTP: Conjugate Transpose Project"""
            return A.T.conj() @ (B @ A)

        def commutator(A: np.ndarray, B: np.ndarray) -> np.ndarray:
            return A @ B - B @ A

        derivatives = []
        for psi_i, (i, j), theta_i in zip(self.mid_circuit_states, self.mid_circuit_indices, angles):
            """U_yy_theta = U_yy(theta=theta_i, i=i, j=j)
            U_xx_theta = U_xx(theta=theta_i, i=i, j=j)

            U_xx_theta_pi_plus = U_xx(theta=theta_i + np.pi/2.0, i=i, j=j)
            U_xx_theta_pi_minus = U_xx(theta=theta_i - np.pi/2.0, i=i, j=j)

            U_yy_theta_pi_plus = U_yy(theta=theta_i + np.pi / 2.0, i=i, j=j)
            U_yy_theta_pi_minus = U_yy(theta=theta_i - np.pi / 2.0, i=i, j=j)

            d_c_d_theta_i_1 = CTP(psi_i, CTP(U_yy_theta, (CTP(U_xx_theta_pi_plus, self.O) - CTP(U_xx_theta_pi_minus, self.O))))
            d_c_d_theta_i_2 = CTP(psi_i, CTP(U_xx_theta, (CTP(U_yy_theta_pi_plus, self.O) - CTP(U_yy_theta_pi_minus, self.O))))
            d_c_d_theta_i = 0.5 * (d_c_d_theta_i_1 + d_c_d_theta_i_2)"""
            qcircuit = QuantumCircuit(self.n_qubits)
            # Define the Hamiltonian for XX and YY interactions
            xx_term = theta_i * (X ^ X)
            yy_term = theta_i * (Y ^ Y)
            hamiltonian = xx_term + yy_term
            # Create the time-evolved operator
            time_evolved_operator = PauliEvolutionGate(hamiltonian, time=1.0)
            qcircuit.append(time_evolved_operator, [i, j])
            U_G_theta = np.array(Operator(qcircuit))
            d_c_d_theta_i = 1j / 2.0 * CTP(psi_i, CTP(U_G_theta,
                                                      commutator(get_generator(i, j, theta_i, self.n_qubits), self.O)))
            derivatives.append(d_c_d_theta_i)

        return np.real_if_close(np.array(derivatives))

    def get_state_probabilities(self, flip_states: bool = True) -> Dict:
        counts = self.counts
        if flip_states:
            return {bitstring[::-1]: probability for bitstring, probability in counts.items()}
        return {bitstring: probability for bitstring, probability in counts.items()}

    def get_layer_prob_dist(self, N_layers: int, angles) -> List[Dict]:
        original_number_of_layers = self.layers
        result = []
        ## STARTING STATE ##
        qcircuit = QuantumCircuit(self.n_qubits)
        if self.with_evenly_distributed_start_x:
            # Distributing x-gates across string evenly
            for i in range(1, self.cardinality + 1):
                qcircuit.x(int(self.step_size * i))
        else:
            # Setting 'k' first with x-gates
            for qubit_index in range(self.cardinality):
                qcircuit.x(qubit_index)
        counts = execute(qcircuit, self.simulator).result().get_counts()
        result.append({bitstring: probability for bitstring, probability in counts.items()})

        ## REMAINING LAYERS ##
        for layers in range(1, N_layers + 1):
            self.layers = layers
            result.append(self.get_state_probabilities(angles=angles, flip_states=False))
        self.layers = original_number_of_layers
        return result
