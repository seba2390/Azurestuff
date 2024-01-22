from typing import List, Dict, Union
import random
from collections import Counter

from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import Parameter
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import Operator
from qiskit.opflow import X, Y
from qiskit.primitives import Estimator
from qiskit_algorithms.gradients import ParamShiftEstimatorGradient, LinCombEstimatorGradient, FiniteDiffEstimatorGradient
import numpy as np
from scipy.linalg import expm
import torch

from src.TorchQcircuit import TorchQcircuit
from src.Tools import get_qiskit_H
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
        __angles__ = iter(angles)

        # Defining circuit
        qcircuit = QuantumCircuit(self.n_qubits)

        # Setting 'k' qubits to |1>
        for qubit_index in self.initialization_strategy:
            qcircuit.x(qubit_index)

        for layer in range(self.layers):
            # XX+YY terms
            for (qubit_i, qubit_j) in self.qubit_indices:
                theta_ij = next(__angles__)
                qcircuit.rxx(theta=theta_ij, qubit1=qubit_i, qubit2=qubit_j)
                qcircuit.ryy(theta=theta_ij, qubit1=qubit_i, qubit2=qubit_j)
            # Z terms
            if self.with_z_phase:
                for qubit_i in range(self.n_qubits):
                    qcircuit.rz(phi=next(__angles__), qubit=qubit_i)

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
        """return np.mean([probability * qubo_cost(state=string_to_array(bitstring), QUBO_matrix=self.Q) for
                        bitstring, probability in self.counts.items()])"""
        H_c = np.array(Operator(get_qiskit_H(Q=self.Q)))
        state_vector = np.array(execute(circuit, self.simulator).result().get_statevector()).flatten()
        return float(np.real(np.dot(state_vector.conj(), np.dot(H_c, state_vector))))

    def get_gradient(self, angles) -> np.ndarray:
        """ Using parameter shift rule to calculate exact derivatives"""

        params = [Parameter(f'theta_{i}') for i in range(len(angles))]
        #__angles__ = iter(params)
        torch_angles = torch.tensor(angles,
                                       dtype=torch.cfloat,
                                       requires_grad=True)

        # Defining circuit
        #qcircuit = QuantumCircuit(self.n_qubits)
        qcircuit = TorchQcircuit(self.n_qubits)

        psi_0 = torch.tensor(np.array([1.0] + [0.0 for _ in range(2 ** self.n_qubits - 1)]),
                                             dtype=torch.cfloat,requires_grad=True)

        # Setting 'k' qubits to |1>
        for qubit_index in self.initialization_strategy:
            #qcircuit.x(qubit_index)
            qcircuit.add_x(qubit_index)

        counter = 0
        for layer in range(self.layers):
            # XX+YY terms
            for (qubit_i, qubit_j) in self.qubit_indices:
                theta_ij = torch_angles[counter]
                #qcircuit.rxx(theta=theta_ij, qubit1=qubit_i, qubit2=qubit_j)
                #qcircuit.ryy(theta=theta_ij, qubit1=qubit_i, qubit2=qubit_j)
                qcircuit.add_rxx(angle=theta_ij, qubit_1=qubit_i, qubit_2=qubit_j)
                qcircuit.add_ryy(angle=theta_ij, qubit_1=qubit_i, qubit_2=qubit_j)
                counter += 1

            # Z terms
            if self.with_z_phase:
                for qubit_i in range(self.n_qubits):
                    #qcircuit.rz(phi=next(__angles__), qubit=qubit_i)
                    theta_i = torch_angles[counter]
                    qcircuit.add_rz(angle=theta_i, target_qubit=qubit_i)
                    counter += 1

        # Get cost hamiltonian
        #H_c = get_qiskit_H(Q=self.Q)
        H_c = torch.tensor(np.array(Operator(get_qiskit_H(Q=self.Q))),
                           dtype=torch.cfloat,
                           requires_grad=True)
        U_theta = qcircuit.get_circuit_unitary()
        psi_final = torch.matmul(U_theta, psi_0)
        cost = torch.real(torch.dot(torch.conj(psi_final), torch.matmul(H_c, psi_final)))

        # Backpropagation to calculate gradients
        cost.backward()

        print(torch_angles.grad, psi_0.grad)

        # Extracting and returning the gradient as a numpy array
        angle_gradients = torch_angles.grad.numpy()

        """# Parameter values list (multiplying w. 2 as values in set circuit = 2*angle)
        param_values = [[theta for theta in angles]]

        # Define the gradient
        gradient = FiniteDiffEstimatorGradient(Estimator(), epsilon=0.001)

        # Evaluate the gradient of the circuits using parameter shift gradients
        pse_grad_result = gradient.run(circuits=[qcircuit],
                                       observables=[H_c],
                                       parameters=[params],
                                       parameter_values=param_values).result().gradients"""

        return angle_gradients

    def get_state_probabilities(self, flip_states: bool = True) -> Dict:
        counts = self.counts
        if flip_states:
            return {bitstring[::-1]: probability for bitstring, probability in counts.items()}
        return {bitstring: probability for bitstring, probability in counts.items()}
