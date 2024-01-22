from typing import List, Dict, Union
import random
from collections import Counter

from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import Parameter
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import Operator
import numpy as np
from scipy.linalg import expm
import torch

from src.Tools import get_qiskit_H
from src.Tools import qubo_cost, string_to_array, create_operator, operator_expectation, get_generator
from src.Grid import Grid
from src.Tools import get_full_hamiltonian
from src.Chain import Chain


def create_Rxx_matrix(n_qubits: int, qubit_1: int, qubit_2: int, angle: torch.Tensor) -> torch.Tensor:
    # Check if qubit indices are within the range
    if qubit_1 >= n_qubits or qubit_2 >= n_qubits or qubit_1 < 0 or qubit_2 < 0:
        raise ValueError("Qubit indices are out of bounds.")
    elif qubit_1 == qubit_2:
        raise ValueError("Qubit indices are equal (the should be different)")
    # Pauli X gate
    X = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128)
    # Identity matrix for other qubits
    I = torch.eye(2, dtype=torch.complex128)
    # Create the full tensor product for the R_xx gate
    if qubit_1 == 0 or qubit_2 == 0:
        gate = X
    else:
        gate = I
    for i in range(1, n_qubits):
        if i == qubit_1 or i == qubit_2:
            gate = torch.kron(gate, X)
        else:
            gate = torch.kron(gate, I)
    # R_xx gate matrix
    R_xx = torch.matrix_exp(-1j * angle / 2 * gate)
    return R_xx


def create_Rz_matrix(n_qubits: int, qubit: int, angle: torch.Tensor) -> torch.Tensor:
    # Check if qubit indices are within the range
    if qubit >= n_qubits or qubit < 0:
        raise ValueError("Qubit indices are out of bounds.")
    # Pauli Z gate
    Z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128)
    # Identity matrix for other qubits
    I = torch.eye(2, dtype=torch.complex128)
    # Create the full tensor product for the R_z gate
    if qubit:
        gate = Z
    else:
        gate = I
    for i in range(1, n_qubits):
        if i == qubit:
            gate = torch.kron(gate, Z)
        else:
            gate = torch.kron(gate, I)
    # R_z gate matrix
    R_z = torch.matrix_exp(-1j * angle / 2 * gate)
    return R_z


def create_Ryy_matrix(n_qubits: int, qubit_1: int, qubit_2: int, angle: torch.Tensor) -> torch.Tensor:
    # Check if qubit indices are within the range
    if qubit_1 >= n_qubits or qubit_2 >= n_qubits or qubit_1 < 0 or qubit_2 < 0:
        raise ValueError("Qubit indices are out of bounds.")
    elif qubit_1 == qubit_2:
        raise ValueError("Qubit indices are equal (the should be different)")
    # Pauli Y gate
    Y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex128)
    # Identity matrix for other qubits
    I = torch.eye(2, dtype=torch.complex128)
    # Create the full tensor product for the R_yy gate
    if qubit_1 == 0 or qubit_2 == 0:
        gate = Y
    else:
        gate = I
    for i in range(1, n_qubits):
        if i == qubit_1 or i == qubit_2:
            gate = torch.kron(gate, Y)
        else:
            gate = torch.kron(gate, I)
    # R_yy gate matrix
    R_yy = torch.matrix_exp(-1j * angle / 2 * gate)
    return R_yy


def create_x_matrix(n_qubits, qubit) -> torch.Tensor:
    # Check if qubit indices are within the range
    if qubit >= n_qubits or qubit < 0:
        raise ValueError("Qubit indices are out of bounds.")
    # Pauli X gate
    X = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128)
    # Identity matrix for other qubits
    I = torch.eye(2, dtype=torch.complex128)
    # Create the full tensor product for the x gate
    if qubit == 0:
        gate = X
    else:
        gate = I
    for i in range(1, n_qubits):
        if i == qubit:
            gate = torch.kron(gate, X)
        else:
            gate = torch.kron(gate, I)
    return gate


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
                 approximate_hamiltonian: bool = True,
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
        self.approximate_hamiltonian = approximate_hamiltonian
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
                H = get_full_hamiltonian(indices=self.qubit_indices, angles=angles, N_qubits=self.n_qubits, with_z_phase=self.with_z_phase)
                U_H = PauliEvolutionGate(H, time=1.0)
                qcircuit.append(U_H, list(set([q[0] for q in self.qubit_indices] + [q[1] for q in self.qubit_indices])))
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
        #H_c = np.array(Operator(get_qiskit_H(Q=self.Q)))
        #state_vector = np.array(execute(circuit, self.simulator).result().get_statevector()).flatten()
        #return float(np.real(np.dot(state_vector.conj(), np.dot(H_c, state_vector))))

    def get_gradient(self, angles) -> np.ndarray:
        """ Using parameter shift rule to calculate exact derivatives"""
        torch_angles = torch.tensor(angles,
                                    requires_grad=True)

        # Defining circuit

        psi_0 = torch.tensor([1.0] + [0.0 for _ in range(2 ** self.n_qubits - 1)],
                             dtype=torch.complex128,
                             requires_grad=True)

        # Setting 'k' qubits to |1>
        for qubit_index in self.initialization_strategy:
            X_i = create_x_matrix(n_qubits=self.n_qubits, qubit=qubit_index)
            psi_0 = torch.matmul(X_i, psi_0)

        counter = 0
        for layer in range(self.layers):
            # XX+YY terms
            for (qubit_i, qubit_j) in self.qubit_indices:
                theta_ij = torch_angles[counter]
                rxx = create_Rxx_matrix(n_qubits=self.n_qubits, qubit_1=qubit_i, qubit_2=qubit_j, angle=theta_ij)
                ryy = create_Ryy_matrix(n_qubits=self.n_qubits, qubit_1=qubit_i, qubit_2=qubit_j, angle=theta_ij)
                psi_0 = torch.matmul(ryy, torch.matmul(rxx, psi_0))
                counter += 1

            # Z terms
            if self.with_z_phase:
                for qubit_i in range(self.n_qubits):
                    theta_i = torch_angles[counter]
                    rz = create_Rz_matrix(n_qubits=self.n_qubits, qubit=qubit_i, angle=theta_i)
                    psi_0 = torch.matmul(rz,psi_0)
                    counter += 1

        # Get cost hamiltonian
        H_c = torch.tensor(np.array(Operator(get_qiskit_H(Q=self.Q))),
                           dtype=torch.complex128,
                           requires_grad=True)
        c = torch.real(torch.dot(torch.conj(psi_0), torch.matmul(H_c, psi_0)))
        c.backward()

        # Extracting and returning the gradient as a numpy array
        angle_gradients = torch_angles.grad.numpy()

        return angle_gradients

    def get_state_probabilities(self, flip_states: bool = True) -> Dict:
        counts = self.counts
        if flip_states:
            return {bitstring[::-1]: probability for bitstring, probability in counts.items()}
        return {bitstring: probability for bitstring, probability in counts.items()}
