from typing import *

import torch
import numpy as np


def generate_string_representation(gate_name: str,
                                   qubit_i: int,
                                   qubit_j: int,
                                   N: int):
    if not 0 <= qubit_i < N or not 0 <= qubit_j < N:
        raise ValueError("Qubit indices are out of bounds..")
    if gate_name not in ['X', 'Y', 'Z', 'I']:
        raise ValueError("unknown gate name..")
    gates = ['I' for qubit in range(N)]
    gates[qubit_i] = gate_name
    gates[qubit_j] = gate_name
    return ''.join(gate for gate in gates)


def generate_string_representation_single(gate_name: str,
                                          qubit_i: int,
                                          N: int):
    if not 0 <= qubit_i < N:
        raise ValueError("Qubit indices are out of bounds..")
    if gate_name not in ['X', 'Y', 'Z', 'I']:
        raise ValueError("unknown gate name..")
    gates = ['I' for qubit in range(N)]
    gates[qubit_i] = gate_name
    return ''.join(gate for gate in gates)


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

    str_repr = generate_string_representation(gate_name='X',
                                              qubit_i=qubit_1,
                                              qubit_j=qubit_2,
                                              N=n_qubits)
    gate_map = {'X': X, 'I': I}
    # notice reversing str_repr to match qiskit convention
    gates = [gate_map[gate] for gate in str_repr[::-1]]
    generator = gates[0]
    for gate in gates[1:]:
        generator = torch.kron(generator, gate)
    # R_xx gate matrix
    R_xx = torch.matrix_exp(-1j * angle / 2 * generator)
    return R_xx


def create_Rz_matrix(n_qubits: int, qubit: int, angle: torch.Tensor) -> torch.Tensor:
    # Check if qubit indices are within the range
    if qubit >= n_qubits or qubit < 0:
        raise ValueError("Qubit indices are out of bounds.")
    # Pauli Z gate
    Z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128)
    # Identity matrix for other qubits
    I = torch.eye(2, dtype=torch.complex128)
    str_repr = generate_string_representation_single(gate_name='Z',
                                                     qubit_i=qubit,
                                                     N=n_qubits)
    gate_map = {'Z': Z, 'I': I}
    # notice reversing str_repr to match qiskit convention
    gates = [gate_map[gate] for gate in str_repr[::-1]]
    generator = gates[0]
    for gate in gates[1:]:
        generator = torch.kron(generator, gate)
    R_z = torch.matrix_exp(-1j * angle / 2 * generator)
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

    str_repr = generate_string_representation(gate_name='Y',
                                              qubit_i=qubit_1,
                                              qubit_j=qubit_2,
                                              N=n_qubits)
    gate_map = {'Y': Y, 'I': I}
    # notice reversing str_repr to match qiskit convention
    gates = [gate_map[gate] for gate in str_repr[::-1]]
    generator = gates[0]
    for gate in gates[1:]:
        generator = torch.kron(generator, gate)
    # R_yy gate matrix
    R_yy = torch.matrix_exp(-1j * angle / 2 * generator)
    return R_yy


def get_full_torch_hamiltonian(indices: List[Tuple[int, int]], angles: torch.Tensor, N_qubits: int,
                               with_z_phase: bool = False):
    terms = []
    X = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128)
    Y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex128)
    Z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128)
    I = torch.eye(2, dtype=torch.complex128)

    gate_map = {'X': X, 'Y': Y, 'Z': Z, 'I': I}

    Normal = True
    for (qubit_i, qubit_j), theta_ij in zip(indices, angles[:len(indices)]):
        x_str = generate_string_representation(gate_name='X',
                                               qubit_i=qubit_i,
                                               qubit_j=qubit_j,
                                               N=N_qubits)
        y_str = generate_string_representation(gate_name='Y',
                                               qubit_i=qubit_i,
                                               qubit_j=qubit_j,
                                               N=N_qubits)
        # notice reversing str_repr to match qiskit convention
        x_gates = [gate_map[gate] for gate in x_str[::-1]]
        # notice reversing str_repr to match qiskit convention
        y_gates = [gate_map[gate] for gate in y_str[::-1]]
        H_xx, H_yy = x_gates[0], y_gates[0]
        for x_gate, y_gate in zip(x_gates[1:], y_gates[1:]):
            H_xx = torch.kron(H_xx, x_gate)
            H_yy = torch.kron(H_yy, y_gate)

        H_ij = theta_ij * (H_xx + H_yy)
        terms.append(H_ij)
    if with_z_phase:
        for qubit_i, theta_i in zip(list(range(N_qubits)), angles[len(angles):]):
            z_str = generate_string_representation_single(gate_name='Z',
                                                          qubit_i=qubit_i,
                                                          N=N_qubits)
            # notice reversing str_repr to match qiskit convention
            z_gates = [gate_map[gate] for gate in z_str[::-1]]
            H_z = z_gates[0]
            for gate in z_gates[1:]:
                H_z = torch.kron(H_z, gate)

            H_i = theta_i * H_z
            terms.append(H_i)
    H = terms[0]
    for term in terms[1:]:
        H += term
    return H
