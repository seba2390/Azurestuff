from typing import Union

import qulacs
from qulacs.gate import H, CNOT, RZ, RX, PauliRotation, ParametricPauliRotation, ParametricRX, ParametricRZ
import numpy as np


# TODO: Find out why Qulacs single qubit rotation is opposite sign of qiskit??

def RXX(circuit: qulacs.QuantumCircuit,
        angle: float,
        qubit_1: int,
        qubit_2: int,
        use_native: bool = False):
    if use_native:
        angle *= -1
        target_list = [qubit_2, qubit_1]
        pauli_index = [1, 1]  # 1:X , 2:Y, 3:Z
        circuit.add_gate(PauliRotation(index_list=target_list,
                                       pauli_ids=pauli_index,
                                       angle=2 * angle))
    else:
        circuit.add_gate(H(index=qubit_1))
        circuit.add_gate(H(index=qubit_2))
        circuit.add_gate(CNOT(control=qubit_1, target=qubit_2))
        circuit.add_gate(RZ(index=qubit_2, angle=-angle))
        circuit.add_gate(CNOT(control=qubit_1, target=qubit_2))
        circuit.add_gate(H(index=qubit_1))
        circuit.add_gate(H(index=qubit_2))


def RYY(circuit: qulacs.QuantumCircuit,
        angle: float,
        qubit_1: int,
        qubit_2: int,
        use_native: bool = False):
    if use_native:
        angle *= -1
        target_list = [qubit_2, qubit_1]
        pauli_index = [2, 2]  # 1:X , 2:Y, 3:Z
        circuit.add_gate(PauliRotation(index_list=target_list,
                                       pauli_ids=pauli_index,
                                       angle=2 * angle))
    else:
        circuit.add_gate(RX(index=qubit_1, angle=-np.pi / 2))
        circuit.add_gate(RX(index=qubit_2, angle=-np.pi / 2))
        circuit.add_gate(CNOT(control=qubit_1, target=qubit_2))
        circuit.add_gate(RZ(index=qubit_2, angle=-angle))
        circuit.add_gate(CNOT(control=qubit_1, target=qubit_2))
        circuit.add_gate(RX(index=qubit_1, angle=np.pi / 2))
        circuit.add_gate(RX(index=qubit_2, angle=np.pi / 2))


def parametric_RXX(circuit: qulacs.ParametricQuantumCircuit,
                   angle: float,
                   qubit_1: int,
                   qubit_2: int,
                   use_native: bool = False):
    if use_native:
        target_list = [qubit_2, qubit_1]
        pauli_index = [1, 1]  # 1:X , 2:Y, 3:Z
        circuit.add_parametric_multi_Pauli_rotation_gate(index_list=target_list,
                                                         pauli_ids=pauli_index,
                                                         angle=-angle)
    else:
        circuit.add_gate(H(index=qubit_1))
        circuit.add_gate(H(index=qubit_2))
        circuit.add_gate(CNOT(control=qubit_1, target=qubit_2))
        circuit.add_parametric_RZ_gate(index=qubit_2, angle=-angle)
        circuit.add_gate(CNOT(control=qubit_1, target=qubit_2))
        circuit.add_gate(H(index=qubit_1))
        circuit.add_gate(H(index=qubit_2))


def parametric_RYY(circuit: qulacs.ParametricQuantumCircuit,
                   angle: float,
                   qubit_1: int,
                   qubit_2: int,
                   use_native: bool = False):
    if use_native:
        target_list = [qubit_2, qubit_1]
        pauli_index = [2, 2]  # 1:X , 2:Y, 3:Z
        circuit.add_parametric_multi_Pauli_rotation_gate(index_list=target_list,
                                                         pauli_ids=pauli_index,
                                                         angle=-angle)
    else:
        circuit.add_gate(RX(index=qubit_1, angle=-np.pi / 2))
        circuit.add_gate(RX(index=qubit_2, angle=-np.pi / 2))
        circuit.add_gate(CNOT(control=qubit_1, target=qubit_2))
        circuit.add_parametric_RZ_gate(index=qubit_2, angle=-angle)
        circuit.add_gate(CNOT(control=qubit_1, target=qubit_2))
        circuit.add_gate(RX(index=qubit_1, angle=np.pi / 2))
        circuit.add_gate(RX(index=qubit_2, angle=np.pi / 2))
