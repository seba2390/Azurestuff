from qulacs.gate import H, CNOT, RZ, RX, PauliRotation
import numpy as np


def RXX(circuit, angle, qubit_1, qubit_2, use_native: bool = True):
    if use_native:
        target_list = [qubit_1, qubit_2]
        pauli_index = [1, 1]  # 1:X , 2:Y, 3:Z
        circuit.add_gate(PauliRotation(target_list, pauli_index, -angle))
    circuit.add_gate(H(index=qubit_1))
    circuit.add_gate(H(index=qubit_2))
    circuit.add_gate(CNOT(control=qubit_1, target=qubit_2))
    circuit.add_gate(RZ(index=qubit_2, angle=-angle))
    circuit.add_gate(CNOT(control=qubit_1, target=qubit_2))
    circuit.add_gate(H(index=qubit_1))
    circuit.add_gate(H(index=qubit_2))


def RYY(circuit, angle, qubit_1, qubit_2, use_native: bool = True):
    if use_native:
        target_list = [qubit_1, qubit_2]
        pauli_index = [2, 2]  # 1:X , 2:Y, 3:Z
        circuit.add_gate(PauliRotation(target_list, pauli_index, -angle))
    circuit.add_gate(RX(index=qubit_1, angle=-np.pi / 2))
    circuit.add_gate(RX(index=qubit_2, angle=-np.pi / 2))
    circuit.add_gate(CNOT(control=qubit_1, target=qubit_2))
    circuit.add_gate(RZ(index=qubit_2, angle=-angle))
    circuit.add_gate(CNOT(control=qubit_1, target=qubit_2))
    circuit.add_gate(RX(index=qubit_1, angle=np.pi / 2))
    circuit.add_gate(RX(index=qubit_2, angle=np.pi / 2))
