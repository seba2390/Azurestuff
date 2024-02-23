import cirq
import numpy as np


def RXX(circuit, angle, qubit_1, qubit_2):
    circuit.append(cirq.H(qubit_1))
    circuit.append(cirq.H(qubit_2))
    circuit.append(cirq.CNOT(qubit_1, qubit_2))
    circuit.append((cirq.rz(angle))(qubit_2))
    circuit.append(cirq.CNOT(qubit_1, qubit_2))
    circuit.append(cirq.H(qubit_1))
    circuit.append(cirq.H(qubit_2))


def RYY(circuit, angle, qubit_1, qubit_2):
    circuit.append((cirq.rx(np.pi / 2))(qubit_1))
    circuit.append((cirq.rx(np.pi / 2))(qubit_2))
    circuit.append(cirq.CNOT(qubit_1, qubit_2))
    circuit.append((cirq.rz(angle))(qubit_2))
    circuit.append(cirq.CNOT(qubit_1, qubit_2))
    circuit.append((cirq.rx(-np.pi / 2))(qubit_1))
    circuit.append((cirq.rx(-np.pi / 2))(qubit_2))


def RZZ(circuit, angle, qubit_1, qubit_2):
    circuit.append(cirq.CNOT(qubit_1, qubit_2))
    circuit.append((cirq.rz(angle))(qubit_2))
    circuit.append(cirq.CNOT(qubit_1, qubit_2))


def RX(circuit, angle, qubit):
    circuit.append((cirq.rx(angle))(qubit))


def RZ(circuit, angle, qubit):
    circuit.append((cirq.rz(angle))(qubit))
