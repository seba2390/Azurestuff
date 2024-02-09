import cirq
import numpy as np


class RXX(cirq.Gate):
    def __init__(self, theta):
        super(RXX, self)
        self.theta = theta

    def _num_qubits_(self):
        return 2

    def _decompose_(self, qubits):
        qubit_1, qubit_2 = qubits
        yield cirq.H(qubit_1)
        yield cirq.H(qubit_2)
        yield cirq.CNOT(qubit_1, qubit_2)
        yield (cirq.rz(self.theta))(qubit_2)
        yield cirq.CNOT(qubit_1, qubit_2)
        yield cirq.H(qubit_1)
        yield cirq.H(qubit_2)

    def _circuit_diagram_info_(self, args):
        return [f"RXX({self.theta})"] * self.num_qubits()


class RYY(cirq.Gate):
    def __init__(self, theta):
        super(RYY, self)
        self.theta = theta

    def _num_qubits_(self):
        return 2

    def _decompose_(self, qubits):
        qubit_1, qubit_2 = qubits
        yield (cirq.rx(np.pi / 2))(qubit_1)
        yield (cirq.rx(np.pi / 2))(qubit_2)
        yield cirq.CNOT(qubit_1, qubit_2)
        yield (cirq.rz(self.theta))(qubit_2)
        yield cirq.CNOT(qubit_1, qubit_2)
        yield (cirq.rx(-np.pi / 2))(qubit_1)
        yield (cirq.rx(-np.pi / 2))(qubit_2)

    def _circuit_diagram_info_(self, args):
        return [f"RYY({self.theta})"] * self.num_qubits()
