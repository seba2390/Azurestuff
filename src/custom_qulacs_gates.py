from qulacs.gate import H, CNOT, RZ, RX
import numpy as np


def RXX(circuit, angle, qubit_1, qubit_2):
    circuit.add_gate(H(index=qubit_1))
    circuit.add_gate(H(index=qubit_2))
    circuit.add_gate(CNOT(control=qubit_1, target=qubit_2))
    circuit.add_gate(RZ(index=qubit_2, angle=-angle))
    circuit.add_gate(CNOT(control=qubit_1, target=qubit_2))
    circuit.add_gate(H(index=qubit_1))
    circuit.add_gate(H(index=qubit_2))


def RYY(circuit, angle, qubit_1, qubit_2):
    circuit.add_gate(RX(index=qubit_1, angle=-np.pi / 2))
    circuit.add_gate(RX(index=qubit_2, angle=-np.pi / 2))
    circuit.add_gate(CNOT(control=qubit_1, target=qubit_2))
    circuit.add_gate(RZ(index=qubit_2, angle=-angle))
    circuit.add_gate(CNOT(control=qubit_1, target=qubit_2))
    circuit.add_gate(RX(index=qubit_1, angle=np.pi / 2))
    circuit.add_gate(RX(index=qubit_2, angle=np.pi / 2))
