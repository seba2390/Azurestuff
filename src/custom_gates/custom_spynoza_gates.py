from spynoza import QuantumCircuit
import numpy as np


def RXX(circuit: QuantumCircuit, qubit_1: int, qubit_2: int, angle: float):
    circuit.h(qubit_1)
    circuit.h(qubit_2)
    RZZ(circuit=circuit, qubit_1=qubit_1, qubit_2=qubit_2, angle=angle)
    circuit.h(qubit_1)
    circuit.h(qubit_2)


def RYY(circuit: QuantumCircuit, qubit_1: int, qubit_2: int, angle: float):
    circuit.rx(np.pi / 2, qubit_1)
    circuit.rx(np.pi / 2, qubit_2)
    RZZ(circuit=circuit, qubit_1=qubit_1, qubit_2=qubit_2, angle=angle)
    circuit.rx(-np.pi / 2, qubit_1)
    circuit.rx(-np.pi / 2, qubit_2)


def RZZ(circuit: QuantumCircuit, qubit_1: int, qubit_2: int, angle: float):
    circuit.cx(qubit_1, qubit_2)
    RZ(circuit=circuit, angle=angle, qubit=qubit_2)
    circuit.cx(qubit_1, qubit_2)


def RZ(circuit: QuantumCircuit, qubit: int, angle: float):
    circuit.rz(angle, qubit)


def RX(circuit: QuantumCircuit, qubit: int, angle: float):
    circuit.rx(angle, qubit)
