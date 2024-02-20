import pennylane as qml
import numpy as np


def RZ(angle: float, qubit: int) -> None:
    qml.RZ(phi=angle, wires=qubit)


def RZZ(angle: float, qubit_1: int, qubit_2: int) -> None:
    qml.CNOT(wires=[qubit_1, qubit_2])
    qml.RZ(phi=angle, wires=qubit_2)
    qml.CNOT(wires=[qubit_1, qubit_2])


def RXX(qubit_1: int, qubit_2: int, angle: float):
    qml.Hadamard(wires=qubit_1)
    qml.Hadamard(wires=qubit_2)
    RZZ(qubit_1=qubit_1, qubit_2=qubit_2, angle=angle)
    qml.Hadamard(wires=qubit_1)
    qml.Hadamard(wires=qubit_2)


def RYY(qubit_1: int, qubit_2: int, angle: float):
    RX(qubit=qubit_1, angle=np.pi / 2)
    RX(qubit=qubit_2, angle=np.pi / 2)
    RZZ(qubit_1=qubit_1, qubit_2=qubit_2, angle=angle)
    RX(qubit=qubit_1, angle=-np.pi / 2)
    RX(qubit=qubit_2, angle=-np.pi / 2)


def RX(qubit: int, angle: float):
    qml.RX(phi=angle, wires=qubit)
