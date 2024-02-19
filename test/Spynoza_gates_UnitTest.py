from typing import *
import random

import pytest
import numpy as np
import qiskit
from qiskit import Aer, execute
from spynoza import QuantumCircuit, QuantumRegister, PyState

from src.custom_spynoza_gates import RXX, RYY, RZZ, RZ, RX


def PyState_to_NumpyArray(state: PyState) -> np.ndarray:
    return np.array([state.__getitem__(i)[0] + 1j * state.__getitem__(i)[1] for i in range(state.__len__())],
                    dtype=np.complex64)


######################################################################################################
#                                 TEST CASE GENERATOR FUNCTIONS                                      #
######################################################################################################

__N_QUBITS__ = 8


def generate_tests(n_rng_trials: int, gate_name: str) -> List[Tuple[np.ndarray[complex], np.ndarray[complex]]]:
    test_cases = []
    for trial in range(n_rng_trials):
        np.random.seed(trial)
        theta = np.random.uniform(-2 * np.pi, 2 * np.pi)
        pair = np.random.choice(__N_QUBITS__, 2, replace=False)

        # --- Spynoza --- #
        register = QuantumRegister(__N_QUBITS__)
        circuit = QuantumCircuit(register)
        if gate_name == 'RXX':
            RXX(circuit=circuit, angle=theta, qubit_1=pair[0], qubit_2=pair[1])
        elif gate_name == 'RYY':
            RYY(circuit=circuit, angle=theta, qubit_1=pair[0], qubit_2=pair[1])
        elif gate_name == 'RZZ':
            RZZ(circuit=circuit, angle=theta, qubit_1=pair[0], qubit_2=pair[1])
        elif gate_name == 'RZ':
            RZ(circuit=circuit, angle=theta, qubit=pair[0])
        elif gate_name == 'RX':
            RX(circuit=circuit, angle=theta, qubit=pair[0])
        circuit.execute()
        spynoza_state_vector = PyState_to_NumpyArray(circuit.state_vector)

        # --- Qiskit --- #
        qiskit_circuit = qiskit.QuantumCircuit(__N_QUBITS__)
        if gate_name == 'RXX':
            qiskit_circuit.rxx(theta=theta, qubit1=pair[0], qubit2=pair[1])
        elif gate_name == 'RYY':
            qiskit_circuit.ryy(theta=theta, qubit1=pair[0], qubit2=pair[1])
        elif gate_name == 'RZZ':
            qiskit_circuit.rzz(theta=theta, qubit1=pair[0], qubit2=pair[1])
        elif gate_name == 'RZ':
            qiskit_circuit.rz(phi=theta, qubit=pair[0])
        elif gate_name == 'RX':
            qiskit_circuit.rx(theta=theta, qubit=pair[0])
        qiskit_state_vector = np.array(execute(qiskit_circuit,
                                               Aer.get_backend('statevector_simulator')).result().get_statevector())

        test_cases.append((qiskit_state_vector, spynoza_state_vector))

    return test_cases


#############################################################################
#                                 TESTING                                   #
#############################################################################

N_RNG_TRIALS = 10
rxx_test_cases = generate_tests(n_rng_trials=N_RNG_TRIALS, gate_name='RXX')
ryy_test_cases = generate_tests(n_rng_trials=N_RNG_TRIALS, gate_name='RYY')
rzz_test_cases = generate_tests(n_rng_trials=N_RNG_TRIALS, gate_name='RZZ')

rx_test_cases = generate_tests(n_rng_trials=N_RNG_TRIALS, gate_name='RX')
rz_test_cases = generate_tests(n_rng_trials=N_RNG_TRIALS, gate_name='RZ')


@pytest.mark.parametrize('qiskit_state_vector, spynoza_state_vector', rxx_test_cases, )
def test_rxx_gate(qiskit_state_vector: np.ndarray,
                  spynoza_state_vector: np.ndarray):
    # Comparing
    assert np.allclose(qiskit_state_vector, spynoza_state_vector)


@pytest.mark.parametrize('qiskit_state_vector, spynoza_state_vector', ryy_test_cases, )
def test_ryy_gate(qiskit_state_vector: np.ndarray,
                  spynoza_state_vector: np.ndarray):
    # Comparing
    assert np.allclose(qiskit_state_vector, spynoza_state_vector)


@pytest.mark.parametrize('qiskit_state_vector, spynoza_state_vector', rzz_test_cases, )
def test_rzz_gate(qiskit_state_vector: np.ndarray,
                  spynoza_state_vector: np.ndarray):
    # Comparing
    assert np.allclose(qiskit_state_vector, spynoza_state_vector)


@pytest.mark.parametrize('qiskit_state_vector, spynoza_state_vector', rx_test_cases, )
def test_rx_gate(qiskit_state_vector: np.ndarray,
                  spynoza_state_vector: np.ndarray):
    # Comparing
    assert np.allclose(qiskit_state_vector, spynoza_state_vector)


@pytest.mark.parametrize('qiskit_state_vector, spynoza_state_vector', rz_test_cases, )
def test_rz_gate(qiskit_state_vector: np.ndarray,
                  spynoza_state_vector: np.ndarray):
    # Comparing
    assert np.allclose(qiskit_state_vector, spynoza_state_vector)
