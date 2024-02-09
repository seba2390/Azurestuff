from typing import List, Tuple
import pytest
import numpy as np
import cirq
from scipy.linalg import expm
from custom_cirq_gates import RXX, RYY


######################################################################################################
#                                 TEST CASE GENERATOR FUNCTIONS                                      #
######################################################################################################


def generate_rxx_test_cases(nr_rng_trials: int = 10) -> List[Tuple[np.ndarray, np.ndarray]]:
    test_cases = []
    qubits = [cirq.NamedQubit('q_0'), cirq.NamedQubit('q_1')]
    for seed in range(nr_rng_trials):
        theta = np.random.uniform(-2 * np.pi, 2 * np.pi)

        circuit_1 = cirq.Circuit()
        circuit_1.append(cirq.X(qubits[0]))
        circuit_1.append(cirq.X(qubits[1]))
        U_1 = expm(-1j * theta / 2 * circuit_1.unitary())

        circuit_2 = cirq.Circuit()
        circuit_2.append(RXX(theta).on(qubits[0], qubits[1]))
        U_2 = circuit_2.unitary()

        test_cases.append((U_1, U_2))
    return test_cases


def generate_ryy_test_cases(nr_rng_trials: int = 10) -> List[Tuple[np.ndarray, np.ndarray]]:
    test_cases = []
    qubits = [cirq.NamedQubit('q_0'), cirq.NamedQubit('q_1')]
    for seed in range(nr_rng_trials):
        theta = np.random.uniform(-2 * np.pi, 2 * np.pi)

        circuit_1 = cirq.Circuit()
        circuit_1.append(cirq.Y(qubits[0]))
        circuit_1.append(cirq.Y(qubits[1]))
        U_1 = expm(-1j * theta / 2 * circuit_1.unitary())

        circuit_2 = cirq.Circuit()
        circuit_2.append(RYY(theta).on(qubits[0], qubits[1]))
        U_2 = circuit_2.unitary()

        test_cases.append((U_1, U_2))
    return test_cases


#############################################################################
#                                 TESTING                                   #
#############################################################################

N_RNG_TRIALS = 10
rxx_test_cases = generate_rxx_test_cases(nr_rng_trials=N_RNG_TRIALS)
ryy_test_cases = generate_ryy_test_cases(nr_rng_trials=N_RNG_TRIALS)


@pytest.mark.parametrize('matrix_1, matrix_2', rxx_test_cases, )
def test_rxx_gate(matrix_1: np.ndarray,
                  matrix_2: np.ndarray):
    # Comparing
    assert np.allclose(matrix_1, matrix_2)


@pytest.mark.parametrize('matrix_1, matrix_2', ryy_test_cases, )
def test_ryy_gate(matrix_1: np.ndarray,
                  matrix_2: np.ndarray):
    # Comparing
    assert np.allclose(matrix_1, matrix_2)
