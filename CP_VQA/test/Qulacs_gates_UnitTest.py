from typing import List, Tuple
import pytest
import numpy as np
import qulacs
from qiskit import QuantumCircuit, Aer, execute
from src.custom_gates.custom_qulacs_gates import RXX, RYY

# TODO: Find out why Qulacs single qubit rotation is opposite sign of qiskit??

######################################################################################################
#                                 TEST CASE GENERATOR FUNCTIONS                                      #
######################################################################################################

__N__ = 6


def generate_rxx_test_cases(nr_rng_trials: int = 10) -> List[Tuple[np.ndarray, np.ndarray]]:
    test_cases = []
    for seed in range(nr_rng_trials):
        np.random.seed(seed + 2)
        theta = np.random.uniform(-2 * np.pi, 2 * np.pi)
        pair = np.random.choice(__N__, 2, replace=False)

        # --- Qulacs --- #
        qulacs_state_vector = qulacs.QuantumState(__N__)
        qulacs_circuit = qulacs.QuantumCircuit(__N__)
        RXX(circuit=qulacs_circuit, angle=theta, qubit_1=pair[0], qubit_2=pair[1])
        qulacs_circuit.update_quantum_state(qulacs_state_vector)
        qulacs_state_vector = qulacs_state_vector.get_vector()

        # --- Qiskit --- #
        qiskit_circuit = QuantumCircuit(__N__)
        qiskit_circuit.rxx(theta=theta, qubit1=pair[0], qubit2=pair[1])
        qiskit_state_vector = np.array(execute(qiskit_circuit,
                                               Aer.get_backend('statevector_simulator')).result().get_statevector())

        test_cases.append((qiskit_state_vector, qulacs_state_vector))

    return test_cases


def generate_ryy_test_cases(nr_rng_trials: int = 10) -> List[Tuple[np.ndarray, np.ndarray]]:
    test_cases = []
    for seed in range(nr_rng_trials):
        np.random.seed(seed)
        theta = np.random.uniform(-2 * np.pi, 2 * np.pi)
        pair = np.random.choice(__N__, 2, replace=False)

        # --- Qulacs --- #
        qulacs_state_vector = qulacs.QuantumState(__N__)
        qulacs_circuit = qulacs.QuantumCircuit(__N__)
        RYY(circuit=qulacs_circuit, angle=theta, qubit_1=pair[0], qubit_2=pair[1])
        qulacs_circuit.update_quantum_state(qulacs_state_vector)
        qulacs_state_vector = qulacs_state_vector.get_vector()

        # --- Qiskit --- #
        qiskit_circuit = QuantumCircuit(__N__)
        qiskit_circuit.ryy(theta=theta, qubit1=pair[0], qubit2=pair[1])
        qiskit_state_vector = np.array(execute(qiskit_circuit,
                                               Aer.get_backend('statevector_simulator')).result().get_statevector())

        test_cases.append((qiskit_state_vector, qulacs_state_vector))

    return test_cases


def generate_rz_test_cases(nr_rng_trials: int = 10) -> List[Tuple[np.ndarray, np.ndarray]]:
    test_cases = []
    for seed in range(nr_rng_trials):
        np.random.seed(seed)
        theta = np.random.uniform(-2 * np.pi, 2 * np.pi)
        idx = np.random.choice(__N__, 1, replace=False)

        # --- Qulacs --- #
        qulacs_state_vector = qulacs.QuantumState(__N__)
        qulacs_circuit = qulacs.QuantumCircuit(__N__)
        qulacs_circuit.add_gate(qulacs.gate.RZ(index=idx, angle=-theta))
        qulacs_circuit.update_quantum_state(qulacs_state_vector)
        qulacs_state_vector = qulacs_state_vector.get_vector()

        # --- Qiskit --- #
        qiskit_circuit = QuantumCircuit(__N__)
        qiskit_circuit.rz(phi=theta, qubit=idx)
        qiskit_state_vector = np.array(execute(qiskit_circuit,
                                               Aer.get_backend('statevector_simulator')).result().get_statevector())

        test_cases.append((qiskit_state_vector, qulacs_state_vector))

    return test_cases


#############################################################################
#                                 TESTING                                   #
#############################################################################

N_RNG_TRIALS = 10
rxx_test_cases = generate_rxx_test_cases(nr_rng_trials=N_RNG_TRIALS)
ryy_test_cases = generate_ryy_test_cases(nr_rng_trials=N_RNG_TRIALS)
rz_test_cases = generate_rz_test_cases(nr_rng_trials=N_RNG_TRIALS)


@pytest.mark.parametrize('qiskit_state_vector, qulacs_state_vector', rxx_test_cases, )
def test_rxx_gate(qiskit_state_vector: np.ndarray,
                  qulacs_state_vector: np.ndarray):
    # Comparing
    assert np.allclose(qiskit_state_vector, qulacs_state_vector)


@pytest.mark.parametrize('qiskit_state_vector, qulacs_state_vector', ryy_test_cases, )
def test_ryy_gate(qiskit_state_vector: np.ndarray,
                  qulacs_state_vector: np.ndarray):
    # Comparing
    assert np.allclose(qiskit_state_vector, qulacs_state_vector)


@pytest.mark.parametrize('qiskit_state_vector, qulacs_state_vector', rz_test_cases, )
def test_rz_gate(qiskit_state_vector: np.ndarray,
                 qulacs_state_vector: np.ndarray):
    # Comparing
    assert np.allclose(qiskit_state_vector, qulacs_state_vector)
