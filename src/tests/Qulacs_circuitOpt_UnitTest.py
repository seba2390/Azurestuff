from typing import List, Tuple
import pytest
import numpy as np
import qulacs
from qulacs.circuit import QuantumCircuitOptimizer
from qiskit import QuantumCircuit, Aer, execute
from scipy.linalg import expm
from src.custom_qulacs_gates import RXX, RYY

# TODO: Find out why Qulacs single qubit rotation is opposite sign of qiskit??

######################################################################################################
#                                 TEST CASE GENERATOR FUNCTIONS                                      #
######################################################################################################

__N__ = 6
__L__ = 4


def generate_test_cases_1(nr_rng_trials: int = 10) -> List[Tuple[np.ndarray, np.ndarray]]:
    test_cases = []
    # The maximum quantum gate size allowed to be created
    max_block_size = 1
    for seed in range(nr_rng_trials):
        opt = QuantumCircuitOptimizer()
        np.random.seed(seed + 2)
        thetas = np.random.uniform(-2 * np.pi, 2 * np.pi, __L__)
        pairs = [np.random.choice(__N__, 2, replace=False) for layer in range(__L__)]

        # --- Qulacs --- #
        qulacs_state_vector = qulacs.QuantumState(__N__)
        qulacs_circuit = qulacs.QuantumCircuit(__N__)
        for layer in range(__L__):
            RXX(circuit=qulacs_circuit, angle=thetas[layer], qubit_1=pairs[layer][0], qubit_2=pairs[layer][1])
            RYY(circuit=qulacs_circuit, angle=thetas[layer], qubit_1=pairs[layer][0], qubit_2=pairs[layer][1])

        opt.optimize(qulacs_circuit, max_block_size)
        qulacs_circuit.update_quantum_state(qulacs_state_vector)
        qulacs_state_vector = qulacs_state_vector.get_vector()

        # --- Qiskit --- #
        qiskit_circuit = QuantumCircuit(__N__)
        for layer in range(__L__):
            qiskit_circuit.rxx(theta=thetas[layer], qubit1=pairs[layer][0], qubit2=pairs[layer][1])
            qiskit_circuit.ryy(theta=thetas[layer], qubit1=pairs[layer][0], qubit2=pairs[layer][1])
        qiskit_state_vector = np.array(execute(qiskit_circuit,
                                               Aer.get_backend('statevector_simulator')).result().get_statevector())

        test_cases.append((qiskit_state_vector, qulacs_state_vector))

    return test_cases


def generate_test_cases_2(nr_rng_trials: int = 10) -> List[Tuple[np.ndarray, np.ndarray]]:
    test_cases = []
    # The maximum quantum gate size allowed to be created
    max_block_size = 2
    for seed in range(nr_rng_trials):
        opt = QuantumCircuitOptimizer()
        np.random.seed(seed + 2)
        thetas = np.random.uniform(-2 * np.pi, 2 * np.pi, __L__)
        pairs = [np.random.choice(__N__, 2, replace=False) for layer in range(__L__)]

        # --- Qulacs --- #
        qulacs_state_vector = qulacs.QuantumState(__N__)
        qulacs_circuit = qulacs.QuantumCircuit(__N__)
        for layer in range(__L__):
            RXX(circuit=qulacs_circuit, angle=thetas[layer], qubit_1=pairs[layer][0], qubit_2=pairs[layer][1])
            RYY(circuit=qulacs_circuit, angle=thetas[layer], qubit_1=pairs[layer][0], qubit_2=pairs[layer][1])

        opt.optimize(qulacs_circuit, max_block_size)
        qulacs_circuit.update_quantum_state(qulacs_state_vector)
        qulacs_state_vector = qulacs_state_vector.get_vector()

        # --- Qiskit --- #
        qiskit_circuit = QuantumCircuit(__N__)
        for layer in range(__L__):
            qiskit_circuit.rxx(theta=thetas[layer], qubit1=pairs[layer][0], qubit2=pairs[layer][1])
            qiskit_circuit.ryy(theta=thetas[layer], qubit1=pairs[layer][0], qubit2=pairs[layer][1])
        qiskit_state_vector = np.array(execute(qiskit_circuit,
                                               Aer.get_backend('statevector_simulator')).result().get_statevector())

        test_cases.append((qiskit_state_vector, qulacs_state_vector))

    return test_cases


#############################################################################
#                                 TESTING                                   #
#############################################################################

N_RNG_TRIALS = 10
test_cases_1 = generate_test_cases_1(nr_rng_trials=N_RNG_TRIALS)
test_cases_2 = generate_test_cases_2(nr_rng_trials=N_RNG_TRIALS)


@pytest.mark.parametrize('qiskit_state_vector, qulacs_state_vector', test_cases_1, )
def test_circuit_opt_1(qiskit_state_vector: np.ndarray,
                       qulacs_state_vector: np.ndarray):
    # Comparing
    assert np.allclose(qiskit_state_vector, qulacs_state_vector)


@pytest.mark.parametrize('qiskit_state_vector, qulacs_state_vector', test_cases_2, )
def test_circuit_opt_2(qiskit_state_vector: np.ndarray,
                       qulacs_state_vector: np.ndarray):
    # Comparing
    assert np.allclose(qiskit_state_vector, qulacs_state_vector)
