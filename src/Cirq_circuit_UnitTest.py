from typing import List, Tuple
import random

import pytest
import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qiskit.quantum_info import Operator
import cirq
from scipy.linalg import expm

from custom_cirq_gates import RXX, RYY


def generate_unique_pairs(N, k):
    # Ensure k is not larger than the number of unique pairs possible
    if k > N * (N - 1):
        raise ValueError("k is too large for the given N")
    # Generate all possible unique tuples
    all_tuples = [(i, j) for i in range(N) for j in range(N) if i != j]
    # Randomly select 'k' tuples from the list
    selected_tuples = random.sample(all_tuples, k)
    return selected_tuples


######################################################################################################
#                                 TEST CASE GENERATOR FUNCTIONS                                      #
######################################################################################################


def generate_test_cases(nr_rng_trials: int = 10) -> List[Tuple[np.ndarray, np.ndarray]]:
    __N_QUBITS__ = 6
    __N_TERMS__ = 10
    __k__ = __N_QUBITS__ // 2
    test_cases = []
    qubits = [cirq.NamedQubit(f'q_{i}') for i in range(__N_QUBITS__)]
    for seed in range(nr_rng_trials):
        # --- Generating random settings for instance --- #
        rotation_angles = np.random.uniform(-2 * np.pi, 2 * np.pi, __N_TERMS__)
        initialization_strategy = np.random.choice(__N_QUBITS__, __k__, replace=False)
        qubit_pairs = generate_unique_pairs(N=__N_QUBITS__, k=__N_TERMS__)

        # ----- Cirq ----- #
        cirq_circuit = cirq.Circuit()
        for qubit_idx in initialization_strategy:
            cirq_circuit.append(cirq.X(qubits[qubit_idx]))
        angle_counter = 0
        for (q_i, q_j) in qubit_pairs:
            theta = rotation_angles[angle_counter]
            cirq_circuit.append(RXX(theta=theta).on(qubits[q_i], qubits[q_j]))
            cirq_circuit.append(RYY(theta=theta).on(qubits[q_i], qubits[q_j]))
            angle_counter += 1
        cirq_simulator = cirq.Simulator()
        # Using qubit_order=reversed(qubits) to match Qiskit convention
        cirq_state_vector = cirq_simulator.simulate(cirq_circuit, qubit_order=reversed(qubits)).final_state_vector
        cirq_probs = np.abs(cirq_state_vector) ** 2

        # ----- Qiskit ----- #
        qiskit_simulator = Aer.get_backend('statevector_simulator')
        qiskit_circuit = QuantumCircuit(__N_QUBITS__)
        for qubit_index in initialization_strategy:
            qiskit_circuit.x(qubit_index)
        angle_idx = 0
        for (qubit_i, qubit_j) in qubit_pairs:
            qiskit_circuit.rxx(theta=float(rotation_angles[angle_idx]), qubit1=qubit_i, qubit2=qubit_j)
            qiskit_circuit.ryy(theta=float(rotation_angles[angle_idx]), qubit1=qubit_i, qubit2=qubit_j)
            angle_idx += 1
        qiskit_state_vector = np.array(execute(qiskit_circuit, qiskit_simulator).result().get_statevector())
        qiskit_probs = np.abs(qiskit_state_vector) ** 2
        test_cases.append((cirq_probs, qiskit_probs))

    return test_cases


#############################################################################
#                                 TESTING                                   #
#############################################################################

N_RNG_TRIALS = 20
test_cases = generate_test_cases(nr_rng_trials=N_RNG_TRIALS)


@pytest.mark.parametrize('vector_1, vector_2', test_cases, )
def test_rxx_gate(vector_1: np.ndarray,
                  vector_2: np.ndarray):
    # Comparing
    assert np.allclose(vector_1, vector_2)
