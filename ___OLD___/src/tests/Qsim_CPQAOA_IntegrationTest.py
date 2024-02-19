from typing import List, Tuple, Dict
import pytest
import numpy as np
from src.Chain import Chain
from src.CP_VQA.Qsim_CP_VQA import Qsim_CPQAOA
from src.CP_VQA.Qiskit_CP_VQA import CP_QAOA


def filter_small_probabilities(counts: dict[str, float], eps: float = 9.5e-13) -> dict[str, float]:
    return {state: prob for state, prob in counts.items() if prob > eps}


######################################################################################################
#                                 TEST CASE GENERATOR FUNCTIONS                                      #
######################################################################################################

__N_VALUES__ = [3, 4, 5]
__LAYER_VALUES__ = [1, 2, 3]


def generate_count_test_cases(nr_rng_trials: int) -> List[Tuple[Dict[str, float], Dict[str, float], int]]:
    test_cases = []
    for seed in range(nr_rng_trials):
        np.random.seed(seed)
        for N in __N_VALUES__:
            k = N // 2
            for layers in __LAYER_VALUES__:
                topology = Chain(N_qubits=N)
                topology.set_initialization_strategy(strategy=np.array([0 if i % 2 == 0 else 1 for i in range(N)]))
                angles = np.random.uniform(-2 * np.pi, 2 * np.pi, layers * len(topology.get_NN_indices()))
                Q = np.random.uniform(0, 1, (N, N))
                Q = (Q + Q.T) / 2.0
                Qsim_ansatz = Qsim_CPQAOA(N_qubits=N,
                                              cardinality=k,
                                              layers=layers,
                                              topology=topology,
                                              QUBO_matrix=Q,
                                              approximate_hamiltonian=True)
                Qsim_ansatz.get_cost(angles=angles)

                Qiskit_ansatz = CP_QAOA(N_qubits=N,
                                        cardinality=k,
                                        layers=layers,
                                        topology=topology,
                                        QUBO_matrix=Q,
                                        approximate_hamiltonian=True)
                Qiskit_ansatz.get_cost(angles=angles)

                test_cases.append((filter_small_probabilities(Qsim_ansatz.counts),
                                   filter_small_probabilities(Qiskit_ansatz.counts),
                                   k))
    return test_cases


#############################################################################
#                                 TESTING                                   #
#############################################################################

N_RNG_TRIALS = 3

test_cases_1 = generate_count_test_cases(nr_rng_trials=N_RNG_TRIALS)


@pytest.mark.parametrize('qsim_counts, qiskit_counts, cardinality', test_cases_1, )
def test_probabilities_1(qsim_counts: Dict[str, float],
                         qiskit_counts: Dict[str, float],
                         cardinality: int):

    # Comparing probabilities of two approaches
    for state, probability in qiskit_counts.items():
        # Comparing probabilities of two approaches
        assert np.isclose(probability, qsim_counts[state])

    # Checking that cardinality is preserved
    for state, probability in qsim_counts.items():
        # Checking that cardinality is preserved
        assert sum([int(q_val) for q_val in state]) == cardinality

    # Checking that all probability is included (should sum to approx. 1)
    assert np.isclose(sum([p for p in list(qsim_counts.values())]), 1.0)

