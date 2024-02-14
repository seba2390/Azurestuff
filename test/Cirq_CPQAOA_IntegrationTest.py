from typing import List, Tuple, Dict
import pytest
import numpy as np
from src.Chain import Chain
from src.Cirq_CPQAOA import Cirq_CPQAOA


def filter_small_probabilities(counts: dict[str, float], eps: float = 1e-10) -> dict[str, float]:
    return {state: prob for state, prob in counts.items() if prob > eps}


######################################################################################################
#                                 TEST CASE GENERATOR FUNCTIONS                                      #
######################################################################################################

__N_VALUES__ = [3, 4, 5]
__LAYER_VALUES__ = [1, 2, 3]


def generate_count_test_cases(nr_rng_trials: int = 3) -> List[Tuple[Dict[str, float], Dict[str, float], int]]:
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
                full_space_ansatz = Cirq_CPQAOA(N_qubits=N,
                                                cardinality=k,
                                                layers=layers,
                                                topology=topology,
                                                QUBO_matrix=Q,
                                                get_full_state_vector=True)
                full_space_ansatz.get_cost(angles=angles)

                reduced_space_ansatz = Cirq_CPQAOA(N_qubits=N,
                                                   cardinality=k,
                                                   layers=layers,
                                                   topology=topology,
                                                   QUBO_matrix=Q,
                                                   get_full_state_vector=False)
                reduced_space_ansatz.get_cost(angles=angles)

                test_cases.append((filter_small_probabilities(full_space_ansatz.counts),
                                   filter_small_probabilities(reduced_space_ansatz.counts),
                                   k))
    return test_cases


#############################################################################
#                                 TESTING                                   #
#############################################################################

N_RNG_TRIALS = 3

test_cases = generate_count_test_cases(nr_rng_trials=N_RNG_TRIALS)


@pytest.mark.parametrize('full_space_counts, reduced_space_counts, cardinality', test_cases, )
def test_reduced_and_full_space(full_space_counts: Dict[str, float],
                                reduced_space_counts: Dict[str, float],
                                cardinality: int):

    # Comparing probabilities of two approaches
    for state, probability in full_space_counts.items():
        # Comparing probabilities of two approaches
        assert np.isclose(probability, reduced_space_counts[state])

    # Checking that cardinality is preserved
    for state, probability in full_space_counts.items():
        # Checking that cardinality is preserved
        assert sum([int(q_val) for q_val in state]) == cardinality

    # Checking that cardinality is preserved
    for state, probability in reduced_space_counts.items():
        assert sum([int(q_val) for q_val in state]) == cardinality

    # Checking that all probability is included (should sum to approx. 1)
    assert np.isclose(sum([p for p in list(full_space_counts.values())]), 1.0)
    assert np.isclose(sum([p for p in list(reduced_space_counts.values())]), 1.0)

