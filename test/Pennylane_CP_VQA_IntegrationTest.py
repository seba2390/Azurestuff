from typing import List, Tuple, Dict
import pytest
import numpy as np
from src.Chain import Chain
from src.CP_VQA.Pennylane_CP_VQA import Pennylane_CP_VQA
from src.CP_VQA.Qiskit_CP_VQA import Qiskit_CP_VQA
from src.Qubo import Qubo


def filter_small_probabilities(counts: dict[str, float], eps: float = 9.5e-15) -> dict[str, float]:
    return {state: prob for state, prob in counts.items() if prob > eps}


######################################################################################################
#                                 TEST CASE GENERATOR FUNCTIONS                                      #
######################################################################################################

__N_VALUES__ = [3, 4, 5]
__LAYER_VALUES__ = [1, 2, 3]


def generate_count_test_cases(nr_rng_trials: int) -> List[
    Tuple[Dict[str, float], Dict[str, float], int]]:
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
                Pennylane_ansatz = Pennylane_CP_VQA(N_qubits=N,
                                                    cardinality=k,
                                                    layers=layers,
                                                    topology=topology,
                                                    qubo=Qubo(Q, 0.0))
                Pennylane_ansatz.get_cost(angles=angles)

                Qiskit_ansatz = Qiskit_CP_VQA(N_qubits=N,
                                              cardinality=k,
                                              layers=layers,
                                              topology=topology,
                                              qubo=Qubo(Q=Q, offset=0.0),
                                              approximate_hamiltonian=True)
                Qiskit_ansatz.get_cost(angles=angles)

                test_cases.append((filter_small_probabilities(Pennylane_ansatz.counts),
                                   filter_small_probabilities(Qiskit_ansatz.counts),
                                   k))
    return test_cases


#############################################################################
#                                 TESTING                                   #
#############################################################################

N_RNG_TRIALS = 3

test_cases_1 = generate_count_test_cases(nr_rng_trials=N_RNG_TRIALS)


@pytest.mark.parametrize('pennylane_counts, qiskit_counts, cardinality', test_cases_1, )
def test_probabilities_1(pennylane_counts: Dict[str, float],
                         qiskit_counts: Dict[str, float],
                         cardinality: int):
    # Comparing probabilities of two approaches
    for state, probability in qiskit_counts.items():
        # Comparing probabilities of two approaches
        assert np.isclose(probability, pennylane_counts[state])

    # Checking that cardinality is preserved
    for state, probability in pennylane_counts.items():
        # Checking that cardinality is preserved
        assert sum([int(q_val) for q_val in state]) == cardinality

    # Checking that all probability is included (should sum to approx. 1)
    assert np.isclose(sum([p for p in list(pennylane_counts.values())]), 1.0)
