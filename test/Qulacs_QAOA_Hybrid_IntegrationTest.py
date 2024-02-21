from typing import List, Tuple, Dict
import pytest
import numpy as np
from src.Chain import Chain
from src.QAOA_HYBRID.Qiskit_QAOA_HYBRID import Qiskit_QAOA_HYBRID
from src.QAOA_HYBRID.Qulacs_QAOA_HYBRID import Qulacs_QAOA_HYBRID
from src.Qubo import Qubo


def filter_small_probabilities(counts: dict[str, float], eps: float = 9.5e-13) -> dict[str, float]:
    return {state: prob for state, prob in counts.items() if prob > eps}


######################################################################################################
#                                 TEST CASE GENERATOR FUNCTIONS                                      #
######################################################################################################

__N_VALUES__ = [3, 4, 5]
__LAYER_VALUES__ = [1, 2, 3]
__N_OPT_STEPS__ = 3


def generate_count_test_cases(nr_rng_trials: int) -> List[Tuple[Dict[str, float], Dict[str, float], int]]:
    test_cases = []
    for seed in range(nr_rng_trials):
        np.random.seed(seed)
        for N in __N_VALUES__:
            k = N // 2
            for layers in __LAYER_VALUES__:
                topology = Chain(N_qubits=N)
                topology.set_initialization_strategy(strategy=np.array([0 if i % 2 == 0 else 1 for i in range(N)]))
                Q = np.random.uniform(0, 1, (N, N))
                Q = (Q + Q.T) / 2.0
                Qiskit_ansatz = Qiskit_QAOA_HYBRID(N_qubits=N,
                                                   cardinality=k,
                                                   layers=layers,
                                                   qubo=Qubo(Q, 0.0),
                                                   topology=topology,
                                                   with_next_nearest_neighbors=True)
                Qulacs_ansatz = Qulacs_QAOA_HYBRID(N_qubits=N,
                                                   cardinality=k,
                                                   layers=layers,
                                                   qubo=Qubo(Q, 0.0),
                                                   topology=topology,
                                                   with_next_nearest_neighbors=True)
                for opt_step in range(__N_OPT_STEPS__):
                    N_angles = 2 * layers + layers * len(topology.get_NNN_indices())
                    angles = np.random.uniform(-2 * np.pi, 2 * np.pi, N_angles)
                    c_1 = Qiskit_ansatz.get_cost(angles=angles)
                    c_2 = Qulacs_ansatz.get_cost(angles=angles)
                    test_cases.append((filter_small_probabilities(Qiskit_ansatz.counts),
                                       filter_small_probabilities(Qulacs_ansatz.counts), k))

    return test_cases


#############################################################################
#                                 TESTING                                   #
#############################################################################

N_RNG_TRIALS = 10

test_cases_1 = generate_count_test_cases(nr_rng_trials=N_RNG_TRIALS)


@pytest.mark.parametrize('qiskit_counts, qulacs_counts, cardinality', test_cases_1, )
def test_probabilities(qiskit_counts: Dict[str, float],
                       qulacs_counts: Dict[str, float],
                       cardinality: int):
    # Checking that all probability is included (should sum to approx. 1)
    assert np.isclose(sum([p for p in list(qiskit_counts.values())]), 1.0)

    # Checking that all probability is included (should sum to approx. 1)
    assert np.isclose(sum([p for p in list(qulacs_counts.values())]), 1.0)

    # Checking that there is only states w. 'k' excitations
    for state, prob in qiskit_counts.items():
        assert state.count('1') == cardinality
    for state, prob in qiskit_counts.items():
        assert state.count('1') == cardinality

    # Comparing the two:
    for state, prob in qulacs_counts.items():
        assert np.isclose(prob, qiskit_counts[state])