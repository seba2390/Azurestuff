import os
from typing import *
from joblib import (Parallel,
                    delayed)

import numpy as np
import scipy as sc
from tqdm import tqdm

from src.Tools import portfolio_metrics, save_data_to_hdf
from qiskit_src.ansatz import QAOA, CP_QAOA, qubo_cost
from qiskit_src.tools import get_qubo, min_cost_partition, normalized_cost


def main():
    def simulate(datapoint: Tuple[int, int, int, int, int, float]) -> List[dict]:
        result = []
        __N__, __k__, __layers__, __max_iter__, __seed__, __alpha__ = datapoint

        expected_returns, covariances = portfolio_metrics(n=__N__,
                                                          seed=__seed__)

        constrained_result, full_result, lmbda = min_cost_partition(nr_qubits=__N__,
                                                                    k=__k__,
                                                                    mu=expected_returns,
                                                                    sigma=covariances,
                                                                    alpha=__alpha__)

        max_cost, min_cost, min_state = constrained_result['c_max'], constrained_result['c_min'], constrained_result[
            's']

        Q, offset = get_qubo(mu=expected_returns,
                             sigma=covariances,
                             alpha=__alpha__,
                             lmbda=lmbda,
                             k=__k__)
        _available_methods_ = ['COBYLA', 'Nelder-Mead', 'L-BFGS-B']

        _method_idx_ = 0

        # ---------------------- #
        # -- Mixer w. k first -- #
        # ---------------------- #
        w_evenly_distributed_k = True
        w_next_nearest_neighbors = True
        w_z_phase = False

        ansatz = CP_QAOA(N_qubits=__N__,
                         cardinality=__k__,
                         layers=__layers__,
                         QUBO_matrix=Q,
                         QUBO_offset=offset,
                         with_evenly_distributed_start_x=w_evenly_distributed_k,
                         with_next_nearest_neighbors=w_next_nearest_neighbors,
                         with_z_phase=w_z_phase)

        # Initial guess for parameters
        N_xx_yy_angles = __layers__ * (N - 1)
        if w_next_nearest_neighbors:
            N_xx_yy_angles += __layers__ * (N - 2)
        if w_z_phase:
            N_xx_yy_angles += N * __layers__
        theta_i = np.random.normal(loc=0, scale=1, size=N_xx_yy_angles)

        iteration_dicts = []

        def callback_function(x):
            iteration_dicts.append(ansatz.get_state_probabilities(angles=x, flip_states=False))

        res = sc.optimize.minimize(fun=ansatz.get_cost, x0=theta_i,
                                   method=_available_methods_[_method_idx_],
                                   options={'disp': False, 'maxiter': __max_iter__},
                                   callback=callback_function)
        _dict_ = ansatz.get_state_probabilities(angles=res.x, flip_states=False)
        Final_circuit_sample_states = np.array([[int(bit) for bit in key] for key in list(_dict_.keys())], dtype=int)
        Final_circuit_sample_probabilities = np.array([_dict_[key] for key in list(_dict_.keys())], dtype=np.float64)
        TO_STORE = {'type': 1,
                    'N': __N__,
                    'k': __k__,
                    'layers': __layers__,
                    'Max_cost': max_cost,
                    'Min_cost': min_cost,
                    'Min_cost_state': min_state,
                    'Normalized_cost': normalized_cost(result=_dict_,
                                                       QUBO_matrix=Q,
                                                       QUBO_offset=offset,
                                                       max_cost=max_cost,
                                                       min_cost=min_cost),
                    'Final_circuit_sample_states': Final_circuit_sample_states,
                    'Final_circuit_sample_probabilities': Final_circuit_sample_probabilities,
                    'Expected_returns': expected_returns,
                    'Covariances': covariances,
                    'Optimizer_nfev': res.nfev,
                    'Optimizer_maxfev': __max_iter__,
                    'Rng_seed': __seed__,
                    'status': res.status,
                    'iteration_cost': [normalized_cost(result=it,
                                                       QUBO_matrix=Q,
                                                       QUBO_offset=offset,
                                                       max_cost=max_cost,
                                                       min_cost=min_cost) for it in iteration_dicts]}
        result.append(TO_STORE)
        return result

    alpha = 0.001
    N_seeds = 100
    max_iter = 5000
    min_layers, max_layers = 5, 5
    N_max = 15
    N_min = 15
    datapoints = []
    for N in range(N_min, N_max + 1):
        k = N // 2
        for layers in range(min_layers, max_layers + 1):
            for seed in np.random.randint(low=0, high=2 ** 31, size=N_seeds):
                datapoints.append((N, k, layers, max_iter, seed, alpha))

    N_jobs = 60
    r = Parallel(n_jobs=N_jobs, verbose=51, backend='loky')(delayed(simulate)(datapoint) for datapoint in datapoints)

    for run in r:
        for data in run:
            save_data_to_hdf(input_data=data)


if __name__ == "__main__":
    main()
