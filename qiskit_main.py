import os
from typing import *
from joblib import (Parallel,
                    delayed)

import numpy as np
import scipy as sc
from tqdm import tqdm

from src.Tools import portfolio_metrics, save_data_to_hdf
from qiskit_src.ansatz import QAOA, CP_QAOA
from qiskit_src.tools import get_qubo, min_cost_partition


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
        _available_methods_ = ['COBYLA', 'Nelder-Mead']
        _method_idx_ = 0

        ####################################################################################
        #################################### ONLY MIXER ####################################
        ####################################################################################

        ansatz = CP_QAOA(N_qubits=__N__,
                         cardinality=__k__,
                         layers=__layers__,
                         QUBO_matrix=Q,
                         QUBO_offset=offset,
                         with_z_phase=False)

        # Initial guess for parameters (gamma, beta) of circuit
        theta_min, theta_max = -np.pi, np.pi
        N_xx_yy_angles = layers * (N - 1)
        theta_i = np.random.normal(loc=0, scale=1, size=N_xx_yy_angles)

        # Use the get_cost method of the specific ansatz instance
        res = sc.optimize.minimize(fun=ansatz.get_cost, x0=theta_i,
                                   method=_available_methods_[_method_idx_],
                                   options={'disp': False, 'maxiter': __max_iter__})

        _dict_ = ansatz.get_state_probabilities(angles=res.x, flip_states=False)
        Final_circuit_sample_states = np.array([[int(bit) for bit in key] for key in list(_dict_.keys())], dtype=int)
        Final_circuit_sample_probabilities = np.array([_dict_[key] for key in list(_dict_.keys())], dtype=np.float64)

        c = res.fun + offset
        TO_STORE = {'type': 1,
                    'N': __N__,
                    'k': __k__,
                    'layers': __layers__,
                    'Max_cost': max_cost,
                    'Min_cost': min_cost,
                    'Min_cost_state': min_state,
                    'Cost': c,
                    'Normalized_cost': 1 / (max_cost - min_cost) * c - 1 / (max_cost / min_cost - 1),
                    'Final_circuit_sample_states': Final_circuit_sample_states,
                    'Final_circuit_sample_probabilities': Final_circuit_sample_probabilities,
                    'Expected_returns': expected_returns,
                    'Covariances': covariances,
                    'Optimizer_nfev': res.nfev,
                    'Optimizer_maxfev': __max_iter__,
                    'Rng_seed': __seed__}

        result.append(TO_STORE)

        #########################################################################################
        #################################### MIXER & Z-PHASE ####################################
        #########################################################################################

        ansatz = CP_QAOA(N_qubits=__N__,
                         cardinality=__k__,
                         layers=__layers__,
                         QUBO_matrix=Q,
                         QUBO_offset=offset,
                         with_z_phase=True)

        # Initial guess for parameters (gamma, beta) of circuit
        theta_min, theta_max = -np.pi, np.pi
        N_xx_yy_angles = layers * (N - 1)
        N_z_angles = layers * N
        theta_i = np.random.normal(loc=0, scale=1, size=N_xx_yy_angles + N_z_angles)

        # Use the get_cost method of the specific ansatz instance
        res = sc.optimize.minimize(fun=ansatz.get_cost, x0=theta_i,
                                   method=_available_methods_[_method_idx_],
                                   options={'disp': False, 'maxiter': __max_iter__})

        _dict_ = ansatz.get_state_probabilities(angles=res.x, flip_states=False)
        Final_circuit_sample_states = np.array([[int(bit) for bit in key] for key in list(_dict_.keys())], dtype=int)
        Final_circuit_sample_probabilities = np.array([_dict_[key] for key in list(_dict_.keys())], dtype=np.float64)

        c = res.fun + offset
        TO_STORE = {'type': 2,
                    'N': __N__,
                    'k': __k__,
                    'layers': __layers__,
                    'Max_cost': max_cost,
                    'Min_cost': min_cost,
                    'Min_cost_state': min_state,
                    'Cost': c,
                    'Normalized_cost': 1 / (max_cost - min_cost) * c - 1 / (max_cost / min_cost - 1),
                    'Final_circuit_sample_states': Final_circuit_sample_states,
                    'Final_circuit_sample_probabilities': Final_circuit_sample_probabilities,
                    'Expected_returns': expected_returns,
                    'Covariances': covariances,
                    'Optimizer_nfev': res.nfev,
                    'Optimizer_maxfev': __max_iter__,
                    'Rng_seed': __seed__}

        result.append(TO_STORE)

        ################################################################################################
        ########################################## CLASSIC QAOA ########################################
        ################################################################################################

        # Create an instance of QAOA
        ansatz = QAOA(N_qubits=__N__,
                      layers=__layers__,
                      QUBO_matrix=Q,
                      QUBO_offset=offset)

        # Initial guess for parameters (gamma, beta) of circuit
        theta_min, theta_max = -np.pi, np.pi
        theta_i = np.random.normal(loc=0, scale=1, size=2 * layers)  # Adjust size for gamma and beta

        # Use the get_cost method of the specific ansatz instance
        res = sc.optimize.minimize(fun=ansatz.get_cost, x0=theta_i,
                                   method=_available_methods_[_method_idx_],
                                   options={'disp': False, 'maxiter': __max_iter__})

        _dict_ = ansatz.get_state_probabilities(angles=res.x, flip_states=False)
        Final_circuit_sample_states = np.array([[int(bit) for bit in key] for key in list(_dict_.keys())], dtype=int)
        Final_circuit_sample_probabilities = np.array([_dict_[key] for key in list(_dict_.keys())], dtype=np.float64)

        c = res.fun + offset
        TO_STORE = {'type': 4,
                    'N': __N__,
                    'k': __k__,
                    'layers': __layers__,
                    'Max_cost': max_cost,
                    'Min_cost': min_cost,
                    'Min_cost_state': min_state,
                    'Cost': c,
                    'Normalized_cost': 1 / (max_cost - min_cost) * c - 1 / (max_cost / min_cost - 1),
                    'Final_circuit_sample_states': Final_circuit_sample_states,
                    'Final_circuit_sample_probabilities': Final_circuit_sample_probabilities,
                    'Expected_returns': expected_returns,
                    'Covariances': covariances,
                    'Optimizer_nfev': res.nfev,
                    'Optimizer_maxfev': __max_iter__,
                    'Rng_seed': __seed__}

        result.append(TO_STORE)

        return result

    alpha = 0.001
    N_seeds = 10
    max_iter = 200
    N_layers = 6
    N_max = 10
    N_min = 10
    datapoints = []
    for N in range(N_min, N_max + 1):
        k = N // 2
        for layers in range(1, N_layers + 1):
            for seed in np.random.randint(low=0, high=2 ** 31, size=N_seeds):
                datapoints.append((N, k, layers, max_iter, seed, alpha))

    N_jobs = os.cpu_count() // 2 + os.cpu_count() // 4
    r = Parallel(n_jobs=N_jobs, verbose=51, backend='loky')(delayed(simulate)(datapoint) for datapoint in datapoints)

    for run in r:
        for data in run:
            save_data_to_hdf(input_data=data)


if __name__ == "__main__":
    main()
