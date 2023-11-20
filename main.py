import os
from typing import *

from joblib import (Parallel,
                    delayed)

import numpy as np
import scipy as sc

from src.Ansatz import CPQAOAansatz

from src.Tools import (portfolio_ising,
                       portfolio_metrics,
                       save_data_to_hdf)


def main():
    def simulate(datapoint: Tuple[int, int, int, int, int, float]) -> list[dict]:
        result = []
        __N__, __k__, __layers__, __max_iter__, __seed__, __alpha__ = datapoint
        seed = __seed__
        expected_returns, covariances = portfolio_metrics(n=__N__,
                                                          seed=seed)

        constrained_dict, full_dict, J, h, offset = portfolio_ising(mu=expected_returns,
                                                                    sigma=covariances,
                                                                    alpha=__alpha__,
                                                                    k=__k__,
                                                                    n=__N__)
        max_cost, min_cost, min_state = constrained_dict['c_max'], constrained_dict['c_min'], constrained_dict['s']

        # Size of circuit
        n_qubits = __N__

        ####################################################################################
        #################################### ONLY MIXER ####################################
        ####################################################################################

        # Defining instance of QAOA ansatz
        QAOA_objective = CPQAOAansatz(n_qubits=n_qubits,
                                      n_layers=__layers__,
                                      w_edges=None,
                                      cardinality=__k__,
                                      precision=64,
                                      with_z_phase=False,
                                      with_cost=False,
                                      classic_QAOA=False)
        QAOA_objective.set_ising_model(J=J, h=h, offset=offset)

        # Initial guess for parameters (gamma, beta) of circuit
        theta_min, theta_max = -np.pi, np.pi
        theta_i = np.random.uniform(low=theta_min, high=theta_max,
                                    size=(QAOA_objective.n_qubits - 1) * __layers__).tolist()

        # ------ Optimizer run ------ #
        _available_methods_ = ['COBYLA', 'Nelder-Mead']
        _method_idx_ = 0

        res = sc.optimize.minimize(fun=QAOA_objective.evaluate_circuit, x0=theta_i,
                                   method=_available_methods_[_method_idx_],
                                   options={'disp': False, 'maxiter': __max_iter__})

        # Final parameters (beta, gamma) for circuit
        theta_f = res.x.tolist()
        c = res.fun

        _dict_ = QAOA_objective.set_circuit(theta=theta_f).get_state_probabilities(reverse_states=False, eps=0)
        Final_circuit_sample_states = [[int(bit) for bit in key] for key in list(_dict_.keys())]
        Final_circuit_sample_states = np.array(Final_circuit_sample_states, dtype=int)

        Final_circuit_sample_probabilities = [_dict_[key] for key in list(_dict_.keys())]
        Final_circuit_sample_probabilities = np.array(Final_circuit_sample_probabilities, dtype=np.float64)

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
                    'Rng_seed': seed}

        result.append(TO_STORE)

        #########################################################################################
        #################################### MIXER & Z-PHASE ####################################
        #########################################################################################

        # Defining instance of QAOA ansatz
        QAOA_objective = CPQAOAansatz(n_qubits=n_qubits,
                                      n_layers=__layers__,
                                      w_edges=None,
                                      cardinality=__k__,
                                      precision=64,
                                      with_z_phase=True,
                                      with_cost=False,
                                      classic_QAOA=False)
        QAOA_objective.set_ising_model(J=J, h=h, offset=offset)

        # Initial guess for parameters (gamma, beta) of circuit
        theta_min, theta_max = -np.pi, np.pi
        theta_i = np.random.uniform(low=theta_min, high=theta_max,
                                    size=2 * (QAOA_objective.n_qubits - 1) * __layers__).tolist()

        # ------ Optimizer run ------ #
        res = sc.optimize.minimize(fun=QAOA_objective.evaluate_circuit, x0=theta_i,
                                   method=_available_methods_[_method_idx_],
                                   options={'disp': False, 'maxiter': __max_iter__})

        # Final parameters (beta, gamma) for circuit
        theta_f = res.x.tolist()
        c = res.fun

        _dict_ = QAOA_objective.set_circuit(theta=theta_f).get_state_probabilities(reverse_states=False, eps=0)
        Final_circuit_sample_states = [[int(bit) for bit in key] for key in list(_dict_.keys())]
        Final_circuit_sample_states = np.array(Final_circuit_sample_states, dtype=int)

        Final_circuit_sample_probabilities = [_dict_[key] for key in list(_dict_.keys())]
        Final_circuit_sample_probabilities = np.array(Final_circuit_sample_probabilities, dtype=np.float64)

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
                    'Rng_seed': seed}

        result.append(TO_STORE)

        ################################################################################################
        ########################################## CLASSIC QAOA ######################################## 
        ################################################################################################

        # Defining instance of QAOA ansatz
        QAOA_objective = CPQAOAansatz(n_qubits=n_qubits,
                                      n_layers=__layers__,
                                      w_edges=None,
                                      cardinality=__k__,
                                      precision=64,
                                      with_z_phase=False,
                                      with_cost=False,
                                      classic_QAOA=True)
        QAOA_objective.set_ising_model(J=J, h=h, offset=offset)

        # Initial guess for parameters (gamma, beta) of circuit
        theta_min, theta_max = -np.pi, np.pi
        n_cost_terms = QAOA_objective.nr_cost_terms
        theta_i = np.random.uniform(low=theta_min, high=theta_max, size=2*__layers__).tolist()

        # ------ Optimizer run ------ #
        res = sc.optimize.minimize(fun=QAOA_objective.evaluate_circuit, x0=theta_i, method=_available_methods_[_method_idx_],
                                   options={'disp': False, 'maxiter': __max_iter__})

        # Final parameters (beta, gamma) for circuit
        theta_f = res.x.tolist()
        c = res.fun

        _dict_ = QAOA_objective.set_circuit(theta=theta_f).get_state_probabilities(reverse_states=False, eps=0)
        Final_circuit_sample_states = [[int(bit) for bit in key] for key in list(_dict_.keys())]
        Final_circuit_sample_states = np.array(Final_circuit_sample_states, dtype=int)

        Final_circuit_sample_probabilities = [_dict_[key] for key in list(_dict_.keys())]
        Final_circuit_sample_probabilities = np.array(Final_circuit_sample_probabilities, dtype=np.float64)

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
                    'Rng_seed': seed}

        result.append(TO_STORE)

        return result

    alpha = 0.001
    N_seeds = 10
    max_iter = 100
    N_layers = 5
    N_max = 5
    N_min = 4
    datapoints = []
    for N in range(N_min, N_max + 1):
        k = N // 2
        for layers in range(1, N_layers + 1):
            for seed in np.random.randint(low=0, high=2 ** 31, size=N_seeds):
                datapoints.append((N, k, layers, max_iter, seed, alpha))

    N_CPU_CORES = os.cpu_count()
    r = Parallel(n_jobs=N_CPU_CORES, verbose=51)(delayed(simulate)(datapoint) for datapoint in datapoints)
    for run in r:
        for data in run:
            save_data_to_hdf(input_data=data)


if __name__ == "__main__":
    main()
