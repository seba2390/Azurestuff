from typing import List
import os
import pickle

import numpy as np
import scipy as sc
from joblib import Parallel, delayed

from src.CP_VQA.Qulacs_CP_VQA import Qulacs_CP_VQA
from src.QAOA_HYBRID.Qulacs_QAOA_HYBRID import Qulacs_QAOA_HYBRID
from src.QAOA.Qulacs_QAOA import Qulacs_QAOA

from src.Result import SimResult
from src.Chain import Chain
from src.Qubo import Qubo
from src.Tools import (portfolio_metrics,
                       min_cost_partition,
                       get_qubo,
                       check_qubo)


def simulate(settings_collection: List[dict]) -> List[SimResult]:
    result = []
    for settings in settings_collection:
        # Generating random problem instance
        expected_returns, covariances = portfolio_metrics(n=settings['N'], seed=settings['seed'])
        # Retrieving C_min, C_max and corresponding states for original portfolio problem
        constrained_result, full_result, lmbda = min_cost_partition(nr_qubits=settings['N'],
                                                                    k=settings['k'],
                                                                    mu=expected_returns,
                                                                    sigma=covariances,
                                                                    alpha=settings['alpha'])

        # Generating QUBO corresponding to current problem instance
        Q, offset = get_qubo(mu=expected_returns,
                             sigma=covariances,
                             alpha=settings['alpha'],
                             lmbda=lmbda + 1,  # Adding small constant purposely
                             k=settings['k'])
        qubo = Qubo(Q=Q, offset=offset)
        qubo.subspace_c_min, qubo.subspace_c_max = constrained_result['c_min'], constrained_result['c_max']
        qubo.subspace_x_min, qubo.subspace_x_max = constrained_result['s_min'], constrained_result['s_max']
        qubo.full_space_c_min, qubo.full_space_c_max = full_result['c_min'], full_result['c_max']
        check_qubo(QUBO_matrix=Q,
                   QUBO_offset=offset,
                   expected_returns=expected_returns,
                   covariances=covariances,
                   alpha=settings['alpha'],
                   k=settings['k'])

        qaoa = Qulacs_QAOA(N_qubits=settings['N'],
                           cardinality=settings['k'],
                           layers=settings['L'],
                           qubo=qubo)

        cp_vqa = Qulacs_CP_VQA(N_qubits=settings['N'],
                               cardinality=settings['k'],
                               layers=settings['L'],
                               topology=settings['topology'],
                               get_full_state_vector=False,
                               with_next_nearest_neighbors=settings['w_nnn'],
                               qubo=qubo)

        qaoa_hybrid = Qulacs_QAOA_HYBRID(N_qubits=settings['N'],
                                         cardinality=settings['k'],
                                         layers=settings['L'],
                                         topology=settings['topology'],
                                         get_full_state_vector=False,
                                         with_next_nearest_neighbors=settings['w_nnn'],
                                         qubo=qubo)

        # Generating initial guess for rotation angles
        np.random.seed(settings['seed'])
        theta_min, theta_max = -2 * np.pi, 2 * np.pi
        N_angles = settings['L'] * len(settings['topology'].get_NNN_indices()) if settings['w_nnn'] else settings[
                                                                                                             'L'] * len(
            settings['topology'].get_NN_indices())
        CPVQA_theta_i = np.random.uniform(theta_min, theta_max, N_angles)
        QAOA_theta_i = np.random.uniform(theta_min, theta_max, 2 * settings['L'])

        qaoa_sim_res = sc.optimize.minimize(fun=qaoa.get_cost,
                                            x0=QAOA_theta_i,
                                            method=settings['opt_method'],
                                            options={'disp': False,
                                                     'maxiter': settings['max_iter']},
                                            callback=qaoa.callback)
        qaoa_norm_c = np.min(qaoa.normalized_costs)
        qaoa_p = np.max(qaoa.opt_state_probabilities)

        cp_vqa_sim_res = sc.optimize.minimize(fun=cp_vqa.get_cost,
                                              x0=CPVQA_theta_i,
                                              method=settings['opt_method'],
                                              options={'disp': False,
                                                       'maxiter': settings['max_iter']},
                                              callback=cp_vqa.callback)
        cp_vqa_norm_c = np.min(cp_vqa.normalized_costs)
        cp_vqa_p = np.max(cp_vqa.opt_state_probabilities)

        qaoa_hybrid_sim_res = sc.optimize.minimize(fun=qaoa_hybrid.get_cost,
                                                   x0=QAOA_theta_i,
                                                   method=settings['opt_method'],
                                                   options={'disp': False,
                                                            'maxiter': settings['max_iter']},
                                                   callback=qaoa_hybrid.callback)
        qaoa_hybrid_norm_c = np.min(qaoa_hybrid.normalized_costs)
        qaoa_hybrid_p = np.max(qaoa_hybrid.opt_state_probabilities)

        result.append(SimResult(N=settings['N'],
                                k=settings['k'],
                                L=settings['L'],
                                alpha=settings['alpha'],
                                w_nnn=settings['w_nnn'],
                                CP_VQA={'c': cp_vqa_norm_c, 'p': cp_vqa_p, 'nfev': cp_vqa_sim_res.nfev},
                                QAOA={'c': qaoa_norm_c, 'p': qaoa_p, 'nfev': qaoa_sim_res.nfev},
                                QAOA_HYBRID={'c': qaoa_hybrid_norm_c, 'p': qaoa_hybrid_p,
                                             'nfev': qaoa_hybrid_sim_res.nfev}))
    return result



layer_dict = {2:1,  3:1,  4:1,
              5:2,  6:2,  7:2,
              8:2,  9:2,  10:2,
              11:3, 12:3, 13:3,
              14:8, 15:3, 16:3,
              17:3, 18:3, 19:3,
              20:3, 21:7, 22:7}

max_iter_dict = {2: 400,   3:  500,  4:  600,
                 5: 700,   6:  800,  7:  900,
                 8: 1000,  9:  1000, 10: 1000,
                 11: 1000, 12: 1000, 13: 1000,
                 14: 1250, 15: 1250, 16: 1250,
                 17: 1250, 18: 1250, 19: 1250,
                 20: 1500, 21: 2300, 22: 2400}
alpha=0.5
N_seeds = 128
N_min, N_max = 2, 15
sim_settings = []
for seed in range(N_seeds):
    chunk = []
    for N in range(N_min, N_max+1):
        topology = Chain(N_qubits=N)
        topology.set_initialization_strategy(strategy=np.array([0 if i%2 == 0 else 1 for i in range(N)]))
        setting = {'N'         :N,       'alpha'   :alpha,   'L'       :layer_dict[N],
                   'seed'      :seed,    'topology':topology,'max_iter':max_iter_dict[N],
                   'opt_method':'COBYLA','w_nnn'   :True,    'k'       :N//2}
        chunk.append(setting)
    sim_settings.append(chunk)


N_jobs=os.cpu_count()-1
r = Parallel(n_jobs=N_jobs, verbose=51, backend='loky')(delayed(simulate)(chunk) for chunk in sim_settings)

flattened_result = [sim_res for simulation in r for sim_res in simulation]

with open('simulation_results.pkl', 'wb') as f:
    pickle.dump(flattened_result, f)