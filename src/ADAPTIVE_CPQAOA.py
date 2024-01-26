from typing import List, Dict, Union
import random
from collections import Counter

import scipy.linalg
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import Parameter
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import Operator
from scipy.linalg import expm
import numpy as np
import torch

from src.Tools import get_qiskit_H
from src.Tools import qubo_cost, string_to_array, create_operator, operator_expectation, get_generator
from src.CPQAOA import CP_QAOA
from src.Grid import Grid
from src.Tools import get_full_hamiltonian
from src.Chain import Chain
from src.TorchQcircuit import *


class ADAPTIVE_CP_QAOA(CP_QAOA):
    def __init__(self,
                 N_qubits,
                 cardinality,
                 layers,
                 QUBO_matrix,
                 topology: Union[Grid, Chain],
                 with_z_phase: bool = False,
                 with_next_nearest_neighbors: bool = False,
                 with_gradient: bool = False,
                 approximate_hamiltonian: bool = True,
                 normalize_cost: bool = False,
                 backend: str = 'state_vector',
                 N_samples: int = 1000,
                 seed: int = 0,
                 debug_verbose: bool = False):
        random.seed(seed)
        super().__init__(N_qubits, cardinality, layers, QUBO_matrix, topology,
                         with_z_phase, with_next_nearest_neighbors, with_gradient,
                         approximate_hamiltonian, normalize_cost, backend, N_samples,
                         seed, debug_verbose)


