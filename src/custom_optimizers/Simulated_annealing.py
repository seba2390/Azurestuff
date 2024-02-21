from typing import List, Tuple

import numpy as np
import random
import math

from src.custom_optimizers.OptimizerResult import OptResult


def standard_neighbor_func(state: np.ndarray,
                           bounds: List[Tuple[float, float]]) -> np.ndarray:
    """
    Generates a neighboring state by slightly perturbing the parameters of the quantum circuit.

    Args:
        bounds:
        state:  A NumPy array representing the current parameter configuration.

    Returns:
        A new NumPy array representing a neighboring parameter configuration.
    """
    new_state = state.copy()
    parameter_idx = np.random.randint(len(state))
    perturbation = np.random.normal(loc=0, scale=1)
    new_state[parameter_idx] += perturbation

    # Potential constraint handling (if your parameters have bounds)
    new_state = np.clip(new_state, a_min=np.array(bounds)[:, 0], a_max=np.array(bounds)[:, 1])  # Example for angles

    return new_state


def simulated_annealing(
    fun: callable,
    x0: np.ndarray,
    initial_temperature: float,
    callback: callable,
    bounds: List[Tuple[float, float]],
    cooling_rate: float = 0.85,
    neighbor: callable = standard_neighbor_func,
    max_iterations: int = 1000) -> OptResult:
    """Implements simulated annealing with a callback function.

    Args:
        bounds:
        fun: Function to evaluate the cost/energy of a state.
        neighbor: Function to generate a neighboring state.
        x0: Starting state for the optimization.
        initial_temperature: Starting temperature for the annealing process.
        cooling_rate: Rate at which the temperature decreases.
        max_iterations: Maximum number of iterations.
        callback: Function called after each evaluation of the objective function.

    Returns:
        A tuple containing the best state found and its corresponding energy (cost).
    """
    if not np.allclose(np.clip(x0, a_min=np.array(bounds)[:, 0], a_max=np.array(bounds)[:, 1]), np.array(x0)):
        raise ValueError('x0 must be within bounds...')
    current_state = x0
    current_energy = fun(current_state)
    callback()
    best_state = current_state
    best_energy = current_energy

    nfev = 0
    for t in range(max_iterations):
        temperature = initial_temperature * (cooling_rate ** t)
        if temperature <= 0:
            return OptResult(nfev=nfev, x=best_state, fun=best_energy)

        next_state = neighbor(current_state, bounds)
        next_energy = fun(next_state)
        callback()
        nfev += 1

        delta_energy = next_energy - current_energy
        if delta_energy < 0 or random.random() < math.exp(-delta_energy / temperature):
            current_state = next_state
            current_energy = next_energy

        if current_energy < best_energy:
            best_state = current_state
            best_energy = current_energy

    return OptResult(nfev=nfev, x=best_state, fun=best_energy)
