from typing import Callable
import numpy as np
from src.custom_optimizers.OptimizerResult import OptResult
from src.custom_optimizers.GradientEstimators import central_diff_grad_2pt, central_diff_grad_3pt


def BFGS(fun: Callable,
         x0: np.ndarray,
         callback: Callable,
         grad: Callable = central_diff_grad_2pt,
         tolerance: float = 1e-6,
         max_iterations: int = 100,
         c1: float = 1e-4,
         c2: float = 0.9) -> OptResult:
    """
    BFGS quasi-Newton optimization algorithm with line search satisfying Wolfe conditions.

    Args:
        fun: The function to be minimized. Takes a NumPy array as input and returns a scalar.
        grad: The gradient of the function. Takes a NumPy array as input and returns a NumPy array (the gradient).
        x0: Initial guess (NumPy array).
        tolerance: Stopping tolerance for the norm of the gradient.
        max_iterations: Maximum number of iterations allowed.
        c1: Constant for the Armijo condition (0 < c1 < 1).
        c2: Constant for the curvature condition (c1 < c2 < 1).

    Returns:
        NumPy array representing the approximate minimizer.
    """

    Hk = np.eye(len(x0))  # Initialize approximate inverse Hessian as identity matrix
    xk = x0.copy()  # Create a copy of the initial guess
    iterations = 0

    while True:
        pk = -np.dot(Hk, grad(fun, xk))  # Calculate the search direction

        # Wolfe line search to find a suitable step size alpha
        alpha, _ = _wolfe_line_search(fun, grad, xk, pk, c1, c2)
        callback()
        xk_new = xk + alpha * pk

        sk = xk_new - xk
        yk = grad(fun, xk_new) - grad(fun, xk)

        if np.linalg.norm(grad(fun,xk_new)) < tolerance:
            return OptResult(nfev=iterations, x=xk_new, fun=fun(xk_new))

        # Update approximate inverse Hessian using the BFGS formula
        rho = 1.0 / np.dot(yk, sk)
        Hk = (np.eye(len(x0)) - rho * sk[:, None] @ yk[None, :]) @ Hk @ (
            np.eye(len(x0)) - rho * yk[:, None] @ sk[None, :]) + rho * sk[:, None] @ sk[None, :]

        xk = xk_new.copy()
        iterations += 1

        if iterations >= max_iterations:
            return OptResult(nfev=iterations, x=xk, fun=fun(xk))


def _wolfe_line_search(fun: Callable,
                       grad_f: Callable,
                       xk: np.ndarray,
                       pk: np.ndarray,
                       c1: float,
                       c2: float) -> tuple[float, float]:
    """
    Performs a line search satisfying the Wolfe conditions.

    Args:
        fun: The function to be minimized.
        grad_f: The gradient of the function.
        xk: Current iterate (NumPy array).
        pk: Search direction (NumPy array)
        c1: Constant for the Armijo condition.
        c2: Constant for the curvature condition.

    Returns:
        A tuple containing:
            alpha: The step size satisfying the Wolfe conditions.
            f_new: The function value at the new point (xk + alpha * pk).
    """

    alpha_max = 2.0  # Upper bound for the step size
    alpha_prev = 0.0
    alpha = 1.0  # Initial step size

    while True:
        f_new = fun(xk + alpha * pk)
        grad_f_new = grad_f(fun, xk + alpha * pk)

        if f_new <= fun(xk) + c1 * alpha * np.dot(grad_f(fun, xk), pk):
            if np.dot(grad_f_new, pk) >= c2 * np.dot(grad_f(fun, xk), pk):
                return alpha, f_new

        alpha_temp = alpha
        if (alpha_max - alpha) < (alpha - alpha_prev):
            alpha = (alpha_max + alpha_prev) / 2.0
        else:
            alpha = (alpha + alpha_max) / 2.0

        alpha_prev = alpha_temp
