import numpy as np


def central_diff_grad_2pt(f, x, eps=1e-6):
    """
    Approximates the gradient of a function using the 2-point central difference method.

    Args:
      f: The function to differentiate.
      x: A NumPy array representing the point at which to evaluate the gradient.
      eps: A small step size for numerical differentiation.

    Returns:
      A NumPy array representing the approximate gradient of f at x.
    """
    n = len(x)
    grad = np.zeros_like(x)
    for i in range(n):
        h = eps * np.eye(n)[i]
        grad[i] = (f(x + h) - f(x - h)) / (2 * eps)
    return grad


def central_diff_grad_3pt(f, x, eps=1e-6):
    """
    Approximates the gradient of a function using the 3-point central difference method.

    Args:
      f: The function to differentiate.
      x: A NumPy array representing the point at which to evaluate the gradient.
      eps: A small step size for numerical differentiation.

    Returns:
      A NumPy array representing the approximate gradient of f at x.
    """
    n = len(x)
    grad = np.zeros_like(x)
    for i in range(n):
        h = eps * np.eye(n)[i]
        grad[i] = (-f(x + 2 * h) + 8 * f(x + h) - 8 * f(x - h) + f(x - 2 * h)) / (12 * eps)
    return grad
