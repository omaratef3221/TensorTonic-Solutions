import numpy as np

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Returns the last-axis normalized array.
    """
    mean = x.mean(axis = -1, keepdims = True)
    sigma = x.var(axis = -1, keepdims = True)

    numerator = x - mean
    den = np.sqrt(sigma + eps)

    division_result = numerator/den

    result = gamma * division_result + beta

    return result