import numpy as np


def normalize_array(x: np.ndarray) -> np.ndarray:  # type: ignore[type-arg]
    """Normalizes the futher axis of the given array.

    (eg. in an array of shape (100,50,3) normalizes all the  5000 3D vectors)

    Parameters:
        x:
            the array to be normalized

    Returns:
        the normalized array
    """
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    norm[norm == 0] = 1
    return x / norm
