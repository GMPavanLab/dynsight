from typing import Any

import numpy as np
from scipy.signal import find_peaks


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


def find_extrema_points(
    x_axis: np.ndarray[float, Any],
    y_axis: np.ndarray[float, Any],
    extrema_type: str,
    prominence: float,
) -> np.ndarray[float, Any]:
    """Find the extrema points of a mathematical function (minima or maxima).

    Parameters:
        x_axis:
            x values of the function.
        y_axis:
            y values of the function.
        extrema_type:
            It can be "min" or "max" depending on what the user's choice.
        prominence:
            Required prominence of peaks. Higher values will provides only
            well defined and sharp peaks excluding the softer ones.

    Returns:
        A NumPy array with dimensions (n_peaks, 2), containing
        the x and y coordinates for each peak.
    """
    if extrema_type not in {"min", "max"}:
        type_msg = "extrema_type must be 'min' or 'max'"
        raise ValueError(type_msg)
    if extrema_type == "min":
        inverted_function = -y_axis
        # Find peaks in the inverted RDF
        peaks, properties = find_peaks(
            x=inverted_function, prominence=prominence
        )
    elif extrema_type == "max":
        peaks, properties = find_peaks(x=y_axis, prominence=prominence)
    minima_xcoord = x_axis[peaks]
    minima_ycoord = y_axis[peaks]
    return np.column_stack((minima_xcoord, minima_ycoord))
