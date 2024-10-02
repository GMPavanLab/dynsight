from typing import Literal

import numpy as np
import numpy.typing as npt
from scipy.signal import find_peaks


def normalize_array(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
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
    x_axis: npt.NDArray[np.float64],
    y_axis: npt.NDArray[np.float64],
    extrema_type: Literal["min", "max"],
    prominence: float,
) -> npt.NDArray[np.float64]:
    """Find the extrema points of a mathematical function (minima or maxima).

    Parameters:
        x_axis:
            x values of the function.
        y_axis:
            y values of the function.
        extrema_type:
            It can be "min" or "max".
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

    function = -y_axis if extrema_type == "min" else y_axis
    peaks, _ = find_peaks(x=function, prominence=prominence)
    minima_xcoord = x_axis[peaks]
    minima_ycoord = y_axis[peaks]
    return np.column_stack((minima_xcoord, minima_ycoord))
