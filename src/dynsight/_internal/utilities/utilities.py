from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from pathlib import Path

import numpy as np
import numpy.typing as npt
from scipy.signal import find_peaks

from dynsight.trajectory import Insight, Trj


def normalize_array(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Normalizes the further axis of the given array.

    (e.g., in an array of shape (100, 50, 3), normalizes all the 5000 3D
    vectors.)

    Parameters:
        x:
            The array to be normalized.

    Returns:
        The normalized array.
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
            The x values of the function.
        y_axis:
            The y values of the function.
        extrema_type:
            The type of extrema to find, which can be "min" for minima or
            "max" for maxima.
        prominence:
            The required prominence of peaks. Higher values will provide only
            well-defined and sharp peaks, excluding softer ones.

    Returns:
        A NumPy array with dimensions (n_peaks, 2), containing the x and y
        coordinates for each peak.
    """
    if extrema_type not in {"min", "max"}:
        type_msg = "extrema_type must be 'min' or 'max'"
        raise ValueError(type_msg)

    function = -y_axis if extrema_type == "min" else y_axis
    peaks, _ = find_peaks(x=function, prominence=prominence)
    minima_xcoord = x_axis[peaks]
    minima_ycoord = y_axis[peaks]
    return np.column_stack((minima_xcoord, minima_ycoord))


def load_or_compute_soap(
    trj: Trj,
    r_cut: float,
    n_max: int,
    l_max: int,
    selection: str = "all",
    centers: str = "all",
    respect_pbc: bool = True,
    n_core: int = 1,
    soap_path: Path | None = None,
) -> Insight:
    """Load or compute SOAP.

    If a valid path to a .json file with a SOAP Insight is provided, that
    Insight is loaded and returned. Otherwise, the Insight is computed from
    the Trj.

    Returns:
        Insight object containing SOAP descriptors.
    """
    if soap_path and soap_path.exists():
        return Insight.load_from_json(soap_path)

    soap = trj.get_soap(
        r_cut=r_cut,
        n_max=n_max,
        l_max=l_max,
        selection=selection,
        centers=centers,
        respect_pbc=respect_pbc,
        n_core=n_core,
    )

    if soap_path:
        soap.dump_to_json(soap_path)

    return soap
