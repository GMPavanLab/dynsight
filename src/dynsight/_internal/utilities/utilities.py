from __future__ import annotations

from pathlib import Path
from typing import Callable, Literal, Mapping, Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd
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
    if soap_path is not None and soap_path.exists():
        return Insight.load_from_json(soap_path)

    soap = trj.get_soap(
        r_cut=r_cut,
        n_max=n_max,
        l_max=l_max,
        selection=selection,
        centers=centers,
        respect_pbc=respect_pbc,
        n_jobs=n_core,
    )

    if soap_path is not None:
        soap.dump_to_json(soap_path)

    return soap


Col = Literal["name", "x", "y", "z", "ID"]


def _entry_from_parts(
    parts: Sequence[str],
    cols_order: Sequence[Col],
    frame: int,
) -> dict[str, object]:
    converters: Mapping[Col, Callable[[str], object]] = {
        "name": str,
        "x": float,
        "y": float,
        "z": float,
        "ID": int,
    }
    entry: dict[str, object] = {"frame": frame}
    for c, col in enumerate(cols_order):
        entry[col] = converters[col](parts[c])
    return entry


def read_xyz(
    input_xyz: Path | str,
    cols_order: Sequence[Col],
) -> pd.DataFrame:
    """Read an XYZ trajectory file into a pandas DataFrame.

    The function parses a file in extended XYZ format where each frame begins
    with a line containing the number of atoms, followed by a comment/title
    line, and then one line per atom containing at least one of the columns
    specified in `cols_order`, following the correct order in the file.

    Parameters:
        input_xyz :
            Path to the XYZ file to read.
        cols_order :
            The expected column order for each atom line (e.g.,
            ["name", "x", "y", "z", "ID"]).

    Returns:
        A DataFrame containing all parsed atomic entries. Each row corresponds
        to one atom in one frame, with columns given by `cols_order` plus the
        current frame indexing.
    """
    lines = Path(input_xyz).read_text().splitlines()
    data: list[dict[str, object]] = []

    frame = -1
    row = 0
    nlines = len(lines)

    while row < nlines:
        token = lines[row].strip()
        if token.isdigit():
            n_atoms = int(token)
            frame += 1
            row += 2  # skip comment/title line

            end = min(row + n_atoms, nlines)
            for i in range(row, end):
                parts = lines[i].split()
                if len(parts) < len(cols_order):
                    continue
                data.append(_entry_from_parts(parts, cols_order, frame))

            row += n_atoms
        else:
            row += 1

    return pd.DataFrame(data)


def save_xyz_from_ndarray(
    output_path: Path | str,
    coords: npt.NDArray[np.float64],
    comment_line: str = "# comment line",
    atom_type: str = "C",
) -> None:
    """Saves a .xyz file with the coordinates from a numpy.ndarray.

    This saves an array of coordinates as an .xyz file. This is useful to
    visualize simple systems' trajectory with visualization tools like Ovito.
    If your system requires more information that what is stored in a simple
    .xyz format, you should consider building a Trj object.

    Parameters:
        output_file:
            The path to the .xyz output trajectory.

        coords:
            The array containing the coordinates of all the particles at each
            frame. Has shape (n_frames, n_atoms, 3).

        comment_line:
            The second line in each frame.

        atom_type:
            The type to assign to all atoms in the trajectory.
    """
    if isinstance(output_path, str):
        output_path = Path(output_path)

    n_coordinates = 3
    if coords.shape[2] != n_coordinates:
        msg = "coords array must have shape (n_frames, n_atoms, 3)."
        raise ValueError(msg)

    with output_path.open("w+") as file:
        for _, frame in enumerate(coords):
            print(coords.shape[1], file=file)
            print(comment_line, file=file)
            for _, atom in enumerate(frame):
                print(atom_type, atom[0], atom[1], atom[2], file=file)
