import logging
from pathlib import Path

import pandas as pd
import trackpy as tp

from dynsight.trajectory import Trj

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def track_xyz(
    input_xyz: Path,
    output_xyz: Path,
    search_range: float = 10,
    memory: int = 1,
    adaptive_step: float = 0.5,
    adaptive_stop: float = 0.95,
) -> Trj:
    """Track particles from an xyz file and write a new file with particle IDs.

    The input .xyz is assumed to contain only raw 3D coordinates
    (no atom labels), and each frame begins with a line indicating the number
    of atoms, followed by a comment line, then a list of positions.
    Each frame in the input file must follow this structure::
        <number of atoms>
        comment line
        <x> <y> <z>
        <x> <y> <z>
        ...
        <x> <y> <z>
    The output file will have the same structure, but each line will start
    with the tracked particle ID.
    Parameters:
        input_xyz:
            Path to the input .xyz file containing positions only.
        output_xyz:
            Path where the output .xyz file with particle IDs will be saved.
        search_range:
            Maximum linking distance between frames.
        memory:
            Maximum number of frames a particle can vanish and still be
            re-identified.
        adaptive_step:
            Factor by which the search range is multiplied to reduce it during
            adaptive search.
        adaptive_stop:
            Minimum allowable search range during adaptive search. If the
            search range becomes smaller than this value and ambiguities
            persist, the linking process is aborted for the problematic region.
    """
    input_xyz = Path(input_xyz)
    output_xyz = Path(output_xyz)

    if not input_xyz.exists():
        msg = f"Input file not found: {input_xyz}"
        raise FileNotFoundError(msg)

    positions = _collect_positions(input_xyz=input_xyz)

    if not {"frame", "x", "y", "z"}.issubset(positions.columns):
        msg = "Error in the .xyz format. Each line must be <x> <y> <z>."
        raise ValueError(msg)

    linked = tp.link_df(
        positions,
        search_range=search_range,
        memory=memory,
        adaptive_step=adaptive_step,
        adaptive_stop=adaptive_stop,
    )

    with output_xyz.open("w") as f:
        for frame_num in sorted(linked["frame"].unique()):
            frame_data = linked[linked["frame"] == frame_num]
            f.write(f"{len(frame_data)}\n")
            f.write(f"Frame {frame_num}\n")
            for _, row in frame_data.iterrows():
                pid = int(row["particle"])
                x, y, z = row["x"], row["y"], row["z"]
                f.write(f"{pid} {x:.6f} {y:.6f} {z:.6f}\n")

    logger.info(f"Linked .xyz file written to: {output_xyz}")
    return Trj.init_from_xyz(traj_file=output_xyz, dt=1)


def _collect_positions(input_xyz: Path) -> pd.DataFrame:
    """Read the xyz file and return the positions dataset at each frame."""
    lines = input_xyz.read_text().splitlines()

    data = []
    frame = -1
    row = 0
    dimensions = 3
    for _ in range(len(lines)):
        if row >= len(lines):
            break
        if lines[row].strip().isdigit():
            num_atoms = int(lines[row])
            frame += 1
            row += 2  # skip comment line.
            for a in range(num_atoms):
                if row + a >= len(lines):
                    break
                parts = lines[row + a].strip().split()
                if len(parts) >= dimensions:
                    x, y, z = map(float, parts[0:3])
                    data.append({"frame": frame, "x": x, "y": y, "z": z})
            row += num_atoms
        else:
            row += 1

    return pd.DataFrame(data)
