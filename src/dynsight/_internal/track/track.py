import logging
from pathlib import Path

import pandas as pd
import trackpy as tp

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
) -> None:
    """Track particles from an xyz file and write a new .xyz with particle IDs.

    The input .xyz is assumed to contain only raw 3D coordinates
    (no atom labels), and each frame begins with a line indicating the number
    of atoms, followed by a comment line, then a list of positions.

    Example frame block:
        100
        Frame 0
        1.0 2.0 3.0
        2.1 1.9 3.5
        ...

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
        Maximum number of frames during which a particle can vanish and
        still be tracked.
    adaptive_step:
        Step size for adaptive linking.
    adaptive_stop:
        Stopping criterion for adaptive linking.

    Raises:
    ------
    FileNotFoundError
        If the input file does not exist.
    ValueError
        If the input format is incorrect or missing required columns.
    """
    input_xyz = Path(input_xyz)
    output_xyz = Path(output_xyz)

    if not input_xyz.exists():
        msg = f"Input file not found: {input_xyz}"
        raise FileNotFoundError(msg)

    lines = input_xyz.read_text().splitlines()

    data = []
    frame = -1
    i = 0
    dimensions = 3
    while i < len(lines):
        if lines[i].strip().isdigit():
            num_atoms = int(lines[i])
            frame += 1
            i += 2  # Skip comment line
            for j in range(num_atoms):
                parts = lines[i + j].strip().split()
                if len(parts) >= dimensions:
                    x, y, z = map(float, parts[0:3])
                    data.append({"frame": frame, "x": x, "y": y, "z": z})
            i += num_atoms
        else:
            i += 1

    positions = pd.DataFrame(data)

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
