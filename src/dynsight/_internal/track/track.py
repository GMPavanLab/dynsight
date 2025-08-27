"""dynsight.track module for particle tracking from an .xyz file."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import trackpy as tp

from dynsight.trajectory import Trj
from dynsight.utilities import read_xyz

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def track_xyz(
    input_xyz: Path,
    output_xyz: Path,
    search_range: float,
    memory: int = 1,
    adaptive_stop: None | float = 0.95,
    adaptive_step: None | float = 0.5,
) -> Trj:
    """Track particles from an ``.xyz`` file and write a new file with IDs.

    The input ``.xyz`` is assumed to contain only raw 3D coordinates
    (without atom labels/identity), and each frame begins with a line
    indicating the number of objects, followed by a comment line, then a list
    of positions. Each frame in the input file must follow this structure::

        <number of objects>
        comment line
        <x> <y> <z>
        <x> <y> <z>
        ...
        <x> <y> <z>

    or::

        <number of objects>
        comment line
        <name> <x> <y> <z>
        <name> <x> <y> <z>
        ...
        <name> <x> <y> <z>

    The output file will have the same structure, but each line will start
    with the tracked particle ID.

    Parameters:
        input_xyz:
            Path to the input .xyz file containing positions only.

        output_xyz:
            Path where the output .xyz file with particle IDs will be saved.

        search_range:
            The maximum allowable displacement of objects between frames for
            them to be considered the same particle. Units depend on the
            coordinate system used in the input file. If the file comes from
            vision, then the unit is ``pixels``. We do not provide a default
            parameter here, because fine tuning is required to get good
            behaviour. We recommend starting with a value around 2-3 times the
            diameter of the particles. But if you are unsure, start with a
            value of 10. Additionally, test on a small trajectory to start
            with.

        memory:
            The maximum number of frames during which an object can vanish,
            then re-appear nearby, and be considered the same particle.

        adaptive_stop:
            If not `None`, when encountering a region with too many candidate
            links (subnet), retry by progressively reducing `search_range`
            until the subnet is solvable. If `search_range` becomes less or
            equal than the `adaptive_stop`, give up and raise a
            `SubnetOversizeException`.

        adaptive_step:
            Factor by which the `search_range` is multiplied to reduce it
            during adaptive search. Effective only if `adaptive_stop` is not
            `None`.
    """
    if adaptive_stop is None and adaptive_step is not None:
        msg = "adaptive_step is set but adaptive_stop is None."
        raise ValueError(msg)
    if adaptive_stop is not None and adaptive_step is None:
        msg = "adaptive_stop is set but adaptive_step is None."
        raise ValueError(msg)

    input_xyz = Path(input_xyz)
    output_xyz = Path(output_xyz)

    if not input_xyz.exists():
        msg = f"Input file not found: {input_xyz}"
        raise FileNotFoundError(msg)

    positions = read_xyz(
        input_xyz=input_xyz, cols_order=["name", "x", "y", "z"]
    )

    if not {"frame", "x", "y", "z"}.issubset(positions.columns):
        msg = (
            "Error in the .xyz format. Each line must be "
            "<x> <y> <z> or <name> <x> <y> <z>."
        )
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
            frame_data = linked[linked["frame"] == frame_num].sort_values(
                "particle"
            )
            f.write(f"{len(frame_data)}\n")
            f.write(f"Frame {frame_num}\n")
            for _, row in frame_data.iterrows():
                pid = int(row["particle"])
                x, y, z = row["x"], row["y"], row["z"]
                name = row.get("name")
                if name is not None and pd.notna(name):
                    f.write(f"{name} {x:.6f} {y:.6f} {z:.6f} {pid}\n")
                else:
                    f.write(f"{x:.6f} {y:.6f} {z:.6f} {pid}\n")

    logger.info(f"Linked .xyz file written to: {output_xyz}")
    return Trj.init_from_xyz(traj_file=output_xyz, dt=1)


def _collect_positions(input_xyz: Path) -> pd.DataFrame:
    """Read the xyz file and return the positions dataset at each frame."""
    lines = input_xyz.read_text().splitlines()

    data: list[dict[str, object]] = []
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
                if len(parts) == dimensions:
                    x, y, z = map(float, parts[0:3])
                    data.append({"frame": frame, "x": x, "y": y, "z": z})
                elif len(parts) > dimensions:
                    name = parts[0]
                    x, y, z = map(float, parts[1:4])
                    data.append(
                        {
                            "frame": frame,
                            "name": name,
                            "x": x,
                            "y": y,
                            "z": z,
                        }
                    )
                else:
                    msg = (
                        "Invalid line format, expected 3 or 4 columns, "
                        f"found {len(parts)}"
                    )
                    raise ValueError(msg)
            row += num_atoms
        else:
            row += 1

    return pd.DataFrame(data)
