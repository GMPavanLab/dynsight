from pathlib import Path

import pandas as pd
import trackpy as tp


def track_xyz(
    input_xyz: Path,
    output_xyz: Path,
    search_range: int = 5,
    memory: int = 1,
    adaptive_stop: float = 0.5,
    adaptive_step: float = 0.95,
):
    """Load a .xyz file (positions only), perform particle tracking using trackpy,
    and write a new .xyz file with particle IDs.

    Args:
        input_xyz (Path): Path to the input .xyz file.
        output_xyz (Path): Path to the output .xyz file with particle IDs.
        search_range (float): Max distance allowed for linking particles between frames.
        memory (int): Max number of frames a particle can disappear and still be linked.
    """
    input_xyz = Path(input_xyz)
    output_xyz = Path(output_xyz)

    if not input_xyz.exists():
        raise FileNotFoundError(f"Input file not found: {input_xyz}")

    lines = input_xyz.read_text().splitlines()

    data = []
    frame = -1
    i = 0
    while i < len(lines):
        if lines[i].strip().isdigit():
            num_atoms = int(lines[i])
            frame += 1
            i += 2  # Skip comment line
            for j in range(num_atoms):
                parts = lines[i + j].strip().split()
                if len(parts) >= 4:
                    x, y, z = map(float, parts[1:4])
                    data.append({"frame": frame, "x": x, "y": y, "z": z})
            i += num_atoms
        else:
            i += 1

    df = pd.DataFrame(data)

    # Perform particle linking
    linked = tp.link_df(
        df,
        search_range=search_range,
        memory=memory,
        adaptive_step=adaptive_step,
        adaptive_stop=adaptive_stop,
    )

    # Write the output .xyz with IDs
    with output_xyz.open("w") as f:
        for frame_num in sorted(linked["frame"].unique()):
            frame_data = linked[linked["frame"] == frame_num]
            f.write(f"{len(frame_data)}\n")
            f.write(f"Frame {frame_num}\n")
            for _, row in frame_data.iterrows():
                atom_type = "X"  # Generic atom label
                f.write(
                    f"{atom_type} {row['x']:.6f} {row['y']:.6f} {row['z']:.6f} id={int(row['particle'])}\n"
                )

    print(f"Linked .xyz file written to: {output_xyz}")
