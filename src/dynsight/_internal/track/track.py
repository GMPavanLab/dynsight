from pathlib import Path

import pandas as pd
import trackpy as tp


def track_xyz(input_xyz: Path, output_xyz: Path, search_range=5, memory=0):
    """Reads a minimalist .xyz file (positions only, no atom labels), tracks particles using trackpy,
    and writes a new .xyz file with IDs as the first column.

    Args:
        input_xyz (Path): Path to the input .xyz file.
        output_xyz (Path): Path to the output .xyz file with particle IDs as first column.
        search_range (float): Max linking distance between frames.
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
                if len(parts) >= 3:
                    x, y, z = map(float, parts[0:3])
                    data.append({"frame": frame, "x": x, "y": y, "z": z})
            i += num_atoms
        else:
            i += 1

    df = pd.DataFrame(data)

    if not {"frame", "x", "y", "z"}.issubset(df.columns):
        raise ValueError(
            "Parsed DataFrame missing required columns: frame, x, y, z"
        )

    linked = tp.link_df(df, search_range=search_range, memory=memory)

    with output_xyz.open("w") as f:
        for frame_num in sorted(linked["frame"].unique()):
            frame_data = linked[linked["frame"] == frame_num]
            f.write(f"{len(frame_data)}\n")
            f.write(f"Frame {frame_num}\n")
            for _, row in frame_data.iterrows():
                f.write(
                    f"{int(row['particle'])} {row['x']:.6f} {row['y']:.6f} {row['z']:.6f}\n"
                )

    print(f"Linked .xyz file written to: {output_xyz}")
