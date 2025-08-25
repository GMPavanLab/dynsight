from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Iterator, TextIO

import numpy as np

from dynsight.track import track_xyz

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _iter_xyz_frames(f: TextIO) -> Iterator[list[str]]:
    for line in f:
        header = line.strip()
        if not header:
            continue
        natoms = int(header)
        comment = f.readline()
        if comment == "":
            raise ValueError
        atoms = [f.readline() for _ in range(natoms)]
        if any(a == "" for a in atoms):
            raise ValueError
        yield atoms


def _parse_atom_line(line: str) -> tuple[float, float, float, str]:
    parts = line.split()
    cols = 5
    if len(parts) < cols:
        raise ValueError
    x, y, z = map(float, parts[1:4])
    at_id = parts[4]
    return x, y, z, at_id


def _coords_by_id_from_xyz(
    path: Path | str, target_id: int | str, *, strict: bool = False
) -> NDArray[np.float64]:
    target = str(target_id)
    out: list[tuple[float, float, float]] = []

    with Path(path).open("r", encoding="utf-8") as f:
        for atoms in _iter_xyz_frames(f):
            xyz = (np.nan, np.nan, np.nan)
            for atom_line in atoms:
                x, y, z, at_id = _parse_atom_line(atom_line)
                if at_id == target:
                    xyz = (x, y, z)
                    break
            if strict and np.isnan(xyz[0]):
                raise ValueError
            out.append(xyz)

    if not out:
        raise ValueError

    return np.asarray(out, dtype=float)


def test_track_xyz(tmp_path: Path) -> None:
    original_dir = Path(__file__).resolve().parent
    filename = original_dir / "../systems/lj_noid.xyz"
    file_with_id = original_dir / "../systems/lj_id.xyz"
    output = tmp_path / "trajectory.xyz"

    track_xyz(input_xyz=filename, output_xyz=output)
    for i in range(5):
        arr1 = _coords_by_id_from_xyz(output, target_id=i)
        arr2 = _coords_by_id_from_xyz(file_with_id, target_id=i + 1)
        assert arr1.shape == arr2.shape
        assert np.allclose(arr1, arr2)
