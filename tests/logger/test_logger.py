from __future__ import annotations

import zipfile
from pathlib import Path

from dynsight.logs import logger
from dynsight.trajectory import Trj


def test_zip_arch() -> None:
    """Ensure the zip archive is created and contains the expected files."""
    logger.clear_history()

    original_dir = Path(__file__).absolute().parent
    lens_trajectory_file = original_dir / "../systems/2_particles.xyz"
    tsoap_topology_file = original_dir / "../systems/balls_7_nvt.gro"
    tsoap_trajectory_file = original_dir / "../systems/balls_7_nvt.xtc"

    lens_traj = Trj.init_from_xyz(lens_trajectory_file, dt=1)
    tsoap_traj = Trj.init_from_xtc(tsoap_trajectory_file, tsoap_topology_file)

    lens_traj.get_lens(r_cut=4)
    tsoap_traj.get_timesoap(r_cut=4, n_max=4, l_max=4)

    logger.extract_datasets(original_dir / "archive")

    zip_files = list(original_dir.glob("*.zip"))
    assert zip_files, f"No zip archive found in {original_dir}"

    zip_path = zip_files[0]

    try:
        assert zip_path.exists(), f"Archive not found: {zip_path}"

        with zipfile.ZipFile(zip_path, "r") as zf:
            names = [
                Path(name).stem
                for name in zf.namelist()
                if not name.endswith("/")
            ]
            n_insight_computed = 3
            assert len(names) == n_insight_computed, (
                f"Expected {n_insight_computed}"
                f" files in archive, found {len(names)}: {names}"
            )
            expected = {
                "lens_4_1_all_all",
                "soap_4_4_4_True_all_all",
                "timesoap_4_4_4_True_all_all_1",
            }
            assert set(names) == expected, f"Unexpected file names: {names}"

    finally:
        if zip_path.exists():
            zip_path.unlink(missing_ok=True)
        logger.clear_history()


def test_disabling_auto_record() -> None:
    """Ensure no zip archive is created if auto_recording is disabled."""
    logger.clear_history()
    logger.configure(auto_recording=False)

    original_dir = Path(__file__).absolute().parent
    lens_trajectory_file = original_dir / "../systems/2_particles.xyz"
    tsoap_topology_file = original_dir / "../systems/balls_7_nvt.gro"
    tsoap_trajectory_file = original_dir / "../systems/balls_7_nvt.xtc"

    lens_traj = Trj.init_from_xyz(lens_trajectory_file, dt=1)
    tsoap_traj = Trj.init_from_xtc(tsoap_trajectory_file, tsoap_topology_file)

    lens_traj.get_lens(r_cut=4)
    tsoap_traj.get_timesoap(r_cut=4, n_max=4, l_max=4)

    logger.extract_datasets(original_dir / "archive")

    zip_files = list(original_dir.glob("*.zip"))
    assert not zip_files, f"Unexpected zip archive(s) found: {zip_files}"

    logger.clear_history()
