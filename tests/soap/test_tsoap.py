from pathlib import Path

import h5py
import numpy as np
import pytest

import dynsight


def test_time_soap_vectors() -> None:
    """Test the consistency of tSOAP calculations with a control calculation.

    This test verifies that the tSOAP calculation yields similar
    values as a control calculation at different r_cut. The calculation of SOAP
    (and consequently tSOAP) is influenced by the architecture of the machine
    it's run on. As a result, the values of the SOAP components might exhibit
    minor variations.
    To disregard these differences, the function np.allclose() is employed.

    Control file path:
        - tests/systems/octahedron.hdf5

    Dynsight function tested:
        - dynsight.soap.timesoap()

    r_cuts checked:
        - [1.75, 2.0, 2.15, 2.3, 2.45, 2.60, 2.75]
    """
    # Define input and output files
    original_dir = Path(__file__).absolute().parent
    input_file = original_dir / "../systems/octahedron.hdf5"

    # Define the number of SOAP calulation made in the octahedron test
    traj_name = "Octahedron"
    # Define r_cuts
    soap_r_cuts = [1.75, 2.0, 2.15, 2.3, 2.45, 2.60, 2.75]

    # Run tSOAP calculation for different r_cuts
    with h5py.File(input_file, "r") as file:
        group = file["Trajectories"][traj_name]
        universe = dynsight.hdf5er.create_universe_from_slice(group)

        for i, r_c in enumerate(soap_r_cuts):
            soap_traj = dynsight.soap.saponify_trajectory(
                universe=universe,
                soaprcut=r_c,
            )

            with pytest.raises(ValueError, match="delay value outside bounds"):
                test_tsoap = dynsight.soap.timesoap(
                    soaptrajectory=soap_traj, delay=20
                )

            test_tsoap = dynsight.soap.timesoap(soaptrajectory=soap_traj)

            check_tsoap = np.array(file[f"timeSOAP_{i}"][f"timeSOAP_{i}"])
            check_tsoap = np.transpose(check_tsoap, (1, 0))

            # Check if control and test array are similar
            assert np.allclose(test_tsoap, check_tsoap, atol=1e-8, rtol=1e-2)
