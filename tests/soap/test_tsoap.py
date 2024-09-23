from pathlib import Path

import h5py
import numpy as np

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
        - dynsight.soapify.saponify_trajectory()
            - soaplmax = 8
            - soapnmax = 8

    r_cuts checked:
        - [1.75, 2.0, 2.15, 2.3, 2.45, 2.60, 2.75]
    """
    # Define input and output files
    original_dir = Path(__file__).absolute().parent
    input_file = original_dir / "../systems/octahedron.hdf5"
    output_file = original_dir / "../octahedron_test_tsoap.hdf5"

    # Define the number of SOAP calulation made in the octahedron test
    n_soap_rcuts = 7
    traj_name = "Octahedron"

    # Run tSOAP calculation for different r_cuts
    with h5py.File(input_file, "r") as in_file, h5py.File(
        output_file, "w"
    ) as out_file:
        for i in range(n_soap_rcuts):
            soap_traj = in_file[f"SOAP_{i}"][traj_name]
            timed_soap, delta_time_soap = dynsight.time_soap.timesoap(
                soaptrajectory=soap_traj
            )
            out_file.create_group(f"timeSOAP_test{i}")
            out_file[f"timeSOAP_test{i}"].create_dataset(
                f"timeSOAP_test{i}", data=timed_soap
            )
            # Define control and test tSOAP calculations as numpy array
            check_timed_soap = np.array(
                in_file[f"timeSOAP_{i}"][f"timeSOAP_{i}"]
            )
            check_delta_time_soap = np.array(
                in_file[f"delta_timeSOAP_{i}"][f"delta_timeSOAP_{i}"]
            )

            # Check if control and test array are similar
            assert np.allclose(
                timed_soap, check_timed_soap, atol=1e-11, rtol=1e-11
            )
            assert np.allclose(
                delta_time_soap, check_delta_time_soap, atol=1e-11, rtol=1e-11
            )
    # If test passed remove test_soap array from test folder
    output_file.unlink()
