from pathlib import Path

import dynsight
import h5py
import numpy as np

"""
Test description:tests if a tSOAP calculation yields the same
                    values as a control calculation at different r_cut.

Control file path: tests/systems/octahedron.hdf5

Dynsyght function tested: dynsight.time_soap.timesoap()
SOAP calculation parameters
                            --> soaplmax = 8
                            --> soapnmax = 8

r_cuts checked: 1.75, 2.0, 2.15, 2.3, 2.45, 2.60, 2.75 (7)
"""


def test_time_soap_vectors() -> None:
    # i/o files
    input_file = "tests/systems/octahedron.hdf5"
    output_file = "tests/systems/octahedron_test_tsoap.hdf5"
    # number of SOAP calulation made in octahedron test
    n_soap_rcuts = 7
    traj_name = "Octahedron"

    # tSOAP calculation for different r_cuts
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
            out_file.create_group(f"delta_timeSOAP_test{i}")
            out_file[f"delta_timeSOAP_test{i}"].create_dataset(
                f"delta_timeSOAP_test{i}", data=delta_time_soap
            )
            # control tSOAP calculation (timed and delta time) to numpy array
            check_timed_soap = np.array(
                in_file[f"timeSOAP_{i}"][f"timeSOAP_{i}"]
            )
            check_delta_time_soap = np.array(
                in_file[f"delta_timeSOAP_{i}"][f"delta_timeSOAP_{i}"]
            )

            # check if control and test array are equal
            assert np.allclose(
                timed_soap, check_timed_soap, atol=1e-5, rtol=1e-5
            )
            assert np.allclose(
                delta_time_soap, check_delta_time_soap, atol=1e-5, rtol=1e-5
            )
    # if test passed remove test_soap array from test folder
    Path(output_file).unlink()
