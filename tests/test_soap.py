from pathlib import Path

import dynsight
import h5py
import numpy as np

"""
Test description:tests if a SOAP calculation yields the same
                    values as a control calculation at different r_cut.

Control file path: tests/systems/octahedron.hdf5

Dynsyght function tested: dynsight.soapify.saponify_trajectory()
                            --> soaplmax = 8
                            --> soapnmax = 8

r_cuts checked: 1.75, 2.0, 2.15, 2.3, 2.45, 2.60, 2.75
"""


def test_soap_vectors() -> None:
    # i/o files
    input_file = "tests/systems/octahedron.hdf5"
    output_file = "tests/systems/octahedron_test.hdf5"

    # trajectory name
    traj_name = "Octahedron"
    # r_cuts
    soap_r_cuts = [1.75, 2.0, 2.15, 2.3, 2.45, 2.60, 2.75]

    # SOAP calculation for different r_cuts
    with h5py.File(input_file, "r") as work_file, h5py.File(
        output_file, "a"
    ) as out_file:
        for i in range(len(soap_r_cuts)):
            dynsight.soapify.saponify_trajectory(
                trajcontainer=work_file["Trajectories"][traj_name],
                soapoutcontainer=out_file.require_group(
                    f"SOAP_test_{soap_r_cuts[i]}"
                ),
                soaprcut=soap_r_cuts[i],
                soaplmax=8,
                soapnmax=8,
                dooverride=True,
                verbose=False,
            )

            # control and test SOAP calculation to numpy array
            check_soap = np.array(work_file[f"SOAP_{i}"][traj_name])
            test_soap = np.array(
                out_file[f"SOAP_test_{soap_r_cuts[i]}"][traj_name]
            )

            # check if control and test array are equal
            assert np.allclose(check_soap, test_soap, atol=1e-1), (
                f"SOAP analyses provided different values "
                f"compared to the control system "
                f"for r_cut: {soap_r_cuts[i]} (results: {output_file})."
            )

        # if test passed remove test_soap array from test folder
        Path(output_file).unlink()
