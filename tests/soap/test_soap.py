from pathlib import Path

import h5py
import numpy as np

import dynsight


def test_soap_vectors() -> None:
    """Test the consistency of SOAP calculations with a control calculation.

    This test verifies that the SOAP calculation yields similar
    values as a control calculation at different r_cut. The calculation of SOAP
    is influenced by the architecture of the machine it's run on. As a result,
    the values of the SOAP components might exhibit minor variations.
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
    output_file = original_dir / "../octahedron_test.hdf5"

    # Define trajectory parameters
    traj_name = "Octahedron"
    # Define r_cuts
    soap_r_cuts = [1.75, 2.0, 2.15, 2.3, 2.45, 2.60, 2.75]

    # Run SOAP calculation for different r_cuts
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
            # Define control and test SOAP calculation as numpy array
            check_soap = np.array(work_file[f"SOAP_{i}"][traj_name])
            test_soap = np.array(
                out_file[f"SOAP_test_{soap_r_cuts[i]}"][traj_name]
            )

            # Check if control and test array are similar
            assert np.allclose(check_soap, test_soap, atol=1e-2, rtol=1e-2), (
                f"SOAP analyses provided different values "
                f"compared to the control system "
                f"for r_cut: {soap_r_cuts[i]} (results: {output_file})."
            )
        # If test passed remove test_soap array from test folder
        output_file.unlink()
