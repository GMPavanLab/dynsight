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

    # Define trajectory parameters
    traj_name = "Octahedron"
    # Define r_cuts
    soap_r_cuts = [1.75, 2.0, 2.15, 2.3, 2.45, 2.60, 2.75]

    with h5py.File(input_file, "r") as file:
        group = file["Trajectories"][traj_name]
        universe = dynsight.hdf5er.create_universe_from_slice(group)

        # Run SOAP calculation for different r_cuts
        for i, r_c in enumerate(soap_r_cuts):
            test_soap = dynsight.soapify.saponify_trajectory(
                universe=universe,
                soaprcut=r_c,
                soaplmax=8,
                soapnmax=8,
            )

            _ = dynsight.soapify.fill_soap_vector_from_dscribe(
                test_soap,
            )

            # Define control and test SOAP calculation as numpy array
            tmp_soap = np.array(file[f"SOAP_{i}"][traj_name])
            check_soap = np.transpose(tmp_soap, (1, 0, 2))

            # Check if control and test array are similar
            assert np.allclose(check_soap, test_soap, atol=1e-8, rtol=1e-2), (
                f"SOAP analyses provided different values "
                f"compared to the control system "
                f"for r_cut: {r_c}."
            )
