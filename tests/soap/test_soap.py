from pathlib import Path

import MDAnalysis
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
        - tests/systems/octahedron.xyz

    Dynsight function tested:
        - dynsight.soap.saponify_trajectory()
            - soaplmax = 8
            - soapnmax = 8

    r_cuts checked:
        - [1.75, 2.0, 2.15, 2.3, 2.45, 2.60, 2.75]
    """
    # Define input and output files
    original_dir = Path(__file__).absolute().parent
    input_file = original_dir / "../systems/octahedron.xyz"

    # Define r_cuts
    soap_r_cuts = [1.75, 2.0, 2.15, 2.3, 2.45, 2.60, 2.75]
    check_file = np.load(original_dir / "../systems/SOAP.npz")

    universe = MDAnalysis.Universe(input_file, dt=1)

    # Run SOAP calculation for different r_cuts
    for i, r_c in enumerate(soap_r_cuts):
        test_soap = dynsight.soap.saponify_trajectory(
            universe=universe,
            soaprcut=r_c,
            soaplmax=8,
            soapnmax=8,
        )

        _ = dynsight.soap.fill_soap_vector_from_dscribe(
            test_soap[0][0],
        )
        _ = dynsight.soap.fill_soap_vector_from_dscribe(
            test_soap[0],
        )
        _ = dynsight.soap.fill_soap_vector_from_dscribe(
            test_soap,
        )

        check_soap = check_file[f"arr{i + 1}"]

        # Check if control and test array are similar
        assert np.allclose(check_soap, test_soap, atol=1e-6, rtol=1e-2), (
            f"SOAP analyses provided different values "
            f"compared to the control system "
            f"for r_cut: {r_c}."
        )
