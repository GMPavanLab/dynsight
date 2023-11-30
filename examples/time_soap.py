from __future__ import annotations

import logging
import pathlib
import sys

import dynsight
import h5py
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


def preparesoap(
    trajfilename: pathlib.Path,
    trajaddress: str,
    soaprcut: float,
    soapnmax: int,
    soaplmax: int,
) -> str:
    soapfilename = str(trajfilename).split(".")[0] + "soap.hdf5"
    logging.info(f"{trajfilename} -> {soapfilename}")
    with h5py.File(trajfilename, "r") as workfile, h5py.File(
        soapfilename, "a"
    ) as soapfile:
        soapfile.require_group("SOAP")
        # skips if the soap trajectory is already present
        if trajaddress not in soapfile["SOAP"]:
            dynsight.soapify.saponify_trajectory(
                trajcontainer=workfile[f"Trajectories/{trajaddress}"],
                soapoutcontainer=soapfile["SOAP"],
                soapoutputchunkdim=1000,
                soapnjobs=32,
                soaprcut=soaprcut,
                soapnmax=soapnmax,
                soaplmax=soaplmax,
            )
    return soapfilename


def gettimesoap(
    soapfilename: str,
    trajaddress: str,
) -> tuple[int, np.ndarray, np.ndarray]:  # type: ignore[type-arg]
    with h5py.File(soapfilename, "r") as f:
        ds = f[f"/SOAP/{trajaddress}"]
        fillsettings = dynsight.soapify.get_soap_settings(ds)
        logging.info(fillsettings)
        logging.info(ds.shape)
        nat = ds.shape[1]

        timedsoap = np.zeros((ds.shape[0] - 1, ds.shape[1]))

        logging.info(timedsoap.shape)
        slide = 0
        # this looks a lot convoluted, but it is way faster than working one
        # atom at a time
        for c in ds.iter_chunks():
            theslice = slice(c[0].start - slide, c[0].stop, c[0].step)
            outslice = slice(c[0].start - slide, c[0].stop - 1, c[0].step)
            logging.info(f"{c[0]}, {theslice}, {outslice}")
            timedsoap[outslice], _ = dynsight.time_soap.timesoapsimple(
                dynsight.utilities.normalize_array(
                    dynsight.soapify.fill_soap_vector_from_dscribe(
                        soapfromdscribe=ds[theslice],
                        lmax=fillsettings["nMax"],
                        nmax=fillsettings["lMax"],
                        atomtypes=fillsettings["atomTypes"],
                        atomicslices=fillsettings["atomicSlices"],
                    )
                )
            )
            slide = 1

        return nat, timedsoap, np.diff(timedsoap.T, axis=-1)


def main() -> None:
    """Run the example.

    Example available for cpctools here:
    https://github.com/GMPavanLab/cpctools/blob/main/Examples/timeSOAP.ipynb

    Paper: https://arxiv.org/abs/2302.09673v2

    """
    # Let's get the data:
    # wget https://github.com/GMPavanLab/dynNP/releases/download/V1.0-trajectories/ico309.hdf5
    # We'll start by caclulating the SOAP fingerprints of the 500 K simulation
    # using lMax=8, nMax=8, and cutoff=4.48023312 that is 10% more than the Au
    # cell
    first_line = f"Usage: {__file__}.py data_directory"
    num_args = 2
    if len(sys.argv) != num_args:
        logging.info(first_line)
        sys.exit()
    else:
        data_directory = sys.argv[1]

    data_path = pathlib.Path(data_directory)

    trajfilename = data_path / "ico309.hdf5"
    trajaddress = "ico309-SV_18631-SL_31922-T_500"
    soapfilename = preparesoap(
        trajfilename,
        trajaddress,
        soaprcut=4.48023312,
        soapnmax=8,
        soaplmax=8,
    )

    # Let's get the raw tempSOAP array without clogging up the memory.
    # There is an improved version of the action in shis cell in the
    # analysis submodule: analysis.getTimeSOAPSimple
    natoms, tsoap, dtsoap = gettimesoap(soapfilename, trajaddress)
    logging.info(tsoap.shape)
    logging.info(dtsoap.shape)

    fig, axes = plt.subplots(2, sharey=True)
    for i in range(4):
        axes[0].plot(tsoap[:, i], label=f"Atom {i}")
        axes[1].plot(tsoap[:, natoms - 1 - i], label=f"Atom {natoms-1-i}")

    for ax in axes:
        ax.legend()
    fig.tight_layout()
    fig.savefig(
        "tsoap.png",
        dpi=360,
        bbox_inches="tight",
    )
    plt.close()

    fig, axes = plt.subplots(2, sharey=True)
    for i in range(4):
        axes[0].plot(dtsoap[i], label=f"Atom {i}")
        axes[1].plot(dtsoap[308 - i], label=f"Atom {308-i}")

    for ax in axes:
        ax.legend()
    fig.tight_layout()
    fig.savefig(
        "dtsoap.png",
        dpi=360,
        bbox_inches="tight",
    )
    plt.close()


if __name__ == "__main__":
    main()
