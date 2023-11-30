import logging
import pathlib
import sys

import dynsight
import h5py
from MDAnalysis import Universe

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


def main() -> None:
    """Run the example.

    Example available for cpctools here:
    https://github.com/GMPavanLab/cpctools/blob/main/Examples/FileHandling.ipynb

    """
    # Getting the example trajectory.
    # Assuming that we are on linux, let's download the small example lammps
    # trajectory (it is a 55 gold atoms nanoparticle NVT at few temperatures):
    # wget https://github.com/GMPavanLab/SOAPify/releases/download/0.1.0rc0/SmallExample.zip
    # and unzip it
    # unzip SmallExample.zip
    first_line = f"Usage: {__file__}.py data_directory"
    num_args = 2
    if len(sys.argv) != num_args:
        logging.info(first_line)
        sys.exit()
    else:
        data_directory = sys.argv[1]

    data_path = pathlib.Path(data_directory)

    examplehdf5 = data_path / "ih55.hdf5"
    examplesoaphdf5 = data_path / "ih55soap.hdf5"

    # Let's create the base .hdf5 file with the trajectory.
    extra_attributes = {
        "ts": "5fs",
        "pair_style": "smatb/single",
        "pair_coeff": (
            "1 1 2.88 10.35 4.178 0.210 1.818 4.07293506 4.9883063257983666"
        ),
    }
    for time in [100, 200]:
        u = Universe(
            data_path / "ih55.data",
            [data_path / f"ih55-T_{time}.lammpsdump"],
            atom_style="id type x y z",
        )
        u.atoms.types = ["Au"] * len(u.atoms)
        dynsight.hdf5er.mda_to_hdf5(
            mdatrajectory=u,
            targethdf5file=examplehdf5,
            groupname=f"SmallExample_{time}",
            trajchunksize=1000,
            attrs=extra_attributes,
        )

    # The attributes are then accessible, and can be used to our advantage (or
    # to reproduce the simulations)
    with h5py.File(examplehdf5, "r") as work_file:
        trjcontainers = work_file["Trajectories"]
        for name, trjgroup in trjcontainers.items():
            logging.info(f"Trajectory group name: {name}")
            logging.info("Attributes:")
            for attname, attval in trjgroup.attrs.items():
                logging.info(f'\t{attname}: "{attval}"')

    # Applying SOAP.
    # Then let's calculate the SOAP fingerprints using dscribe to the all of
    # the trajectories in the file.
    for soap_l_max in [4]:
        with h5py.File(examplehdf5, "r") as workfile, h5py.File(
            examplesoaphdf5, "a"
        ) as soapfile:
            dynsight.soapify.saponify_multiple_trajectories(
                trajcontainers=workfile["Trajectories"],
                soapoutcontainers=soapfile.require_group(
                    f"SOAP{soap_l_max}_4_4 "
                ),
                soapoutputchunkdim=1000,
                verbose=False,
                soapnjobs=16,
                soaprcut=4.48023312,
                soapnmax=4,
                soaplmax=soap_l_max,
            )

    # The information about the SOAP calculation are stored in the attributes
    # of the SOAP fingerprint datasets:
    with h5py.File(examplesoaphdf5, "r") as workfile:
        for name, trjgroup in workfile.items():
            logging.info(f'SOAP group name: "{name}"')
            logging.info("Attributes:")
            for dsname, trjds in trjgroup.items():
                logging.info(f"\tSOAP dataset: {dsname}, shape {trjds.shape}")
                for attname, attval in trjds.attrs.items():
                    logging.info(f'\t\t{attname}: "{attval}"')

    # Selecting a trajectory
    # You can calculate SOAP only for a single trajectory:
    with h5py.File(examplehdf5, "r") as workfile, h5py.File(
        examplesoaphdf5, "a"
    ) as soapfile:
        dynsight.soapify.saponify_trajectory(
            trajcontainer=workfile["/Trajectories/SmallExample_200"],
            soapoutcontainer=soapfile.require_group("SOAP_6_6_4"),
            soapoutputchunkdim=1000,
            soapnjobs=16,
            soaprcut=4.48023312,
            soapnmax=6,
            soaplmax=6,
            verbose=False,
        )

    # Fingerprints for a subsystem.
    # You can also calculate the soap fingerprints of a subgroup of atoms.
    # Here, for example we will calculate the SOAP fingerprints of only the
    # 0th and the 15th atoms.
    with h5py.File(examplehdf5, "r") as workfile, h5py.File(
        examplesoaphdf5, "a"
    ) as soapfile:
        dynsight.soapify.saponify_trajectory(
            trajcontainer=workfile["/Trajectories/SmallExample_200"],
            soapoutcontainer=soapfile.require_group("SOAP_4_4_4_FEW"),
            centersmask=[0, 15],
            soapoutputchunkdim=1000,
            soapnjobs=16,
            soaprcut=4.48023312,
            soapnmax=4,
            soaplmax=4,
            verbose=False,
        )

    # Note the the new attribute centersIndexes and the different shape of the
    # dataset that reflects that SOAP fingerprints have been calculated only
    # for atom 0 and 15:
    with h5py.File(examplesoaphdf5, "r") as workfile:
        name = "SOAP_4_4_4_FEW"
        trjgroup = workfile[name]
        logging.info(f'SOAP group name: "{name}"')
        logging.info("Attributes:")
        for dsname, trjds in trjgroup.items():
            logging.info(f"\tSOAP dataset: {dsname}, shape {trjds.shape}")
            for attname, attval in trjds.attrs.items():
                logging.info(f'\t\t{attname}: "{attval}"')

    # You can call getSOAPSettings on a SOAP dataset to get the necessary data
    # to 'fill' the vector (aka restore the repetition in the data that have
    # been removed to save space) by simpy passing the returned dictionary to
    # fillSOAPVectorFromdscribe.
    with h5py.File(examplesoaphdf5, "r") as workfile:
        for name, trjgroup in workfile.items():
            logging.info(f'SOAP group name: "{name}"')
            for dsname, trjds in trjgroup.items():
                logging.info(f"\tSOAP dataset: {dsname}, shape {trjds.shape}:")
                fillinfo = dynsight.soapify.get_soap_settings(trjds)
                logging.info(f"\t{fillinfo}")
                example = dynsight.soapify.fill_soap_vector_from_dscribe(
                    soapfromdscribe=trjds[:],
                    lmax=fillinfo["nMax"],
                    nmax=fillinfo["lMax"],
                    atomtypes=fillinfo["atomTypes"],
                    atomicslices=fillinfo["atomicSlices"],
                )
                logging.info(f"\tFilled shape : {example.shape}")


if __name__ == "__main__":
    main()
