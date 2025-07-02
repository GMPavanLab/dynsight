Dimensionality reduction methods 
================================

This recipe explains how to compute descriptors via dimensionality reduction
of a multivariated descriptor. These example use SOAP, but the same approaches
can be applied to a variety of other quantities. The newly computed
descriptors are always stored in an :class:`.trajectory.Insight` variable.

All the functions take as optional parameters one or more file paths; if these
paths are passed, the function, before computing some quantity from scratch,
tries to load it from file, in case it has already been previously computed
and saved.

Please remember that the SOAP dataset can be very large, due to the high
dimensionality, thus calculations can be expensive, and saving to/loading from
file quite slow.

Let's start by creating a :class:`.trajectory.Trj` object to use as a
starting point for the examples:

.. testcode:: recipe2-test

    from pathlib import Path
    from dynsight.trajectory import Trj

    # Loading an example trajectory
    files_path = Path("../tests/systems/")
    trj = Trj.init_from_xtc(
        traj_file=files_path / "balls_7_nvt.xtc",
        topo_file=files_path / "balls_7_nvt.gro",
    )

Principal Component Analysis (PCA)
----------------------------------

Principal Component Analysis is a dimensionality reduction method that finds
the combinations of components that maximize the data variance. More details
on the algorithm here https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html.

This function takes as input a :class:`.trajectory.Trj` and all the relevant
parameters, and performs the PCA of the corresponding SOAP dataset.

.. testcode:: recipe2-test

    from pathlib import Path
    from dynsight.trajectory import Trj, Insight
    from sklearn.decomposition import PCA

    def compute_soap_pca(
        trj: Trj,
        r_cut: float,
        n_max: int,
        l_max: int,
        n_components: int,
        soap_path: Path | None = None,
        pca_path: Path | None = None,
        selection: str = "all",
        centers: str = "all",
        respect_pbc: bool = True,
        n_core: int = 1,
    ) -> Insight:
        if pca_path is not None and pca_path.exists():
            soap_pca = Insight.load_from_json(pca_path)
        else:
            if soap_path is not None and soap_path.exists():
                soap = Insight.load_from_json(soap_path)
            else:
                soap = trj.get_soap(
                    r_cut=r_cut,
                    n_max=n_max,
                    l_max=l_max,
                    selection=selection,
                    centers=centers,
                    respect_pbc=respect_pbc,
                    n_core=n_core,
                )
                if soap_path is not None:
                    soap.dump_to_json(soap_path)

            n_atom, n_frames, n_dims = soap.dataset.shape
            reshaped_soap = soap.dataset.reshape(n_atom * n_frames, n_dims)
            pca = PCA(n_components=n_components)
            transformed_soap = pca.fit_transform(reshaped_soap)
            pca_ds = transformed_soap.reshape(n_atom, n_frames, -1)
            soap_pca = Insight(pca_ds)

            if pca_path is not None:
                soap_pca.dump_to_json(pca_path)

        return soap_pca

    # Example of how to use
    soap_pc1 = compute_soap_pca(
        trj=trj,
        r_cut=2.0,
        n_max=4,
        l_max=4,
        n_components=1,
    )



Other
-----

Notice that, differently from SOAP - which is computed for every frame, tSOAP
is computed for every pair of frames. Thus, the tSOAP dataset has shape 
``(n_particles, n_frames - 1)``. Consequently, if you need to match the tSOAP
values with the particles along the trajectory, you will need to use a sliced
trajectory (removing the last frame). The easiest way to do this is:

.. testcode:: recipe2-test

    trajslice = slice(0, -1, 1)
    shorter_trj = trj.with_slice(trajslice=trajslice)

.. testcode:: recipe2-test
    :hide:

    assert soap_pc1.dataset.shape == (7, 201, 1)
