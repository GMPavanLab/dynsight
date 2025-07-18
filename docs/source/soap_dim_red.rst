Dimensionality reduction methods 
================================

This recipe explains how to compute descriptors via dimensionality reduction
of a multivariated descriptor. These example use SOAP, but the same approaches
can be applied to a variety of other quantities. The newly computed
descriptors are always stored in an :class:`.trajectory.Insight` variable.

All the functions take as optional parameters one or more file paths; if these
paths are passed, the function, before computing some quantity from scratch,
tries to load it from file, in case it has already been previously computed
and saved. For SOAP, which is required in all the examples, we use a function
from dynsight.utilities.

.. warning::

    Please consider that the SOAP dataset can be very large, due to the high
    dimensionality, thus calculations can be expensive, and saving to/loading 
    from file quite slow.

.. warning::

    This code works when run from the ``/docs`` directory of the ``dynsight`` 
    repo. To use it elsewhere, you have to change the ``Path`` variables
    accordingly.

First of all, we import all the packages and objects we'll need:

.. testcode:: recipe2-test

    from pathlib import Path
    import dynsight
    from dynsight.trajectory import Trj, Insight
    from dynsight.utilities import load_or_compute_soap
    from sklearn.decomposition import PCA

Let's start by creating a :class:`.trajectory.Trj` object to use as a
starting point for the examples:

.. testcode:: recipe2-test

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

``n_components`` is the number of PCs that the function stores in the output.

.. testcode:: recipe2-test

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
            return Insight.load_from_json(pca_path)

        soap = load_or_compute_soap(
            trj=trj,
            r_cut=r_cut,
            n_max=n_max,
            l_max=l_max,
            selection=selection,
            centers=centers,
            respect_pbc=respect_pbc,
            n_core=n_core,
            soap_path=soap_path,
        )

        n_atom, n_frames, n_dims = soap.dataset.shape
        reshaped_soap = soap.dataset.reshape(n_atom * n_frames, n_dims)
        pca = PCA(n_components=n_components)
        transformed_soap = pca.fit_transform(reshaped_soap)
        pca_ds = transformed_soap.reshape(n_atom, n_frames, -1)

        soap_pca = Insight(pca_ds, meta=soap.meta.copy())

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

The output :class:`.trajectory.Insight` stores the SOAP information in its
"meta" attribute.

Time-lagged Independent Component Analysis (TICA)
-------------------------------------------------

More details on the algorithm here:

.. toctree::
  :maxdepth: 1

  many_body_tica <_autosummary/dynsight.descriptors.many_body_tica>

This function takes as input a :class:`.trajectory.Trj` and all the relevant
parameters, and performs the TICA of the corresponding SOAP dataset.

``lag_time`` is the time lag used to perform TICA.
``tica_dim`` is the number of TICs that the function stores in the output.

.. testcode:: recipe2-test

    def compute_soap_tica(
        trj: Trj,
        r_cut: float,
        n_max: int,
        l_max: int,
        lag_time: int,
        tica_dim: int,
        soap_path: Path | None = None,
        tica_path: Path | None = None,
        selection: str = "all",
        centers: str = "all",
        respect_pbc: bool = True,
        n_core: int = 1,
    ) -> Insight:
        if tica_path is not None and tica_path.exists():
            return Insight.load_from_json(tica_path)

        soap = load_or_compute_soap(
            trj=trj,
            r_cut=r_cut,
            n_max=n_max,
            l_max=l_max,
            selection=selection,
            centers=centers,
            respect_pbc=respect_pbc,
            n_core=n_core,
            soap_path=soap_path,
        )

        rel_times, _, tica_ds = dynsight.descriptors.many_body_tica(
            soap.dataset,
            lag_time=lag_time,
            tica_dim=tica_dim,
        )

        meta = soap.meta.copy()
        meta.update({
            "lag_time": lag_time,
            "rel_times": rel_times,
        })
        soap_tica = Insight(tica_ds, meta=meta)

        if tica_path is not None:
            soap_tica.dump_to_json(tica_path)

        return soap_tica

    # Example of how to use
    soap_tic1 = compute_soap_tica(
        trj=trj,
        r_cut=10.0,
        n_max=4,
        l_max=4,
        lag_time=10,
        tica_dim=1,
    )

The output :class:`.trajectory.Insight` stores the SOAP information in its
"meta" attribute, together with the ``lag_time`` parameter and ``rel_times``, 
the relaxation times of the computed TICs.


timeSOAP (tSOAP)
----------------

More details on the algorithm here:

.. toctree::
  :maxdepth: 1

  timesoap <_autosummary/dynsight.soap.timesoap>

This function takes as input a :class:`.trajectory.Trj` and all the relevant
parameters, and computes the corresponding timeSOAP dataset.

``delay`` is the time lag used to perform timeSOAP.

.. testcode:: recipe2-test

    def compute_timesoap(
        trj: Trj,
        r_cut: float,
        n_max: int,
        l_max: int,
        delay: int = 1,
        soap_path: Path | None = None,
        tsoap_path: Path | None = None,
        selection: str = "all",
        centers: str = "all",
        respect_pbc: bool = True,
        n_core: int = 1,
    ) -> Insight:
        if tsoap_path is not None and tsoap_path.exists():
            return Insight.load_from_json(tsoap_path)

        soap = load_or_compute_soap(
            trj=trj,
            r_cut=r_cut,
            n_max=n_max,
            l_max=l_max,
            selection=selection,
            centers=centers,
            respect_pbc=respect_pbc,
            n_core=n_core,
            soap_path=soap_path,
        )

        tsoap = soap.get_angular_velocity(delay=delay)

        if tsoap_path is not None:
            tsoap.dump_to_json(tsoap_path)

        return tsoap

    # Example of how to use
    tsoap = compute_timesoap(
        trj=trj,
        r_cut=10.0,
        n_max=4,
        l_max=4,
    )

The output :class:`.trajectory.Insight` stores the SOAP information in its
"meta" attribute, together with the ``delay`` parameter.

Notice that, differently from SOAP - which is computed for every frame, tSOAP
is computed for every pair of frames. Thus, the tSOAP dataset has shape 
``(n_particles, n_frames - 1)``. Consequently, if you need to match the tSOAP
values with the particles along the trajectory, you will need to use a sliced
trajectory (removing the last frame). The easiest way to do this is:

.. testcode:: recipe2-test

    trajslice = slice(0, -1, 1)
    shorter_trj = trj.with_slice(trajslice=trajslice)

.. raw:: html

    <a class="btn-download" href="../_static/recipes/soap_dim_red.py" download>⬇️ Download Python Script</a>

.. testcode:: recipe2-test
    :hide:

    assert soap_pc1.dataset.shape == (7, 201, 1)
    assert soap_tic1.dataset.shape == (7, 201, 1)
    assert tsoap.dataset.shape == (7, 200)
