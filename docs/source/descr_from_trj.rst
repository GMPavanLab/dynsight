Descriptors from a :class:`.trajectory.Trj` 
===========================================

This recipe explains how to compute descriptors directly from a 
:class:`.trajectory.Trj` object. 

.. warning::

    This code works when run from the ``/docs`` directory of the ``dynsight`` 
    repo. To use it elsewhere, you have to change the ``Path`` variables
    accordingly.

First of all, we import all the packages and objects we'll need:

.. testcode:: recipe1-test

    from pathlib import Path
    from dynsight.trajectory import Trj

SOAP
----

Computing SOAP for every particle and frame of the trajectory is easy, since
it's directly calculated by the :class:`.trajectory.Trj.get_soap()` method.

.. warning::

    Please consider that the SOAP dataset can be very large, due to the high
    dimensionality, thus calculations can be expensive, and saving to/loading 
    from file quite slow.

.. testcode:: recipe1-test

    # Loading an example trajectory
    files_path = Path("../tests/systems/")
    trj = Trj.init_from_xtc(
        traj_file=files_path / "balls_7_nvt.xtc",
        topo_file=files_path / "balls_7_nvt.gro",
    )

    soap = trj.get_soap(
        r_cut=2.0,          # cutoff radius for neighbors list
        n_max=4,            # n_max SOAP parameter
        l_max=4,            # l_max SOAP parameter
        selection="all",    # compute on a selection of particles
        centers="all",      # compute for a selection of centers
        respect_pbc=False,  # consider PBC
        n_core=1,           # use multiprocessing to speed up calculations
    )

Number of neighbors and LENS
----------------------------

Similarly to SOAP, computing number of neighbors or LENS can be done with the
respective methods of the :class:`.trajectory.Trj` class.

Since both calculations need to compute the list of neighbors for each
particle at each frame, this list is also returned, and can be passed as an
optional parameter, so that when computing both quantities the second
calculation can be sped up significantly.

.. testcode:: recipe1-test

    # Loading an example trajectory
    files_path = Path("../tests/systems/")
    trj = Trj.init_from_xtc(
        traj_file=files_path / "balls_7_nvt.xtc",
        topo_file=files_path / "balls_7_nvt.gro",
    )

    # Computing number of neighbors from scratch
    neigcounts, n_neig = trj.get_coord_number(
        r_cut=2.0,          # cutoff radius for neighbors list
        selection="all",    # compute on a selection of particles
        neigcounts=None,    # it will be computed and returned
    )

    # Now for LENS we already have neigcounts
    _, lens = trj.get_lens(
        r_cut=2.0,               # cutoff radius for neighbors list
        selection="all",         # compute on a selection of particles
        neigcounts=neigcounts,   # no need to compute it again
    )

Notice that, differently from SOAP - which is computed for every frame, LENS
is computed for every pair of frames. Thus, the LENS dataset has shape 
``(n_particles, n_frames - 1)``. Consequently, if you need to match the LENS
values with the particles along the trajectory, you will need to use a sliced
trajectory (removing the last frame). The easiest way to do this is:

.. testcode:: recipe1-test

    trajslice = slice(0, -1, 1)
    shorter_trj = trj.with_slice(trajslice=trajslice)

.. raw:: html

    <a class="btn-download" href="../_static/recipes/descr_from_trj.py" download>⬇️ Download Python Script</a>

.. testcode:: recipe1-test
    :hide:

    assert soap.dataset.shape == (7, 201, 50)
    assert lens.dataset.shape == (7, 200)
