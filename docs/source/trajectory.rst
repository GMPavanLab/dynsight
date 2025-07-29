How to get started: the ``trajectory`` module
=============================================

The easiest way to use ``dynsight`` is via the ``trajectory`` module, which
offers an object-oriented implementation of most of the functions and utilities,
to facilitate the workflow of any analysis.

Moreover, the ``trajectory`` classes make it easier to save and share results,
and has a built-in logging module.

The :class:`.trajectory.Trj` class is a container for a MDAnalysis.Universe,
which has methods for the computation of all the descriptors implemented in
``dynsight``.

These descriptors, as well as the output of subsequent analyses, are stored in
:class:`.trajectory.Insight` or :class:`.trajectory.ClusterInsight` objects.

We recommend the users, when possible, to write code using this module.

As a minimal example, a typical code for the computation of the SOAP descriptor
may look like this:

.. code-block::

    from pathlib import Path
    from dynsight import Trj

    traj_file = Path("path/to/the/traj.xtc")
    topo_file = Path("path/to/the/traj.gro")

    trj = Trj.init_from_xtc(
        traj_file=traj_file,
        topo_file=topo_file,
    )  # This is a Trj object

    soap = trj.get_soap(
        r_cut=10.0,
        n_max=4,
        l_max=4
    )  # This is an Insight object

    soap_values = soap.dataset  # The np.ndarray with the computed values

Complete examples can be found in the recipes section of this documentation. 

Classes
-------

.. toctree::
  :maxdepth: 1

  Trj <_autosummary/dynsight.trajectory.Trj>
  Insight <_autosummary/dynsight.trajectory.Insight>
  ClusterInsight <_autosummary/dynsight.trajectory.ClusterInsight>
  OnionInsight <_autosummary/dynsight.trajectory.OnionInsight>
  OnionSmoothInsight <_autosummary/dynsight.trajectory.OnionSmoothInsight>
