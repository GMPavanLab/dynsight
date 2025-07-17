trajectory
==========

This module offers an object-oriented implementation of the ``dynsight``
utilities, to facilitate the workflow of any analysis.
Moreover, the ``trajectory`` classes make it easier to save and share results,
and have a built-in logging module.

The :class:`.trajectory.Trj` class is a container for a MDAnalysis.Universe,
which allows for the computation of all the descriptors implemented in ``dynsight``.

These descriptors, as well as the output of subsequent analyses, are stored in
:class:`.trajectory.Insight` or :class:`.trajectory.ClusterInsight` objects.

We reccomand the users, when possible, to write code using this module.

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
