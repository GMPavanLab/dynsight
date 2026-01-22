The ``dynsight`` workflow
=========================

The dynsight platform has been designed to merge different tecniques or methods into a single, 
user-friendly workflow. The final aim is to allow users to easily build complex analysis pipelines
with minimal effort and with a single software. 

.. image:: _static/workflow.jpeg
   :align: center

System
------

``dynsight`` is a modular and flexible framework aimed at the analysis of 
dynamical trajectories, regardless of how they are generated.

This design allows the same set of tools and methods to be applied consistently
to trajectories originating from molecular dynamics simulations as well as from experimental video data.


Trajectory
----------

All the ``dynsight`` applications operate on trajectory files containing
particles coordinates sampled over time. At its actual state, ``dynsight``
supports some among the most common trajectory formats, such as .xtc, .xyz, as well as trajectories
provided through ``MDAnalysis`` Universes.

Descriptor calculations in ``dynsight`` are performed via the :class:`.trajectory.Trj` class which
wraps an ``MDAnalysis`` Universe.

Dataset and analyses
--------------------

Once a trajectory is loaded into a :class:`.trajectory.Trj` object, users can compute a variety of
descriptors using the methods provided by this class. The full list of methods is available `here <_autosummary/dynsight.trajectory.Trj.html>`_.

Most of these methods return datasets encapsulated in :class:`.trajectory.Insight` objects, which can be target of post-processing analyses
or clustering methods. There are also specific type of Insight objects, such as :class:`.trajectory.OnionInsight` that contains tailored
visualization methods or analysis for speicific algorithm (such as the Onion Clustering algorithm in this case).

Example and tutorials
---------------------

As a minimal example, a typical code for the computation of the LENS descriptor
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

    lens_descriptor = trj.get_lens(
        r_cut=10.0,
    )  # This is an Insight object

    lens_values = lens_descriptor.dataset  # The np.ndarray with the computed values

    clustering = lens_descriptor.get_onion_smooth(
        time_window=10,
    )  # This is an OnionSmoothInsight object

We strongly suggest to follow the tutorials available in the `tutorials section <tutorials_menu.html>`_

Classes
-------

.. toctree::
  :maxdepth: 1

  Trj <_autosummary/dynsight.trajectory.Trj>
  Insight <_autosummary/dynsight.trajectory.Insight>
  ClusterInsight <_autosummary/dynsight.trajectory.ClusterInsight>
  OnionInsight <_autosummary/dynsight.trajectory.OnionInsight>
  OnionSmoothInsight <_autosummary/dynsight.trajectory.OnionSmoothInsight>
