The typical dynsight analysis workflow
======================================

The ``dynsight.trajectory`` module provides a unified set of tools that
streamline the analysis of many-body trajectories, offering a consistent and
user-friendly interface across most analysis tasks. 

This is achieved through two main classes, :class:`.trajectory.Trj` and
:class:`.trajectory.Insight`.

The :class:`.trajectory.Trj` class is an object that contains a trajectory,
meaning the coordinates of a set of particles over a series of frames. 

The :class:`.trajectory.Insight` class is an object that stores some useful
information computed on a trajectory, in the form of a dataset containing some
physical observable computed for all the particles along the trajectory. 

Some additional classes, under the general name of 
:class:`.trajectory.ClusterInsight`, contain the result of clustering
procedures performed on the Insight datasets. 

Example
-------

The first step is usually to create a :class:`.trajectory.Trj` object from
some trajectory file (.xtc, .gro). In this example, we are using the water/ice
coexistence trajectory stored in the ``example/`` folder.

.. code-block:: python

    from pathlib import Path
    from dynsight.trajectory import Trj

    files_path = Path("dynsight/examples/analysis_workflow")
    trj = Trj.init_from_xtc(
        traj_file=files_path / "oxygens.xtc",
        topo_file=files_path / "oxygens.gro",
    )

Now ``trj`` contains the trajectory, and using the methods of the
:class:`.trajectory.Trj` class we can perform all the dynsight analyses on
this trajectory. For instance, let's say we want to compute LENS:

.. code-block:: python

    _, lens = trj.get_lens(r_cut=7.5)
    lens_file = files_path / "lens.json"
    lens.dump_to_json(lens_file)

The method :class:`.trajectory.Trj.get_lens()` returns a
:class:`.trajectory.Insight` object, which in its ``.dataset`` attribute
contains the LENS values computed on the ``trj`` trajectory. Moreover, its
``.meta`` attribute stores all the parameters relevant to this descriptor
computation (in this case, the value of the cutoff radius used, ``r_cut``).
The :class:`.trajectory.Insight` object can be easily saved as a .json file.

The :class:`.trajectory.Insight` class offers its own methods for further
analysis. For instance, one can perform spatial averaging of the LENS values:

.. code-block:: python
    
    trj_lens = trj.with_slice(slice(0, -1, 1))
    lens_smooth = lens.spatial_average(
        trj=trj_lens, r_cut=7.5, num_processes=6
    )

Notice that, since LENS is computed for all the frames but the last one, we
use a sliced trajectory, which we get with the ``Trj.with_slice()`` method. 

Finally, we can perform clustering on the ``lens_smooth.dataset``, using for
instance the :class:`Insight.get_onion_smooth()` method: 

.. code-block:: python
    
    lens_onion = lens_smooth.get_onion_smooth(delta_t=10)

    lens_onion.plot_output(
        file_path=files_path / "tmp_fig1.png",
        data_insight=lens_smooth,
    )
    lens_onion.plot_one_trj(
        file_path=files_path / "tmp_fig2.png",
        data_insight=lens_smooth,
        particle_id=1234,
    )
    lens_onion.dump_colored_trj(
        trj=trj_lens,
        file_path=files_path / "colored_trj.xyz",
    )

``lens_onion`` is an :class:`.trajectory.OnionSmoothInsight` object,
which stores the clustering output, and offers several methods to visualize
the results. 

Read the documentation to find out the complete set of objects and tools
offered by the dynsight.trajectory module. 
