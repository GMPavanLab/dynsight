LENS
====

The `Local Environments and Neighbors Shuffling <https://doi.org/10.1073/pnas.2300565120>`_ (LENS) descriptor is a single-particle descriptor which allows measuring local diffusivity by tracking the changes in particles' neighbors identities along a trajectory. 

LENS requires the particles in the system to be uniquely indexed. For a given particle with index :math:`i`, LENS is computed between frames :math:`t_1` and :math:`t_2` as

.. math::
    \text{LENS}_i(t_1, t_2) = \frac{\#(C_i^{t_1} \cup C_i^{t_2}) - \#(C_i^{t_1} \cap C_i^{t_2})}{\#(C_i^{t_1}) + \#(C_i^{t_2})}

where :math:`C_i^t` is the set of neighbors of particle :math:`i` at frame :math:`t`, and :math:`\#C` is the cardinality of set :math:`C`. 

In this way, LENS is defined as a number between 0 and 1, where 0 correspond to the case where no neighbors changed, and 1 to the case where all the neighbors changed. The set of neighbors of each particle at each frame is defined as the particles within a certain cutoff radius :math:`r_c` from that particle.

Examples of how to compute LENS can be found at ``examples/lens.py``. 

Functions
---------

.. toctree::
  :maxdepth: 1

  list_neighbours_along_trajectory <_autosummary/dynsight.lens.list_neighbours_along_trajectory>
  neighbour_change_in_time <_autosummary/dynsight.lens.neighbour_change_in_time>

Acknowledgements
----------------

If you use LENS in your work, please cite `this paper <https://doi.org/10.1073/pnas.2300565120>`_. The LENS code was written by Martina Crippa.