Descriptors
===========

.. toctree::
  :maxdepth: 1

   SOAP <soap>
   timeSOAP <time_soap>
   LENS <lens>

tICA
----

.. toctree::
  :maxdepth: 1

  many_body_tica <_autosummary/dynsight.descriptors.many_body_tica>

tICA (time-lagged Independent Component Analysis) is a dimensionality reduction method. Time-series data are mapped to components characterizing the slowest processes, by maximizing the data correlation function at a given lag-time. 

This module allows to perform tICA on trajectories from many-body systems, where the observables under analysis are single-particle descriptors, which should not be mixed within the tICs. 

This module uses the TICA class from the deeptime.decomposition package (see the `documentation page <https://deeptime-ml.github.io/latest/notebooks/tica.html>`_).
:mod:`deeptime` requires numpy <= 2.1. 


Velocity alignment
------------------

.. toctree::
  :maxdepth: 1

  velocity_alignment <_autosummary/dynsight.descriptors.velocity_alignment>

Computes the average velocity alignment between the central particle and the
neighboring ones. The alignment is computed as the cosine between the velocities.
Thus, the output is a number between 1 (perfect alignment) and -1 (perfect
anti-alignment).

Orinetational order parameter
-----------------------------

.. toctree::
  :maxdepth: 1

  orientational_order_param <_autosummary/dynsight.descriptors.orientational_order_param>

Computes orientational order parameter for the neighbors of each atom in the
trajectory. The output is a real number between 0 and 1, where 1 corresponds
to a perfect order, and 0 to completely random positions.
