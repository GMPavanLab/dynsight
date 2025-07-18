Descriptors
===========

tICA
----

.. toctree::
  :maxdepth: 1

  many_body_tica <_autosummary/dynsight.descriptors.many_body_tica>

tICA (time-lagged Independent Component Analysis) is a dimensionality reduction method. Time-series data are mapped to components characterizing the slowest processes, by maximizing the data correlation function at a given lag-time. 

This module allows to perform tICA on trajectories from many-body systems, where the observables under analysis are single-particle descriptors, which should not be mixed within the tICs. 

This module uses the TICA class from the deeptime.decomposition package (see the `documentation page <https://deeptime-ml.github.io/latest/notebooks/tica.html>`_).
:mod:`deeptime` requires numpy <= 2.1. 
