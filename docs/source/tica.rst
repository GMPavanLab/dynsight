tICA
====

tICA (time-lagged Independent Component Analysis) is a dimensionality reduction method. Time-series data are mapped to components characterizing the slowest processes, by maximizing the data correlation function at a given lag-time. 

This module allows to perform tICA on trajectories from many-body systems, where the observables under analysis are single-particle descriptors, which should not be mixed within the tICs. 

Installation
------------

This module uses the TICA class from the deeptime.decomposition package (see the `documentation page <https://deeptime-ml.github.io/latest/notebooks/tica.html>`_). You will need to install :mod:`deeptime`. This can be done with with pip::

  $ pip install deeptime

or with conda::

  $ conda install -c conda-forge deeptime

:mod:`deeptime` requires numpy <= 2.1. 

Functions
---------

.. toctree::
  :maxdepth: 1

  many_body_tica <_autosummary/dynsight.tica.many_body_tica>
