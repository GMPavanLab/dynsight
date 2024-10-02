Onion clustering
================

Code for single-point clustering of time-correlated data. 

Dynsight acts as an interface to the clustering algorithm implemented at https://github.com/matteobecchi/onion_clustering and described in this paper (https://doi.org/10.1073/pnas.2403771121). 

-----
Usage
-----

.. toctree::
  :maxdepth: 1

  onion_uni <_autosummary/dynsight.onion.onion_uni.rst>
  OnionUni <_autosummary/dynsight.onion.OnionUni.rst>
  onion_multi <_autosummary/dynsight.onion.onion_multi.rst>
  OnionMulti <_autosummary/dynsight.onion.OnionMulti.rst>

Example code for performing clustering on time-series can be found in ``examples/onion_uni.py`` and ``examples/onion_multi.py``, for univariate and multivariate time-series respectively. Refer to https://github.com/matteobecchi/onion_clustering for a complete documentation. 
