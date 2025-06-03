Onion clustering
================

Code for single-point clustering of time-correlated data. 

Dynsight acts as an interface to the clustering algorithm implemented at 
https://github.com/matteobecchi/onion_clustering and described in this paper (
https://doi.org/10.1073/pnas.2403771121). 

-----
Usage
-----

There are two different implementations of Onion Clustering. The most recent, 
which is called with these functions and classes:

.. toctree::
  :maxdepth: 1

  onion_uni_smooth <_autosummary/dynsight.onion.onion_uni_smooth.rst>
  OnionUniSmooth <_autosummary/dynsight.onion.OnionUniSmooth.rst>
  onion_multi_smooth <_autosummary/dynsight.onion.onion_multi_smooth.rst>
  OnionMultiSmooth <_autosummary/dynsight.onion.OnionMultiSmooth.rst>

avoids the signals' segmentation which was present in the original 
implementation (the one described in the `PNAS paper <https://doi.org/10.1073/pnas.2403771121>`_). With this version, the clusters are guaranteed to be stable 
for at least ∆t frames, but there is no segmentation of the time-series in 
consecutive windows. This avoids the excessive growth of unclassified data 
points when increasing ∆t, and helps to have a more precise estimate of the 
physical timescales within the time-series.

The original implementation, instead, uses the segmentation of the time-series 
into consecutive windows of length ∆t, which are then clustered individially. It 
is called with these functions and classes:

.. toctree::
  :maxdepth: 1

  onion_uni <_autosummary/dynsight.onion.onion_uni.rst>
  OnionUni <_autosummary/dynsight.onion.OnionUni.rst>
  onion_multi <_autosummary/dynsight.onion.onion_multi.rst>
  OnionMulti <_autosummary/dynsight.onion.OnionMulti.rst>

Example code for performing clustering on time-series can be found in ``examples/
onion_uni.py`` and ``examples/onion_multi.py``, for univariate and multivariate 
time-series respectively. Refer to https://github.com/matteobecchi/
onion_clustering for a complete documentation. 
