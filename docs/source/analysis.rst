analysis
========

The ``analysis`` modulus contains functions for miscellaneous analyses on many-body trajectories. 


Entropy
-------

The ``analysis`` module offers a variety of functions for entropy- and
information-based calculations. 

.. toctree::
  :maxdepth: 1

  shannon <_autosummary/dynsight.analysis.shannon>
  info_gain <_autosummary/dynsight.analysis.info_gain>
  compute_negentropy <_autosummary/dynsight.analysis.compute_negentropy>
  sample_entropy <_autosummary/dynsight.analysis.sample_entropy>

The following functions are deprecated, the previous ones should be preferred.

.. toctree::
  :maxdepth: 1

  compute_shannon <_autosummary/dynsight.analysis.compute_shannon>
  compute_kl_entropy <_autosummary/dynsight.analysis.compute_kl_entropy>
  compute_entropy_gain <_autosummary/dynsight.analysis.compute_entropy_gain>
  compute_shannon_multi <_autosummary/dynsight.analysis.compute_shannon_multi>
  compute_kl_entropy_multi <_autosummary/dynsight.analysis.compute_kl_entropy_multi>
  compute_entropy_gain_multi <_autosummary/dynsight.analysis.compute_entropy_gain_multi>

Other functions
---------------

.. toctree::
  :maxdepth: 1

  compute_rdf <_autosummary/dynsight.analysis.compute_rdf>
  self_time_correlation <_autosummary/dynsight.analysis.self_time_correlation>
  cross_time_correlation <_autosummary/dynsight.analysis.cross_time_correlation>
  spatialaverage <_autosummary/dynsight.analysis.spatialaverage.rst>
