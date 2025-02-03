SOAP
====

This package uses `DScribe <https://singroup.github.io/dscribe/latest/>`_ to run `SOAP <https://doi.org/10.1103/PhysRevB.87.184115>`_  analysis on molecular systems. For a detailed explanation of the parameters of the SOAP calculation, refer to the `DScribe documentation <https://singroup.github.io/dscribe/latest/tutorials/descriptors/soap.html>`_. 

Examples of how to compute SOAP can be found at ``examples/soap.py``. 

Functions
---------

.. toctree::
  :maxdepth: 1

  saponify_trajectory <_autosummary/dynsight.soap.saponify_trajectory>
  fill_soap_vector_from_dscribe <_autosummary/dynsight.soap.fill_soap_vector_from_dscribe>
  normalize_soap <_autosummary/dynsight.soap.normalize_soap>
  soap_distance <_autosummary/dynsight.soap.soap_distance>
