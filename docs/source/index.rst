.. toctree::
   :hidden:
   :maxdepth: 1

   How to get started <trajectory>

.. toctree::
   :hidden:
   :caption: dynsight
   :maxdepth: 2

   vision <vision>
   track <track>
   descriptors <descriptors>
   onion clustering <onion>
   analysis <analysis>
   data processing <data_processing>
   HDF5er <hdf5er>
   logs <logs>

.. toctree::
  :hidden:
  :maxdepth: 2
  :caption: Examples:

  Typical analysis workflow <example_analysis_workflow>
  Information gain <example_info_gain>
  Sample Entropy <example_sample_entropy>

.. toctree::
  :hidden:
  :maxdepth: 2
  :caption: Recipes:

  Descriptors from a Trj <recipe_descr_from_trj>
  Dimensionality reduction methods <recipe_soap_dim_red>
  Entropy calculations <recipe_entropy>
  Information gain analysis <recipe_info_gain>

.. toctree::
  :hidden:
  :maxdepth: 2
  :caption: Modules:

  Modules <modules>

Overview
========

| GitHub: https://www.github.com/GMPavanLab/dynsight

:mod:`.dynsight` is structured to support a wide range of tasks commonly
encountered in the analysis of many-body dynamical systems. These tasks
include handling trajectory data, computing single-particle descriptors,
performing time-series clustering and conducting various auxiliary analyses.
To achieve this, dynsight is organized into specialized modules, each
addressing a specific aspect of this workflow.

Previously in `cpctools`_.

.. _`cpctools`: https://github.com/GMPavanLab/cpctools


Installation
============

To get :mod:`.dynsight`, you can install it with pip::

  $ pip install dynsight


Optional Dependancies
---------------------

Old versions :mod:`dynsight` used :mod:`cpctools` for SOAP calculations, if
you are using Python 3.10 and below, you can use :mod:`cpctools` to access
:mod:`SOAPify` and :mod:`hd5er` using ::

  $ pip install cpctools


How to get started
------------------

We suggest you give a read to the ``dynsight.trajectory`` module documentation,
which offers a compact and easy way of using most of the ``dynsight`` tools.
Also, the documentation offers some copiable Recipes and Examples for the most
common analyses.

How to contribute
-----------------

If you make changes or improvements to the codebase, please open a pull request
on our GitHub repository. This allows us to review, discuss, and integrate
contributions in a transparent and collaborative manner. Make sure to include
a clear description of the changes and link any related issues if applicable.


Developer Setup
---------------

#. Install `just`_.
#. In a new virtual environment run::

    $ just dev


#. Run code checks::

    $ just check


.. _`just`: https://github.com/casey/just


Examples
========

There are examples throughout the documentation and available in
the ``examples/`` directory of this repository.

There are also examples available in the ``cpctools`` repository
`here <https://github.com/GMPavanLab/cpctools/tree/main/Examples>`_


How To Cite
===========

If you use ``dynsight`` please cite:

S. Martino, M. Becchi, A. Tarzia, D. Rapetti, G. M. Pavan  
*dynsight: an open Python platform for simulation and experimental trajectory data analysis*  
arXiv (2025), DOI: `10.48550/arXiv.2510.23493 <https://doi.org/10.48550/arXiv.2510.23493>`_

``dynsight`` uses many different open-source packages. Please cite them when appropriate:

* Most modules also use MDAnalysis, https://www.mdanalysis.org/pages/citations/
* If you use SOAP, please cite https://doi.org/10.1103/PhysRevB.87.184115 and DScribe https://singroup.github.io/dscribe/latest/citing.html
* If you use timeSOAP, please cite https://doi.org/10.1063/5.0147025
* If you use LENS, please cite: https://doi.org/10.1073/pnas.2300565120
* If you use onion-clustering, please cite: https://doi.org/10.1073/pnas.2403771121
* If you use tICA, please cite ``deeptime`` https://deeptime-ml.github.io/latest/index.html
* If you use ``dynsight.vision``, please cite Ultralytics YOLO https://docs.ultralytics.com/it/models/yolo11/#usage-examples
* If you use ``dynsight.track``, please cite Trackpy https://soft-matter.github.io/trackpy/dev/introduction.html


Acknowledgements
================

``dynsight`` is developed by the Pavan group at Politecnico di Torino, https://www.gmpavanlab.polito.it/.
Many group members have provided and continuously provide with their daily work useful feedbacks,
which we gratefully acknowledge. 
This work is made possible thanks to the funding received by the European Research Council
under the European Union’s Horizon 2020 research and innovation program
(Grant Agreement No. 818776 - DYNAPOL, to G.M.P.), which G.M.P. and the whole Pavan
group thankfully acknowledge. 

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
