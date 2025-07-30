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

  Descriptors from a Trj <descr_from_trj>
  Dimensionality reduction methods <soap_dim_red>
  Information gain analysis <info_gain>

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
`here <https://github.com/GMPavanLab/cpctools/tree/main/Examples>`


How To Cite
===========

If you use ``dynsight`` please cite

    https://github.com/GMPavanLab/dynsight

and

    TBD

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

We developed this code when working in the Pavan group,
https://www.gmpavanlab.polito.it/, whose members often provide very valuable
feedback, which we gratefully acknowledge.

Much of the original code in ``cpctools`` was written by Daniele Rapetti (Iximiel).

The work was funded by the European Union and ERC under projects DYNAPOL and the
NextGenerationEU project, CAGEX.

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
