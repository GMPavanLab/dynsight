.. dynsight documentation master file, created by
   sphinx-quickstart on Thu Oct 19 15:55:32 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. toctree::
   :hidden:
   :caption: dynsight
   :maxdepth: 2

   SOAP <soap>
   timeSOAP <time_soap>
   LENS <lens>
   tICA <tica>
   onion clustering <onion>
   analysis <analysis>
   data processing <data_processing>
   HDF5er <hdf5er>

.. toctree::
  :hidden:
  :maxdepth: 2
  :caption: Examples:

  Information gain <info_gain>
  Sample Entropy <sample_entropy>

.. toctree::
  :hidden:
  :maxdepth: 2
  :caption: Modules:

  Modules <modules>

============
Introduction
============

| GitHub: https://www.github.com/GMPavanLab/dynsight


:mod:`.dynsight` is a Python library aimed at simplifying the analysis of Molecular
Dynamics simulations.



Previously in `cpctools`_.

.. _`cpctools`: https://github.com/GMPavanLab/cpctools


Installation
------------

To get :mod:`.dynsight`, you can install it with pip::

  $ pip install dynsight

Dependencies
............

The main dependency is for SOAP analysis:

* `dscribe (1.2.0 - 1.2.2) <https://singroup.github.io/dscribe/latest/>`_

Optional Dependancies
.....................

Old versions :mod:`dynsight` used :mod:`cpctools` for SOAP calculations, if you are using Python 3.10 and below, you can use :mod:`cpctools` to access :mod:`SOAPify` and :mod:`hd5er` using ::

  $ pip install cpctools

If you want to use the :mod:`dynsight.tica` module you will need to install the deeptime package. This can be done with with pip::

  $ pip install deeptime

or with conda::

  $ conda install -c conda-forge deeptime


How to contribute
-----------------

If you make changes or improvements to the codebase, please open a pull request on our GitHub repository. This allows us to review, discuss, and integrate contributions in a transparent and collaborative manner. Make sure to include a clear description of the changes and link any related issues if applicable. 

Developer Setup
...............

#. Install `just`_.
#. In a new virtual environment run::

    $ just dev


#. Run code checks::

    $ just check


.. _`just`: https://github.com/casey/just


Overview
--------

To be written. 

Examples
--------

There are simplified examples available in the
`examples <https://github.com/GMPavanLab/dynsight/tree/main/examples>`_
directory of this repository.

There are also examples available in the ``cpctools`` repository
`here <https://github.com/GMPavanLab/cpctools/tree/main/Examples>`_.


How To Cite
-----------

If you use ``dynsight`` please cite

    https://github.com/GMPavanLab/dynsight

* Most modules also use MDAnalysis, https://www.mdanalysis.org/pages/citations/
* If you use SOAP, please cite https://doi.org/10.1103/PhysRevB.87.184115 and DScribe https://singroup.github.io/dscribe/latest/citing.html
* If you use timeSOAP, please cite https://doi.org/10.1063/5.0147025
* If you use LENS, please cite: https://doi.org/10.1073/pnas.2300565120
* If you use onion-clustering, please cite: https://doi.org/10.1073/pnas.2403771121
* If you use tICA, please cite ``deeptime`` https://deeptime-ml.github.io/latest/index.html 
* If you use ``dynsight.vision``, please cite XXX
* If you use ``dynsight.track``, please cite XXX


Acknowledgements
----------------

We developed this code when working in the Pavan group,
https://www.gmpavanlab.com/, whose members often provide very valuable
feedback, which we gratefully acknowledge.

Much of the original code in ``cpctools`` was written by Daniele Rapetti (Iximiel).

The work was funded by the European Union and ERC under projects DYNAPOL and the
NextGenerationEU project, CAGEX.

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
