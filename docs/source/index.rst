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
   onion clustering <onion>
   analysis <analysis>
   data processing <data_processing>
   HDF5er <hdf5er>

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
