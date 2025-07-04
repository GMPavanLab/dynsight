:maintainers:
    `andrewtarzia <https://github.com/andrewtarzia/>`_;
    `matteobecchi <https://github.com/matteobecchi/>`_;
    `simonemartino <https://github.com/SimoneMartino98/>`_
:documentation: https://dynsight.readthedocs.io

Overview
========

``dynsight`` is an open platform for supporting a wide range of tasks commonly
encountered in the trajectory and data analysis of complex dynamical systems, essentially related
to the extraction of relevant information from data obtained from trajectories.
To achieve this, ``dynsight`` is organized into specialized modules, each addressing a 
specific aspect of this workflow. For example, ``dynsight`` includes modules for, e.g., 
resolving trajectory data from movies (experimental: e.g., object recognition and tracking),
handling trajectory data (from simulations and experiments), computing single-particle descriptors, 
performing time-series clustering, maximum information extraction from data, and conducting 
various auxiliary analyses. 

A bounce of all this was previously in `cpctools`_.

.. _`cpctools`: https://github.com/GMPavanLab/cpctools

Installation
============

To get ``dynsight``, you can install it with pip::

    $ pip install dynsight

Optional Dependancies
---------------------

Old versions of ``dynsight`` used ``cpctools`` for SOAP calculations: if
you are using Python 3.10 and below, you can use ``cpctools`` to access
``SOAPify`` and ``hd5er`` using ::

  $ pip install cpctools

If you want to use the ``dynsight.tica`` module you will need to install the
deeptime package. This can be done with with pip::

  $ pip install deeptime

or with conda::

  $ conda install -c conda-forge deeptime

If you want to use the ``dynsight.vision`` and ``dynsight.track`` modules
you will need to install a series of packages. This can be done with with pip::

  $ pip install ultralytics PyYAML


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
* If you use SOAP, please cite also original references: https://doi.org/10.1103/PhysRevB.87.184115 and DScribe https://singroup.github.io/dscribe/latest/citing.html
* If you use TimeSOAP, please cite also original reference: https://doi.org/10.1063/5.0147025
* If you use LENS, please cite also original reference: https://doi.org/10.1073/pnas.2300565120
* If you use onion-clustering, please cite also original reference: https://doi.org/10.1073/pnas.2403771121
* If you use tools to calculate Information Gain and Maximum Information Extraction, please cite also original reference: https://doi.org/10.48550/arXiv.2504.12990
* If you use tICA, please cite also original ``deeptime`` reference: https://deeptime-ml.github.io/latest/index.html
* If you use ``dynsight.vision``, please cite also original ``Ultralytics YOLO`` reference: https://docs.ultralytics.com/it/models/yolo11/#usage-examples
* If you use ``dynsight.track``, please cite also original ``Trackpy`` reference: https://soft-matter.github.io/trackpy/dev/introduction.html


Acknowledgements
================

This code is developed by the G.M. Pavan group, https://www.gmpavanlab.polito.it/, 
whose members often provide very valuable feedback, which we gratefully acknowledge.

Much of the original code in ``cpctools`` was written by Daniele Rapetti (Iximiel).

This work was primarily supported by the European Research Council (ERC) under the European Union’s Horizon 2020 research and innovation program (Grant Agreement no. 818776- DYNAPOL), and also partially by the European Union under the NextGenerationEU program (grant CAGEX, SOE_0000033).

.. figure:: docs/source/_static/EU_image.png
