:maintainers:
    `andrewtarzia <https://github.com/andrewtarzia/>`_ -
    `matteobecchi <https://github.com/matteobecchi/>`_ -
    `simonemartino <https://github.com/SimoneMartino98/>`_
:documentation: https://dynsight.readthedocs.io

Installation
============

To get ``dynsight``, you can install it with pip::

    $ pip install dynsight


DEV installation
============

If you want to setup a develpment version of ``dynsight`` you can follow this simple steps::
    1. Fork the ``dynsight`` repository on your github account by clicking the ``Fork`` button in the github page.
    2. Clone your forked repository by using ``git clone <YOUR_LINK>`` (fill properly).
    3. Create a new virual environment or ``conda`` or ``mamba`` (reccomended) using Python 3.10 version.
    4. Install ``just`` (https://github.com/casey/just).
    5. Activate the virtual environment created and run ``just dev`` within the cloned repository.

Other important ``just`` commands are:
    * ``just check``: tests the installation and the code (for formatting and pytests).
    * ``just docs``: generates the documantation locally for testing and modifying.

Examples
========

There are examples available in the ``examples/`` directory of this repository.

There are also examples available in the ``cpctools`` repository
`here <https://github.com/GMPavanLab/cpctools/tree/main/Examples>`

How To Cite
===========

If you use ``dynsight`` please cite

    https://github.com/GMPavanLab/dynsight



Publications
============

* people need to add PRs to mention these.


Acknowledgements
================

We developed this code when working in the Pavan group,
https://www.gmpavanlab.com/, whose members often provide very valuable
feedback, which we gratefully acknowledge.

Much of the original code in ``cpctools`` was written by Daniele Rapetti (Iximiel).

The work was funded by the European Union and ERC under projects DYNAPOL and the
NextGenerationEU project, CAGEX.

.. figure:: docs/source/_static/EU_image.png

