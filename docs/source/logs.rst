Logs
====

dynsight logging system.

.. note::

   A default :class:`Logger` is istantiated to keep the user updated on the ongoing computational steps.
   
   An option of :class:`Logger` that automatically saves and records the dataset
   can be activated after importing the ``dynsight`` package by using:

   .. code-block:: python

      import dynsight
      dynsight.logs.logger.configure(auto_recording=True)

   You can also access all its attributes and methods described in the Logs page below.

-----
Usage
-----

.. toctree::
  :maxdepth: 1

  Logs <_autosummary/dynsight.logs.rst>
