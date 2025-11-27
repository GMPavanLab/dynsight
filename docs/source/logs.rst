Logs
====

dynsight logging system.

.. warning::

   A default instance of :class:`Logger` is **automatically created** when importing the ``dynsight`` package.  
   This instance is available as ``dynsight.logs.logger``.

   You can configure it, for example to disable the automatic recording of datasets, using:

   .. code-block:: python

      import dynsight
      dynsight.logs.logger.configure(auto_recording=False)

   You can also access all its attributes and methods described in the Logs page below.

-----
Usage
-----

.. toctree::
  :maxdepth: 1

  Logs <_autosummary/dynsight.logs.rst>
