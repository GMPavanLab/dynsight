Vision
======

Code that contains useful tools to obtain trajectories from video files.
The extraction is possible thanks to the optimization of a Convolutional Neural
Network (CNN) model training to
detect specific objects in the video. Once the positions are recovered, a
tracking algorithm (:doc:`track`) links each object with its own identity number.

.. figure:: _static/vision/vision_scheme.jpeg
   :alt: Schematic representation of a typical dynsight.vision application.
   :align: center

   Schematic representation of a typical `dynsight.vision` application.

The object detection is managed by the Ultralytics_ external library. If you
use this tool, please cite them by referring to this link_.

.. _Ultralytics: https://www.ultralytics.com
.. _link: https://docs.ultralytics.com/models/yolo11/#citations-and-acknowledgements

--------------
The Vision GUI
--------------

The `dynsight.vision` module includes an internal Graphical User Interface (GUI)
designed to assist users in preparing the training items required for object
detection. It only requires an *initial guess* of the objects present in
the video, which is then used to generate a synthetic dataset to bootstrap the
training process.

.. tip::

   Applying some basic color correction to the input video (especially converting
   it to grayscale with enhanced contrast) can significantly improve the quality
   of the initial model.

.. image:: _static/vision/vision_gui.gif
   :alt: Usage of the GUI.
   :align: center

-----
Usage
-----

.. toctree::
  :maxdepth: 1

  Video <_autosummary/dynsight.vision.Video.rst>
  Detect <_autosummary/dynsight.vision.Detect.rst>

Once obtained the particle positions, is possible to obtain the trajectories of
each particle by using the :doc:`track` module.
