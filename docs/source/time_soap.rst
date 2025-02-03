timeSOAP
========

`TimeSOAP <https://doi.org/10.1063/5.0147025>`_ (tSOAP) is a single-particle descriptor which allows monitoring the changes in the local particles' spatial arrangement by computing the variation, frame by frame, of the direction of the `SOAP <https://doi.org/10.1103/PhysRevB.87.184115>`_ vector centered on each particle. 

For each particle, at each trajectory frame, the SOAP power spectrum can be computed, giving an approximate representation of the chemical surrounding of that particle. 

For a given particle with index :math:`i`, tSOAP is computed between frames :math:`t_1` and :math:`t_2` as the distance of the SOAP spectra of that particle between the two frames. The distance between SOAP spectra is defined as

.. math::
    d(\vec{a},\vec{b}) =
    \sqrt{2-2\frac{\vec{a}\cdot\vec{b}}{||\vec{a}||\cdot||\vec{b}||}}

and represents the angle between the two SOAP vectors :math:`\vec{a}` and :math:`\vec{b}`. 

In this way, tSOAP is defined as a number between 0 and 2, where 0 correspond to the case where no change in the SOAP vector's direction was observed, and 2 to the case where the vector's direction flipped completely. 

Examples of how to compute tSOAP can be found at ``examples/soap.py``. 

Functions
---------

.. toctree::
  :maxdepth: 1

  timesoap <_autosummary/dynsight.soap.timesoap>

Acknowledgements
----------------

If you use tSOAP in your work, please cite `this paper <https://doi.org/10.1063/5.0147025>`_. The tSOAP code was written by Cristina Caruso.