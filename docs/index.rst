.. eWaterCycle-DataAssimilation documentation master file, created by
   sphinx-quickstart on Thu Mar  7 09:14:03 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to eWaterCycle-DataAssimilation's documentation!
========================================================

**eWaterCycle-DA** is a python package which adds data assimilation functionality to the `eWaterCycle <https://ewatercycle.nl/>`_ platform.

More info to be added...


There are two ways in this package of running data assimilation. The easiest & most user friendly is method 1 where you define the parameters used in DA upfront. 

.. image:: _images/method1_define_upfront.png
   :width: 690

The more advanced, but more powerful is method 2 where you define the parameters used in DA as you go. You can run the model, then decide to assimilate on the go. 

.. image:: _images/method2_define_on_the_go.png
   :width: 690


The added benefit of this model structure is that the package can also be used for ensembles which don't assimilate, like a collection of model comparisons:

.. image:: _images/method3_no_DA.png
   :width: 690


.. toctree::
   :maxdepth: 2
   :caption: Contents:
             
   example_HBV_model_PF_1
   example_HBV_model_PF_2
   example_HBV_model_EnKF

