README
======

.. image:: https://github.com/ziatdinovmax/gpax/actions/workflows/actions.yml/badge.svg
    :target: https://github.com/ziatdinovmax/gpax/actions/workflows/actions.yml
    :alt: GiHub Actions
.. image:: https://badge.fury.io/py/gpax.svg
        :target: https://badge.fury.io/py/gpax
        :alt: PyPI version

GPax is a small Python package for physics-based Gaussian processes (GPs) built on top of NumPyro and JAX. Its purpose is to take advantage of prior physical knowledge and different data modalities when using GPs for data reconstruction and active learning. It is a work in progress, and more models will be added in the near future.

Installation
------------

If you would like to utilize a GPU acceleration, follow these `instructions <https://github.com/google/jax#installation>`_ to install JAX with a GPU support.

Then, install GPax using pip as

.. code:: bash

    pip install gpax

for the stable version or

.. code:: bash
    
    pip install git+https://github.com/ziatdinovmax/gpax

to get the most recent updates.

If you are a Windows user, we recommend to use the Windows Subsystem for Linux (WSL2).