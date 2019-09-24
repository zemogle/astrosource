AstroSource
=============

Package for analysing time-series of sources with variability in their brightness. The package takes a sequence of data files (in FITS format, or photometry files) and will attempt to find stable comparison stars to perform differential photometry against. The package outputs a lightcurve for each input target, as ``.csv``, ``.png`` and ``.eps``. Advanced features can provide output data in absolute magnitudes, detrend periodically varying data, and calculate a simple box model for exoplanet transits.

The package was formerly called ``autovar`` but this clashes with an existing Python package.

Getting Started
---------------

Once you have installed ``astrosource``, see the :doc:`basic_usage` documentation for the full set of options and examples.


Installation
------------

It is strongly recommended you use python 3 and a virtual environment

Using the `Anaconda <https://www.anaconda.com/download/>`_ distribution of Python: ::

    conda create -n astrosource python=3
    source activate astrosource

The package is available on PyPi and can be installed with ``pip``. ::

    pip install astrosource


Install Development version
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Clone this repo and install the package with ``pip``: ::

   git clone https://github.com/joesingo/tom_education
   pip install -e tom_education

Documentation
-------------

.. toctree::
  :maxdepth: 2
  :caption: Contents:

  basic_usage
