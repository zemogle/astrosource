Basic Usage
===========

AstroSource is designed to be run as a command line tool. Once you have installed it you can run it using the command ``astrosource``, however you could also use it as a library which is called from other code. In this page we will only discuss **command line usage**.

Time-series data
------------------

AstroSource will analyse a time-series of astronomical targets. For its basic usage, astrosource expects these time-series files to be in an input directory, either as:

1. Files containing astronomical source lists, containing brightness and position in plain text format.
2. Astronomical FITS images, with source lists for the image contained in multi-extension format.

Source List Format
~~~~~~~~~~~~~~~~~~~~~~~

In Case 1. the input source list files must be in the format: ::

    ra, dec, xpixel, ypixel, counts, countserr

In case 2. the sources list contained in each FITS file must have the headers ``x``, ``y``, ``flux`` and ``fluxerr``. Each FITS file must also have a WCS solution in the image header, which is used by AstroSource to translate the ``x`` and ``y`` values into Right Ascension and Declination.

Inputs
--------

When running AstroSource there are a few mandatory inputs

**ra** `required`
  Right Ascension of the target (in decimal)
**dec** `required`
  Declination of the target (in decimal)
**target-file** `optional`
  Path to a file containing a list of targets

You are **required** to provide either ``--ra`` and ``--dec`` parameters, or provide a ``--target-file`` containing one or more targets.

The basic options

**indir** `parameter`
  Path of directory containing LCO data files. If none is given, astrosource assumes the current directory
**format** `parameter`
  Input file format. If not `fz`, `fits`, or `fit` assumes the input files are photometry files with correct headers. If image files given, code will extra photometry from FITS extension. Defaults to `fz`.
**full** `boolean flag`
  Run the whole code. This will run the following steps in this order `stars` > `comparison` > `calc` > `phot` > `plot`

The various stages which are run as part of **full** can be run individually, when combined with the required target information:

**stars** `boolean flag`
  Step 1: Identify and match stars from each data file
**comparison** `boolean flag`
  Step 2: Identify non-varying stars to use for comparisons
**calc** `boolean flag`
  Step 3: Calculate the brightness change of the target
**phot** `boolean flag`
  Step 4: Photometry calculations for either differential or calibrated
**plot** `boolean flag`
  Step 5: Produce lightcurve plots

Advanced Features
------------------

The above options will perform an basic analysis of the provided time-series files. The following options provide more advanced functionality.

**detrend** `boolean flag`
  Detrend exoplanet data
**eebls** `boolean flag`
  EEBLS - box fitting to search for periodic transits
**calib** `boolean flag`
  Perform calibrated photometry to get data in absolute magnitudes
**imgreject** `float`
  Image fraction rejection allowance. Defaults to `0.0`. Increasing this will allow AstroSource to reject some of your data files if there are not enough comparison stars.
**clean** `boolean flag`
  Remove all files except the original data files

Outputs
-------

By default AstroSource provides ``.csv``, ``.npy``, ``.eps``, and ``.png`` files for brightness variation of each target with time. These files are put inside the data directory provided by ``--indir`` input. The ``.csv`` files are provides in 3 different formats, appropriate for Excel/Python, `AstroImageJ <https://www.astro.louisville.edu/software/astroimagej/>`_, and `Peranso <http://www.cbabelgium.com/peranso/>`_. Additionally the photometry files are exported as NumPy array files, for speed of access by other parts of AstroSource.

::

  indir
  ├── outputcats
  │ ├── doerPhotV1.csv
  │ ├── V1_diffAIJ.csv
  │ ├── V1_diffAIJ.txt
  │ ├── V1_diffExcel.csv
  │ └── V1_diffPeranso.txt
  └─── outputplots
    ├── V1_EnsembleVarDiffMag.eps
    └── V1_EnsembleVarDiffMag.png


Examples
--------

If you have a directory ``/home/user/mydata`` which contains FITS files for an exoplanet, Wasp 43b, you can analyse the data with the command:

.. code-block:: bash

  $ astrosource --ra 10.3272222 --dec -9.8063889 --indir /home/user/mydata --full

This will create directories under ``/home/user/mydata`` containing the plots ``outputplots`` and data ``outputcats``. The data you get back will be **differential** photometry only.

If you would like calibrated (i.e. data in absolute magnitudes) use the ``--calib`` flag:

.. code-block:: bash

  $ astrosource --ra 10.3272222 --dec -9.8063889 --indir /home/user/mydata --calib --full

In ``outputcats`` and ``outputplots`` you will get some extra files with **calib** in the names.
