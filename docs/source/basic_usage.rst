Basic Usage
===========

AstroSource is designed to be run as a command line tool or a Python library. Once you have installed it you can run it using the command ``astrosource``, however you could also use it as a library which is called from other code. In this page we will only discuss **command line usage**.

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
**verbose** `boolean flag`
  Output all of the astrosource system messages

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
**period** `boolean flag`
  Search for periodicity in the data. 
**eebls** `boolean flag`
  EEBLS - box fitting to search for periodic transits
**calib** `boolean flag`
  Perform calibrated photometry to get data in absolute magnitudes
**clean** `boolean flag`
  Remove all files except the original data files
  
**periodlower** `float` 
  Shortest period to trial in days. Default is 0.05

**periodupper** `float`
  Longest period to trial in days. Default is one-third the observational baseline of your dataset.

**periodtests** `integer`
  Number of different trial periods to run, Default 10000

**skipvarsearch** `boolean flag`
  If this is set, this skips the variability calculations for each identified star. Pragmatically this skips creating the starVariability outputs. In a crowded field this can take an excessive amount of time. 

**lowcounts** `int`
  Countrate above which to accept a comparison star as a candidate. Defaults to 1000. (Note: This may seem like a low number but realistically stars at this low rate will be highly variable and rejected anyway)

**hicounts** `int`
  Countrate above which to reject a comparison star as a candidate. Defaults to 1000000. (Note: This will vary from camera to camera, but it should be representative of a typical value that would have a peak pixel value significantly below the full range of the ccd, preferably lower rather than higher. )

**thresholdcounts** `int`
  the number of counts at which to stop adding identified comparison stars to the ensemble. Default 1000000. 

**closerejectd** `float`
  astrosource will reject potential comparison calibration stars if there are nearby stars to it. Closerejectd is the limiting distance in arcseconds. For crowded fields, this value may need to be lowered. Default 5.0. While the primary function here is to identify stars that are not in crowded situations, it also has the side-effect of removing false detections in sky surveys due to image artifacts or diffraction spikes which tend to cluster together.

**nopanstarrs** `boolean flag`
  Do not use the PanSTARRS catalogue for calibration. Some regions of the sky in PanSTARRS have poorer quality photometry than SDSS.

**sdss** `boolean flag`
  Do not use the SDSS catalogue for calibration. Some regions of the sky in SDSS have poorer quality photometry than PanSTARRS.

**bjd** `boolean flag`
  Convert the MJD time into BJD time for LCO images.

**imgreject** `float`
  Image fraction rejection allowance based on image size starting value. Defaults to `0.0`. Astrosource automatically adjusts this value, so it is only in very rare cases this might need to be set.

**starreject** `float`
  Image fraction rejection allowance based on number of rejected stars starting value. Defaults to `0.3`. Astrosource automatically adjusts this value, so it is only in very rare cases this might need to be set.



Outputs
-------

By default AstroSource provides ``.csv``, ``.npy``, ``.eps``, and ``.png`` files for brightness variation of each target with time. These files are put inside the data directory provided by ``--indir`` input. The ``.csv`` files are provides in 4 different formats, appropriate for Excel/Python, `AstroImageJ <https://www.astro.louisville.edu/software/astroimagej/>`_, `Peranso <http://www.cbabelgium.com/peranso/>`_. as well as `EXOTIC <https://github.com/rzellem/EXOTIC/>`_. Additionally the photometry files are exported as NumPy array files, for speed of access by other parts of AstroSource.

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


A variety of output files are generated. Some are obvious, some are not so obvious.

**usedImages.txt** : the list of images astrosource chose to use out of the original image set

**screenedComps.csv** : These are the stars brighter than --lowcounts and dimmer than --hicounts identified in every single used image.

**stdComps.csv**: These are the variability of the original set of screenedComps.csv stars after rejecting outlier stars with high variability.

**compsUsed.csv**: These are the stars selected to form the differential magnitude lightcurve. They are the lowest variability stars in the dataset up to --threshcounts.

**calibStands.csv**: There are stars from stdComps.csv (as opposed to compsUsed.csv) that have identified calibrated magnitudes in the catalogue. The magnitudes in this catalogue are from the reference catalogue (e.g. APASS, Skymapper, SDSS or PanSTARRS)

**calibCompsUsed.csv**: These are the calibrated magnitudes of the compsUsed.csv comparison stars calculated by comparison to calibStands.csv. 

The various calibration files may seem like a bit of a puzzle at first. The reason there is a few steps is that the brightest, least variable, most suitable comparison stars (compsUsed.csv) may (and usually are if you are using a smaller telescope) be actually saturated in the reference catalogue which can tend to become problematic at 10th to 12th magnitude. Hence the identified comparison stars in compsUsed.csv to create the shape of the lightcurve are actually sometimes also calibrated to the dimmer, but still low variability, stars available in stdComps.csv. If your comparison stars are 10th magnitude or dimmer, the identified comparison stars are likely also the calibration stars used. 

**calibrationErrors.txt**: The errors output from the calibration to the reference catalogue.

**starVariability.* **: A csv listing the mean differential magnitudes and standard deviation in differential magnitudes for all stars identified in the data set. This can be used to identify variable stars, particularly using the provided png and eps plots.

**LightcurveStats.txt**: A simple list of Maximum, Minimum and Middle Magnitude and Amplitude for each requested target.

**PeriodEstimates.txt**: A simple list of the results of the period-finding function for each requested target using PDM and the String Method.

**outputcats**: This folder contains the catalogues for each target (V1, V2… etc.). There are differential (diff) and calibrated (calib) versions of the final results formatted for various software packages. 

**outputplots**: This folder contains the output lightcurves (phaseplotted also if --period was requested)

**periods**: This folder contains Likelihood plots and differential and calibrated lightcurves for each target when the --period option was selected.

**checkplots**: This folder contains some plots by airmass, which are only really useful for exoplanet transits.



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
