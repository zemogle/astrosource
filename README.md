# astrosource
Analysis script for sources with variability in their brightness. The package was formerly called `autovar` but this clashes with an existing Python package.

[![Build Status](https://travis-ci.org/zemogle/astrosource.svg?branch=master)](https://travis-ci.org/zemogle/astrosource)

## Installation

It is strongly recommended you use python 3 and a virtual environment

Using the [Anaconda](https://www.anaconda.com/download/) distribution of Python:

```bash
conda create -n astrosource python=3
source activate astrosource
```

The package is available on PyPi and can be installed with `pip`.

```bash
pip install astrosource
```

### Install development version

If you need to install the development branch, download from [GitHub](https://github.com/zemogle/astrosource) and from the root of the repo, run:

```bash
cd astrosource
pip install .
```

or directly with the setup script

```bash
cd astrosource
python setup.py install
```

## Usage

There are a few input options when running the scripts. You can either run the whole analysis at once or the individual stages.

`--ra` *[required parameter]* Right Ascension of the target (in decimal)

`--dec` *[required parameter]* Declination of the target (in decimal)

`--target-file` *[required parameter]*

`--indir` [parameter] Path of directory containing LCO data files. If none is given, astrosource assumes the current directory

`--format` [parameter] input file format. If not `fz`, `fits`, or `fit` assumes the input files are photometry files with correct headers. If image files given, code will extra photometry from FITS extension. Defaults to `fz`.

`--stars` [boolean flag] Step 1: Identify and match stars from each data file

`--comparison` [boolean flag] Step 2: Identify non-varying stars to use for comparisons

`--calc` [boolean flag] Step 3: Calculate the brightness change of the target

`--phot` [boolean flag] Step 4: Photometry calculations for either differential or calibrated

`--plot` [boolean flag] Step 5: Produce lightcurve plots

`--full` [boolean flag] Run the whole code. This will run the following steps in this order `stars` > `comparison` > `calc` > `phot` > `plot`



### Extra options
`--verbose` [boolean flag] Show all system messages for AstroSource

`--period` [boolean flag] Search for periodicity in the data, currently with PDM and String methods. This will autoselect a reasonable search range if not provided a range.

`--periodlower` [float] Shortest period to trial in days. Default is 0.05

`--periodupper` [float] Longest period to trial in days. Default is one-third the observational baseline of your dataset.

`--periodtests` [integer] Number of different trial periods to run, Default 10000

`--skipvarsearch` [boolean flag] If this is set, this skips the variability calculations for each identified star. Pragmatically this skips creating the starVariability outputs. In a crowded field this can take an excessive amount of time.

`--detrend` [boolean flag] Detrend exoplanet data

`--eebls` [boolean flag] EEBLS - box fitting to search for periodic transits

`--calib` [boolean flag] Perform calibrated

`--lowcounts` [int] Countrate above which to accept a comparison star as a candidate. Defaults to 1000. (Note: This may seem like a low number but realistically stars at this low rate will be highly variable and rejected anyway)

`--hicounts` [int] Countrate above which to reject a comparison star as a candidate. Defaults to 1000000. (Note: This will vary from camera to camera, but it should be representative of a typical value that would have a peak pixel value significantly below the full range of the ccd, preferably lower rather than higher. )

`--thresholdcounts` [int] the number of counts at which to stop adding identified comparison stars to the ensemble. Default 1000000.

`--closerejectd` [float] astrosource will reject potential comparison calibration stars if there are nearby stars to it. Closerejectd is the limiting distance in arcseconds. For crowded fields, this value may need to be lowered. Default 5.0. While the primary function here is to identify stars that are not in crowded situations, it also has the side-effect of removing false detections in sky surveys due to image artifacts or diffraction spikes which tend to cluster together.

`--nopanstarrs` [boolean flag] Do not use the PanSTARRS catalogue for calibration. Some regions of the sky in PanSTARRS have poorer quality photometry than SDSS.

`--sdss` [boolean flag] Do not use the SDSS catalogue for calibration. Some regions of the sky in SDSS have poorer quality photometry than PanSTARRS.

`--bjd` [boolean flag] Convert the MJD time into BJD time for LCO images.

`--clean` [boolean flag] Remove all files except the original data files, and photometry files

`--imgreject` [float] Image fraction rejection allowance based on image size starting value. Defaults to `0.0`. Astrosource automatically adjusts this value, so it is only in very rare cases this might need to be set.

`--starreject` [float] Image fraction rejection allowance based on number of rejected stars starting value. Defaults to `0.3`. Astrosource automatically adjusts this value, so it is only in very rare cases this might need to be set.


### Example Usage

```bash
astrosource --ra 154.9083708 --dec -9.8062778 --indir /path/to/your/data --full
```

All the files generated will be stored in the directory you specify in `--indir`

### Output Files

A variety of output files are generated. Some are obvious, some are not so obvious.

`usedImages.txt` : the list of images astrosource chose to use out of the original image set

`screenedComps.csv` : These are the stars brighter than --lowcounts and dimmer than --hicounts identified in every single used image.

`stdComps.csv`: These are the variability of the original set of screenedComps.csv stars after rejecting outlier stars with high variability.

`compsUsed.csv`: These are the stars selected to form the differential magnitude lightcurve. They are the lowest variability stars in the dataset up to --threshcounts.

`calibStands.csv`: There are stars from stdComps.csv (as opposed to compsUsed.csv) that have identified calibrated magnitudes in the catalogue. The magnitudes in this catalogue are from the reference catalogue (e.g. APASS, Skymapper, SDSS or PanSTARRS)

`calibCompsUsed.csv`: These are the calibrated magnitudes of the compsUsed.csv comparison stars calculated by comparison to calibStands.csv.

The various calibration files may seem like a bit of a puzzle at first. The reason there is a few steps is that the brightest, least variable, most suitable comparison stars (compsUsed.csv) may (and usually are if you are using a smaller telescope) be actually saturated in the reference catalogue which can tend to become problematic at 10th to 12th magnitude. Hence the identified comparison stars in compsUsed.csv to create the shape of the lightcurve are actually sometimes also calibrated to the dimmer, but still low variability, stars available in stdComps.csv. If your comparison stars are 10th magnitude or dimmer, the identified comparison stars are likely also the calibration stars used.

`calibrationErrors.txt`: The errors output from the calibration to the reference catalogue.

`starVariability.*`: A csv listing the mean differential magnitudes and standard deviation in differential magnitudes for all stars identified in the data set. This can be used to identify variable stars, particularly using the provided png and eps plots.

`LightcurveStats.txt`: A simple list of Maximum, Minimum and Middle Magnitude and Amplitude for each requested target.

`PeriodEstimates.txt`: A simple list of the results of the period-finding function for each requested target using PDM and the String Method.

`outputcats`: This folder contains the catalogues for each target (V1, V2… etc.). There are differential (diff) and calibrated (calib) versions of the final results formatted for various software packages.

`outputplots`: This folder contains the output lightcurves (phaseplotted also if --period was requested)

`periods`: This folder contains Likelihood plots and differential and calibrated lightcurves for each target when the --period option was selected.

`checkplots`: This folder contains some plots by airmass, which are only really useful for exoplanet transits.


### Tests

If you are developing this package, you will want to run the tests. You will need `pytest` installed and then, from the `astrosource` directory within this repo, run:

```bash
pytest
```

To suppress the warning messages use:

```bash
pytest --disable-pytest-warnings
```

You may also want to install this in developer mode

```bash
python setup.py develop
```

## Authors
Written by Michael Fitzgerald and [Edward Gomez](@zemogle)
