Library Usage
=============
To take advantage of some of the more advance options in AstroSource, you want to import the library as part of a larger code base.

Getting Started
---------------
The fundamental role of AstroSource is the analysis of time-series data. So, the basic class in AstroSource is called `TimeSeries`.

There are only *2* required inputs: a list of targets and a directory where the input files can be found.

.. code-block:: python

  from astrosource import TimeSeries
  import numpy as np

  targets = np.array([(12.7249867, -34.68742, 0, 0)])
  indir = '/home/wbyrd/data/my-target/'

  ts = TimeSeries(indir=indir, targets=targets)

This will instantiate the ``TimeSeries`` object and sort through all the files in ``indir``.

Required
~~~~~~~~

The possible inputs are:
**indir** `str`
  The location of either the input photometry files or `fz` files with photometry tables.
**targets**
  A numpy array of targets. Each target must be padded with 2 extra zeros at the end.

Optional Inputs
~~~~~~~~~~~~~~~
**format** `str`
  A file extension for the files in `indir` which contain the photometry data.
**imgreject** `float`
  Image fraction rejection allowance. Defaults to `0.0`. Increasing this will allow AstroSource to reject some of your data files if there are not enough comparison stars.


Analysis
-------
To find the stars in the photometry tables and find comparisons. This will perform photometric calibration unless `calib=False` is passed.

.. code-block:: python

  ts.analyse(calib=False)
  ts.curves()
  ts.photometry()
