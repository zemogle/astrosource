# autovar
Analysis script for sources with variability in their brightness

## Installation

It is strongly recommended you use python 3 and a virtual environment e.g.
```bash
conda create -n autovar python=3
source activate autovar
```

There are only a couple of packages you will need. These are listed in `requirements.pip`. To install them you can use:

```bash
pip install -r requirements.pip
```

## Options

There are a few options when running the scripts. You can either run the whole analysis at once or the individual stages.

--ra [required] = Right Ascension of the target (in decimal)

--dec [required] = Declination of the target (in decimal)

--full = Run the whole code

--stars = Identify and match stars from each photometry file

--comparison = Identify non-varying stars to use for comparisons

--calc = Calculate the brightness change of the target

--plot = Produce lightcurve plots


### Example Usage

```bash
python main.py --full --ra 154.9083708 --dec -9.8062778
```

## Acknowledgements
Written by Michael Fitzgerald. Adapted by Edward Gomez
