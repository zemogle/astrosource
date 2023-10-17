from os import getcwd, makedirs, remove
import shutil
import logging
import time

from numpy import asarray, genfromtxt, load, isnan, delete
from astropy.coordinates import SkyCoord
import astropy.units as u
import click
from colorlog import ColoredFormatter
from pathlib import Path

def setup_logger(name, verbose=False):
    # formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(module)s - %(message)s')
    LOGFORMAT = "%(log_color)s%(levelname)-8s%(reset)s | %(log_color)s%(message)s%(reset)s"
    formatter = ColoredFormatter(LOGFORMAT)
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    # logger.handlers = []
    if verbose == "DEBUG":
        LOG_LEVEL = logging.DEBUG
    elif verbose == "INFO":
        LOG_LEVEL = logging.INFO
    else:
        LOG_LEVEL = logging.CRITICAL
    logger.setLevel(LOG_LEVEL)
    logger.addHandler(handler)
    return logger

class Mutex(click.Option):
    def __init__(self, *args, **kwargs):
        self.not_required_if:list = kwargs.pop("not_required_if")

        assert self.not_required_if, "'not_required_if' parameter required"
        kwargs["help"] = (kwargs.get("help", "") + "Option is mutually exclusive with " + ", ".join(self.not_required_if) + ".").strip()
        super(Mutex, self).__init__(*args, **kwargs)

    def handle_parse_result(self, ctx, opts, args):
        current_opt:bool = self.name in opts
        for mutex_opt in self.not_required_if:
            if mutex_opt in opts:
                if current_opt:
                    raise click.UsageError("Illegal usage: '" + str(self.name) + "' is mutually exclusive with " + str(mutex_opt) + ".")
                else:
                    self.prompt = None
        return super(Mutex, self).handle_parse_result(ctx, opts, args)


class AstrosourceException(Exception):
    ''' Used to halt code with message '''
    pass


def cleanup(parentPath):
    folders = ['calibcats', 'periods', 'checkplots', 'eelbs', 'outputcats','outputplots','trimcats']
    for fd in folders:
        if (parentPath / fd).exists():
            shutil.rmtree(parentPath / fd)

    files = ['calibCompsUsed.csv', 'calibStands.csv', 'compsUsed.csv','screenedComps.csv', \
     'starVariability.csv', 'stdComps.csv', 'usedImages.txt', 'LightcurveStats.txt', \
     'periodEstimates.txt','calibrationErrors.txt']

    for fname in files:
        if (parentPath / fname).exists():
            (parentPath / fname).unlink()

    for fname in parentPath.glob("*.npy"):
        remove(fname)
    return

def folder_setup(parentPath=None):
    #create directory structure for output files
    if not parentPath:
        # Set default inputs directory to be relative to local path
        parentPath = Path(getcwd())
    paths = {
        'parent'     : parentPath,
        'outputPath' : parentPath / "outputplots",
        'outcatPath' : parentPath / "outputcats",
        'checkPath'  : parentPath / "checkplots",
        'periods'    : parentPath / "periods",
        'results'    : parentPath / "results"
    }
    for k, path in paths.items():
        if not path.exists():
            makedirs(path)

    return paths

def get_targets(targetfile):
    targets = genfromtxt(targetfile, dtype=float, delimiter=',')
    # Remove any nan rows from targets
    targetRejecter=[]

    if not (targets.shape[0] == 4 and targets.size == 4):
        for z in range(targets.shape[0]):
          if isnan(targets[z][0]):
            targetRejecter.append(z)
            print('FOUND A NAN ENTRY IN YOUR CSV AT LINE: ' + str(z+1))
            print ('Please check this. This can sometimes happen')
            print ('when using Excel or sophisticated software')
            print ('It is best to construct the csv using a simple text editor')
            time.sleep(5)
        targets=delete(targets, targetRejecter, axis=0)

    return targets

def convert_coords(ra, dec):

    try:
        return float(ra), float(dec)
    except ValueError:
        try:
            coords = SkyCoord(f"{ra} {dec}", unit=(u.hourangle, u.deg))
        except:
            raise AstrosourceException("Coordinates are not valid")
        return coords.ra.degree, coords.dec.degree
