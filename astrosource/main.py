from pathlib import Path
import click
import sys
import logging
from colorlog import ColoredFormatter

from numpy import array
from astropy.utils.exceptions import AstropyWarning, AstropyDeprecationWarning

from astrosource.astrosource import TimeSeries

from astrosource.utils import get_targets, folder_setup, AstrosourceException, cleanup

LOG_LEVEL = logging.CRITICAL
LOGFORMAT = "  %(log_color)s%(levelname)-8s%(reset)s | %(log_color)s%(message)s%(reset)s"
logging.root.setLevel(LOG_LEVEL)
formatter = ColoredFormatter(LOGFORMAT)
stream = logging.StreamHandler()
stream.setLevel(LOG_LEVEL)
stream.setFormatter(formatter)
logger = logging.getLogger('astrosource')
logger.setLevel(LOG_LEVEL)
logger.addHandler(stream)

@click.command()
@click.option('--full', is_flag=True)
@click.option('--stars', is_flag=True)
@click.option('--comparison', is_flag=True)
@click.option('--calc', is_flag=True)
@click.option('--calib', is_flag=True)
@click.option('--phot', is_flag=True)
@click.option('--plot', is_flag=True)
@click.option('--detrend', is_flag=True)
@click.option('--eebls', is_flag=True)
@click.option('--period', is_flag=True)
@click.option('--indir', default=None, type=str, required=True)
@click.option('--ra', type=float)
@click.option('--dec', type=float)
@click.option('--target-file', default=None, type=str)
@click.option('--format', default='fz', type=str)
@click.option('--imgreject','-ir', type=float, default=0.0)
@click.option('--clean', is_flag=True)
@click.option('--verbose','-v', is_flag=True)
def main(full, stars, comparison, calc, calib, phot, plot, detrend, eebls, period, indir, ra, dec, target_file, format, imgreject, clean, verbose):
    if verbose:
        logger.setLevel(logging.DEBUG)
    try:
        parentPath = Path(indir)
        if clean:
            cleanup(parentPath)
            logger.info('All temporary files removed')
            return
        if not (ra and dec) and not target_file:
            logger.error("Either RA and Dec or a targetfile must be specified")
            return

        if ra and dec:
            targets = array([(ra,dec,0,0)])
        elif target_file:
            target_file = parentPath / target_file
            targets = get_targets(target_file)

        ts = TimeSeries(indir=parentPath, targets=targets, format=format)

        if full or comparison:
            ts.analyse()
        if full or calc:
            ts.curves()
        if full or phot:
            ts.photometry()
        if full or plot:
            ts.plot(detrend=detrend, period=period, eebls=eebls)

        sys.stdout.write("✅ AstroSource analysis complete\n")

    except AstrosourceException as e:
        logger.critical(e)
    return

if __name__ == '__main__':
    main()
