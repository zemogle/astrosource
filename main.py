from pathlib import Path
import click
import sys
import logging

from numpy import array

from identify import find_stars
from comparison import find_comparisons
from analyse import calculate_curves, photometric_calculations
from plots import make_plots

from utils import get_targets


logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)


@click.command()
@click.option('--full', is_flag=True)
@click.option('--stars', is_flag=True)
@click.option('--comparison', is_flag=True)
@click.option('--calc', is_flag=True)
@click.option('--phot', is_flag=True)
@click.option('--plot', is_flag=True)
@click.option('--indir', default=None, type=str)
@click.option('--ra', type=float)
@click.option('--dec', type=float)
@click.option('--target-file', default=None, type=str)
@click.option('--format', default='fz', type=str)
def main(full, stars, comparison, calc, phot, plot, indir, ra, dec, target_file, format):
    if not (ra and dec) and not target_file:
        logger.error("Either RA and Dec or a targetfile must be specified")
        return

    parentPath = Path(indir)
    if ra and dec:
        targets = array([(ra,dec,0,0)])
    elif target_file:
        target_file = parentPath / target_file
        targets = get_targets(target_file)

    # sys.tracebacklimit = 0
    if full or stars:
        find_stars(targets, parentPath, filetype=format)
    if full or comparison:
        find_comparisons(parentPath)
    if full or calc:
        calculate_curves(targets, parentPath=parentPath)
    if full or phot:
        photometric_calculations(targets, parentPath=parentPath)
    if full or plot:
        make_plots(filterCode='r', parentPath=parentPath)
    return

if __name__ == '__main__':
    main()
