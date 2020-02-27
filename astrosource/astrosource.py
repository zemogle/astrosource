from pathlib import Path
import logging

from astrosource.analyse import calculate_curves, photometric_calculations
from astrosource.comparison import find_comparisons, find_comparisons_calibrated
from astrosource.detrend import detrend_data
from astrosource.eebls import plot_bls
from astrosource.identify import find_stars, gather_files
from astrosource.periodic import plot_with_period
from astrosource.plots import make_plots
from astrosource.utils import AstrosourceException, folder_setup, cleanup, setup_logger


class TimeSeries:
    def __init__(self, targets, indir, **kwargs):
        self.targets = targets
        self.indir = Path(indir)
        self.format = kwargs.get('format','fz')
        self.imgreject = kwargs.get('imgreject',0.0)
        verbose = kwargs.get('verbose', False)
        self.paths = folder_setup(self.indir)
        logger = setup_logger('astrosource', verbose)
        self.filelist, self.filtercode = gather_files(self.paths, filetype=self.format)

    def analyse(self, calib=True):
        usedimages = find_stars(self.targets, self.paths, self.filelist, imageFracReject=self.imgreject)
        find_comparisons(self.targets, self.indir, usedimages)
        # Check that it is a filter that can actually be calibrated - in the future I am considering calibrating w against V to give a 'rough V' calibration, but not for now.
        if calib == True and self.filtercode in ['B', 'V', 'up', 'gp', 'rp', 'ip', 'zs']:
            try:
                find_comparisons_calibrated(self.filtercode, self.paths)
            except AstrosourceException as e:
                logger.warning(e)

        else:
            sys.stdout.write(f'⚠️ filter {self.filtercode} not supported for calibration')

    def curves(self):
        calculate_curves(targets=self.targets, parentPath=self.paths['parent'])

    def photometry(self):
        photometric_calculations(targets=self.targets, paths=self.paths)

    def plot(self, detrend=False, period=False, eebls=False):
        make_plots(filterCode=self.filtercode, paths=self.paths)
        if detrend:
            detrend_data(filterCode=self.filtercode, paths=self.paths)
        if period:
            plot_with_period(filterCode=self.filtercode, paths=self.paths)
        if eebls:
            plot_bls(paths=self.paths)

    def clean(self):
        cleanup(self.paths['parent'])
