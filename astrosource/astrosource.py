from pathlib import Path
import logging
import sys

from astrosource.analyse import find_stable_comparisons, photometric_calculations
from astrosource.comparison import find_comparisons, find_comparisons_calibrated
from astrosource.detrend import detrend_data
from astrosource.eebls import plot_bls
from astrosource.identify import find_stars, gather_files
from astrosource.periodic import plot_with_period
from astrosource.plots import make_plots, make_calibrated_plots, open_photometry_files, output_files, phased_plots
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
        self.usedimages, self.stars = find_stars(self.targets, self.paths, self.filelist, imageFracReject=self.imgreject)
        find_comparisons(self.targets, self.indir, self.usedimages)
        # Check that it is a filter that can actually be calibrated - in the future I am considering calibrating w against V to give a 'rough V' calibration, but not for now.
        self.calibrated = False
        if calib and self.filtercode in ['B', 'V', 'up', 'gp', 'rp', 'ip', 'zs']:
            try:
                find_comparisons_calibrated(self.filtercode, self.paths)
                self.calibrated = True
            except AstrosourceException as e:
                sys.stdout.write(f'🛑 {e}')
        elif calib:
            sys.stdout.write(f'⚠️ filter {self.filtercode} not supported for calibration')

    def find_stable(self):
        find_stable_comparisons(targets=self.targets, parentPath=self.paths['parent'])

    def photometry(self, filesave=False):
        self.data = photometric_calculations(targets=self.targets, paths=self.paths, filesave=filesave)
        self.ensemblemag = 0.0
        self.output(mode='diff')

    def plot(self, detrend=False, eebls=False, period=0.0, phaseShift=0.0, filesave=False):
        if not hasattr(self, 'data'):
            self.data = open_photometry_files(self.paths['outcatPath'])
        make_plots(filterCode=self.filtercode, paths=self.paths, photometrydata=self.data)
        if self.calibrated:
            self.ensemblemag = make_calibrated_plots(filterCode=self.filtercode, paths=self.paths, photometrydata=self.data)
            self.output(mode='calib')
        if detrend:
            detrend_data(filterCode=self.filtercode, paths=self.paths)
        if period and self.calibrated:
            phased_plots(filterCode=self.filtercode, paths=self.paths, targets=self.targets, period=period, phaseShift=phaseShift)
            plot_with_period(filterCode=self.filtercode, paths=self.paths)
        if eebls:
            plot_bls(paths=self.paths)

    def output(self, mode):
        output_files(self.paths, self.data, mode=mode)

    def clean(self):
        cleanup(self.paths['parent'])
