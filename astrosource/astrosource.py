from pathlib import Path
import logging
import sys

from astrosource.analyse import find_variable_stars, photometric_calculations, calibrated_photometry
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
        filelist = kwargs.get('filelist', None)
        self.format = kwargs.get('format', 'fz')
        self.imgreject = kwargs.get('imgreject', 0.05)
        self.periodupper = kwargs.get('periodupper', -99.9)
        self.periodlower = kwargs.get('periodlower', -99.9)
        self.periodtests = kwargs.get('periodtests', -99)
        self.rejectbrighter = kwargs.get('rejectbrighter', 99)
        self.rejectdimmer = kwargs.get('rejectdimmer', 99)
        self.thresholdcounts = kwargs.get('thresholdcounts', 1000000)
        self.hicounts = kwargs.get('hicounts', 1500000)
        self.lowcounts = kwargs.get('lowcounts', 1000)
        self.starreject = kwargs.get('starreject', 0.3)
        self.nopanstarrs = kwargs.get('nopanstarrs', False)
        self.nosdss = kwargs.get('nosdss', False)
        self.closerejectd = kwargs.get('closerejectd', 5.0)
        self.skipvarsearch = kwargs.get('skipvarsearch', False)
        self.mincompstars = kwargs.get('mincompstars', 0.1)
        verbose = kwargs.get('verbose', False)
        bjd = kwargs.get('bjd', False)
        self.paths = folder_setup(self.indir)
        logger = setup_logger('astrosource', verbose)
        self.files, self.filtercode = gather_files(self.paths, filelist=filelist, filetype=self.format, bjd=bjd)

    def analyse(self, calib=True):
        self.usedimages, self.stars = find_stars(targets=self.targets,
                                                 paths=self.paths,
                                                 fileList=self.files,
                                                 mincompstars=self.mincompstars,
                                                 imageFracReject=self.imgreject,
                                                 starreject=self.starreject,
                                                 hicounts=self.hicounts,
                                                 lowcounts=self.lowcounts)
        find_comparisons(self.targets, self.indir, self.usedimages, thresholdCounts=self.thresholdcounts)
        # Check that it is a filter that can actually be calibrated - in the future I am considering calibrating w against V to give a 'rough V' calibration, but not for now.
        self.calibrated = False
        if calib and self.filtercode in ['B', 'V', 'up', 'gp', 'rp', 'ip', 'zs']:
            try:
                find_comparisons_calibrated(targets=self.targets,
                                            filterCode=self.filtercode,
                                            paths=self.paths,
                                            nopanstarrs=self.nopanstarrs,
                                            nosdss=self.nosdss,
                                            closerejectd=self.closerejectd)
                self.calibrated = True
            except AstrosourceException as e:
                sys.stdout.write(f'🛑 {e}')
        elif calib:
            sys.stdout.write(f'⚠️ filter {self.filtercode} not supported for calibration')

    def find_variables(self):
        find_variable_stars(targets=self.targets, parentPath=self.paths['parent'])

    def photometry(self, filesave=False):
        data = photometric_calculations(targets=self.targets, paths=self.paths, filesave=filesave)
        self.output(mode='diff', data=data)
        if self.calibrated:
            self.data = calibrated_photometry(paths=self.paths, photometrydata=data)
            self.output(mode='calib', data=self.data)
        else:
            self.data = data

    def plot(self, detrend=False, eebls=False, period=False, phaseShift=0.0, filesave=False):
        if not hasattr(self, 'data'):
            self.data = open_photometry_files(self.paths['outcatPath'])
        make_plots(filterCode=self.filtercode, paths=self.paths, photometrydata=self.data)
        if self.calibrated:
            make_calibrated_plots(filterCode=self.filtercode, paths=self.paths, photometrydata=self.data)
        if detrend:
            detrend_data(filterCode=self.filtercode, paths=self.paths)
        if period:
            self.period = plot_with_period(filterCode=self.filtercode, paths=self.paths, minperiod=self.periodlower, maxperiod=self.periodupper, periodsteps=self.periodtests)
            if self.calibrated:
                phased_plots(filterCode=self.filtercode, paths=self.paths, targets=self.targets, period=self.period, phaseShift=phaseShift)
        if eebls:
            plot_bls(paths=self.paths)

    def output(self, mode, data):
        output_files(paths=self.paths, photometrydata=data, mode=mode)

    def clean(self):
        cleanup(parentPath=self.paths['parent'])
