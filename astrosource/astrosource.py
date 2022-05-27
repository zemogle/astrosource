from pathlib import Path
import logging
import sys
import os
import ssl

if (not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None)):
    ssl._create_default_https_context = ssl._create_unverified_context

from astrosource.analyse import find_variable_stars, photometric_calculations, calibrated_photometry
from astrosource.comparison import find_comparisons, find_comparisons_calibrated, check_comparisons_files
from astrosource.detrend import detrend_data
from astrosource.eebls import plot_bls
from astrosource.identify import find_stars, gather_files
from astrosource.periodic import plot_with_period
from astrosource.plots import make_plots, make_calibrated_plots, open_photometry_files, output_files, phased_plots
from astrosource.utils import AstrosourceException, folder_setup, cleanup, setup_logger
from numpy import genfromtxt

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
        self.hicounts = kwargs.get('hicounts', 3000000)
        self.lowcounts = kwargs.get('lowcounts', 5000)
        self.starreject = kwargs.get('starreject', 0.3)
        self.nopanstarrs = kwargs.get('nopanstarrs', False)
        self.nosdss = kwargs.get('nosdss', False)
        self.closerejectd = kwargs.get('closerejectd', 5.0)
        self.targetradius = kwargs.get('targetradius', 1.5)
        self.matchradius = kwargs.get('matchradius', 1.0) 
        self.varsearch = kwargs.get('varsearch', False)
        self.mincompstars = kwargs.get('mincompstars', 0.1)
        self.mincompstarstotal = kwargs.get('mincompstarstotal', -99)
        self.maxcandidatestars= kwargs.get('maxcandidatestars', 10000)
        # Colour stuff
        self.colourdetect = kwargs.get('colourdetect', False)
        self.linearise = kwargs.get('linearise', False)
        self.colourterm = kwargs.get('colourterm', 0.0)
        self.colourerror = kwargs.get('colourerror', 0.0)
        self.targetcolour = kwargs.get('targetcolour', -99.0)
        self.restrictmagbrightest = kwargs.get('restrictmagbrightest', -99.0)
        self.restrictmagdimmest = kwargs.get('restrictmagdimmest', 99.0)
        self.rejectmagbrightest = kwargs.get('rejectmagbrightest', -99.0)
        self.rejectmagdimmest = kwargs.get('rejectmagdimmest', 99.0)
        self.varsearchthresh = kwargs.get('varsearchthresh', 10000)
        verbose = kwargs.get('verbose', False)
        bjd = kwargs.get('bjd', False)
        self.paths = folder_setup(self.indir)
        logger = setup_logger('astrosource', verbose)
        self.files, self.filtercode = gather_files(self.paths, filelist=filelist, filetype=self.format, bjd=bjd)

    def analyse(self, calib=True, usescreenedcomps=False, usecompsused=False, usecompletedcalib=False):


        parentPath = self.paths['parent']
            
        if usescreenedcomps == False:
            self.usedimages, self.stars = find_stars(targets=self.targets,
                                                     paths=self.paths,
                                                     fileList=self.files,
                                                     mincompstars=self.mincompstars,
                                                     mincompstarstotal=self.mincompstarstotal,
                                                     imageFracReject=self.imgreject,
                                                     starreject=self.starreject,
                                                     hicounts=self.hicounts,
                                                     lowcounts=self.lowcounts,
                                                     maxcandidatestars=self.maxcandidatestars)
            
        else:
            print ("Using screened Comparisons from Previous Run")
            self.usedimages=genfromtxt(parentPath / 'usedImages.txt', dtype=str, delimiter=',')
            self.stars=genfromtxt(parentPath / 'screenedComps.csv', dtype=str, delimiter=',')
        
        if usecompsused ==False:
            find_comparisons(self.targets, self.indir, self.usedimages, matchRadius=self.matchradius, thresholdCounts=self.thresholdcounts)
        else:
            self.usedimages=check_comparisons_files(self.indir, self.files, matchRadius=self.matchradius)
            #Check stars are in images
            
        
        # Check that it is a filter that can actually be calibrated - in the future I am considering calibrating w against V to give a 'rough V' calibration, but not for now.
        if usecompletedcalib == False:
            self.calibrated = False
            if calib and self.filtercode in ['B', 'V', 'up', 'gp', 'rp', 'ip', 'zs', 'CV']:
                try:
                    
                    self.colourterm, self.colourerror, _ = find_comparisons_calibrated(targets=self.targets,
                                                                                    filterCode=self.filtercode,
                                                                                    paths=self.paths,
                                                                                    nopanstarrs=self.nopanstarrs,
                                                                                    nosdss=self.nosdss,
                                                                                    closerejectd=self.closerejectd,
                                                                                    colourdetect=self.colourdetect,
                                                                                    linearise=self.linearise,
                                                                                    colourTerm=self.colourterm,
                                                                                    colourError=self.colourerror,
                                                                                    restrictmagbrightest=self.restrictmagbrightest,
                                                                                    restrictmagdimmest=self.restrictmagdimmest)
    
                    self.calibrated = True
                except AstrosourceException as e:
                    sys.stdout.write(f'⚠️ {e}\n')
            elif calib:
                sys.stdout.write(f'⚠️ filter {self.filtercode} not supported for calibration\n')
        else:
            self.calibrated = True
        
        

    def find_variables(self):
        find_variable_stars(targets=self.targets, parentPath=self.paths['parent'], matchRadius=self.matchradius, varsearchthresh=self.varsearchthresh)

    def photometry(self, filesave=False):
        data = photometric_calculations(targets=self.targets, paths=self.paths, targetRadius=self.targetradius, filesave=filesave)
        self.output(mode='diff', data=data)
        if self.calibrated:
            self.data = calibrated_photometry(paths=self.paths, photometrydata=data, colourterm=self.colourterm,colourerror=self.colourerror,colourdetect=self.colourdetect,linearise=self.linearise,targetcolour=self.targetcolour,rejectmagbrightest=self.rejectmagbrightest,rejectmagdimmest=self.rejectmagdimmest)
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
