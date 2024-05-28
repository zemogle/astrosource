import logging
import os
import pickle
import ssl
import sys
from pathlib import Path

if (not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None)):
    ssl._create_default_https_context = ssl._create_unverified_context

from numpy import genfromtxt

from astrosource.analyse import (calibrated_photometry, find_variable_stars,
                                 photometric_calculations)
from astrosource.comparison import (check_comparisons_files, find_comparisons,
                                    find_comparisons_calibrated)
from astrosource.detrend import detrend_data
from astrosource.eebls import plot_bls
from astrosource.identify import find_stars, gather_files
from astrosource.periodic import plot_with_period
from astrosource.plots import (make_calibrated_plots, make_plots,
                               open_photometry_files, output_files,
                               phased_plots)
from astrosource.utils import (AstrosourceException, cleanup, folder_setup,
                               setup_logger)


class TimeSeries:
    def __init__(self, targets, indir, **kwargs):
        self.targets = targets

        self.indir = Path(indir)
        filelist = kwargs.get('filelist', None)
        self.format = kwargs.get('format', 'fz')
        self.imgreject = kwargs.get('imgreject', 0.05)
        self.periodupper = kwargs.get('periodupper', -99.9)
        self.periodlower = kwargs.get('periodlower', 0.05)
        self.periodtests = kwargs.get('periodtests', -99)

        self.racut = kwargs.get('racut', -99.9)
        self.deccut = kwargs.get('deccut', -99.9)
        self.radiuscut =kwargs.get('radiuscut', -99.9)

        self.minfractionimages=kwargs.get('minfractionimages', 0.5)

        self.detrendfraction =kwargs.get('detrendfraction', 0.1)

        self.thresholdcounts = kwargs.get('thresholdcounts', 1000000)
        self.hicounts = kwargs.get('hicounts', 3000000)
        self.lowcounts = kwargs.get('lowcounts', 5000)
        self.starreject = kwargs.get('starreject', 0.3)
        self.nopanstarrs = kwargs.get('nopanstarrs', False)
        self.nosdss = kwargs.get('nosdss', False)
        self.noskymapper = kwargs.get('noskymapper', False)
        
        self.nocalib = kwargs.get('nocalib', False)
        self.closerejectd = kwargs.get('closerejectd', 5.0)
        self.targetradius = kwargs.get('targetradius', 1.5)
        self.matchradius = kwargs.get('matchradius', 1.0)
        self.varsearch = kwargs.get('varsearch', False)

        self.varsearchglobalstdev = kwargs.get('varsearchglobalstdev', -99.9)
        self.varsearchthresh = kwargs.get('varsearchthresh', 10000)
        self.varsearchstdev = kwargs.get('varsearchstdev', 1.5)
        self.varsearchmagwidth = kwargs.get('varsearchmagwidth', 0.5)
        self.varsearchminimages = kwargs.get('varsearchminimages', 0.3)

        self.mincompstars = kwargs.get('mincompstars', 0.1)
        self.mincompstarstotal = kwargs.get('mincompstarstotal', -99)
        self.maxcandidatestars= kwargs.get('maxcandidatestars', 10000)

        self.lowestcounts= kwargs.get('lowestcounts', 1800)

        # Colour stuff
        self.colourdetect = kwargs.get('colourdetect', False)
        self.linearise = kwargs.get('linearise', False)
        self.calibsave = kwargs.get('calibsave', False)

        self.variablehunt = kwargs.get('variablehunt', False)
        self.notarget = kwargs.get('notarget', False)
        self.usescreenedcomps = kwargs.get('usescreenedcomps', False)

        self.colourterm = kwargs.get('colourterm', 0.0)
        self.colourerror = kwargs.get('colourerror', 0.0)
        self.targetcolour = kwargs.get('targetcolour', -99.0)

        self.restrictcompcolourcentre = kwargs.get('restrictcompcolourcentre', -99.0)
        self.restrictcompcolourrange = kwargs.get('restrictcompcolourrange', -99.0)



        self.restrictmagbrightest = kwargs.get('restrictmagbrightest', -99.0)
        self.restrictmagdimmest = kwargs.get('restrictmagdimmest', 99.0)
        self.rejectmagbrightest = kwargs.get('rejectmagbrightest', -99.0)
        self.rejectmagdimmest = kwargs.get('rejectmagdimmest', 99.0)
        self.ignoreedgefraction = kwargs.get('ignoreedgefraction', 0.05)

        self.outliererror = kwargs.get('outliererror', 4)
        self.outlierstdev = kwargs.get('outlierstdev', 4)


        verbose = kwargs.get('verbose', False)
        debug = kwargs.get('debug', False)

        if debug:
            verbosity = "DEBUG"
        elif verbose:
            verbosity = "INFO"
        else:
            verbosity = False
        logger = setup_logger('astrosource', verbosity)

        bjd = kwargs.get('bjd', False)
        self.paths = folder_setup(self.indir)

        if self.usescreenedcomps == False:
            self.files, self.filtercode, self.photFileHolder, self.photCoords = gather_files(self.paths, filelist=filelist, filetype=self.format, bjd=bjd,ignoreedgefraction=self.ignoreedgefraction, lowest=self.lowestcounts, racut=self.racut, deccut=self.deccut, radiuscut=self.radiuscut)

    def analyse(self, calib=True, usescreenedcomps=False, usecompsused=False, usecompletedcalib=False):


        parentPath = self.paths['parent']

        if usescreenedcomps == False:
            self.usedimages, self.stars, self.photFileHolder, self.photCoords = find_stars(targets=self.targets,
                                                                                 paths=self.paths,
                                                                                 fileList=self.files,
                                                                                 nopanstarrs=self.nopanstarrs,
                                                                                 nosdss=self.nosdss,                                                                                 
                                                                                 noskymapper=self.noskymapper,
                                                                                 closerejectd=self.closerejectd,
                                                                                 photCoords=self.photCoords,
                                                                                 photFileHolder=self.photFileHolder,
                                                                                 mincompstars=self.mincompstars,
                                                                                 mincompstarstotal=self.mincompstarstotal,
                                                                                 imageFracReject=self.imgreject,
                                                                                 starreject=self.starreject,
                                                                                 hicounts=self.hicounts,
                                                                                 lowcounts=self.lowcounts,
                                                                                 maxcandidatestars=self.maxcandidatestars,
                                                                                 restrictcompcolourcentre=self.restrictcompcolourcentre,
                                                                                 restrictcompcolourrange=self.restrictcompcolourrange,
                                                                                 restrictmagbrightest=self.restrictmagbrightest,
                                                                                 restrictmagdimmest=self.restrictmagdimmest,
                                                                                 filterCode=self.filtercode,
                                                                                 minfractionimages=self.minfractionimages)

        else:
            sys.stdout.write("Using screened Comparisons from Previous Run")
            # Create or load in skycoord array
            if os.path.exists(parentPath / "photSkyCoord"):
                 print ("Loading Sky coords for photometry files")
                 with open(parentPath / "photSkyCoord", 'rb') as f:
                     self.photCoords=pickle.load(f)
            #Create or load in photfileholder array
            if os.path.exists(parentPath / ""):
                sys.stdout.write("Loading Sky coords for photometry files")
                with open(parentPath / "photFileHolder", 'rb') as f:
                    self.photFileHolder=pickle.load(f)
            if os.path.exists(parentPath / ""):
                sys.stdout.write("Loading filterCode")
                with open(parentPath / "filterCode", 'rb') as f:
                    self.filtercode=pickle.load(f)
            self.usedimages=genfromtxt(parentPath / 'usedImages.txt', dtype=str, delimiter=',')
            self.stars=genfromtxt(parentPath / 'screenedComps.csv', dtype=float, delimiter=',')

        if usecompsused ==False:
            find_comparisons(self.targets, self.indir, self.usedimages,photFileArray=self.photFileHolder, photSkyCoord=self.photCoords,  matchRadius=self.matchradius, thresholdCounts=self.thresholdcounts)
        else:
            self.files=genfromtxt(parentPath / 'usedImages.txt', dtype=str, delimiter=',')
            self.usedimages,self.photCoords,self.photFileHolder=check_comparisons_files(self.indir, self.files, photFileArray=self.photFileHolder, photSkyCoord=self.photCoords, matchRadius=self.matchradius)
            #Check stars are in images

        # Check that it is a filter that can actually be calibrated - in the future I am considering calibrating w against V to give a 'rough V' calibration, but not for now.

        if self.nocalib==True:
            calib=False

        print ("calib: " + str(calib))

        if usecompletedcalib == False:
            self.calibrated = False
            if calib and self.filtercode in ['B', 'V', 'up', 'gp', 'rp', 'ip', 'zs', 'CV', 'w', 'PB', 'PG','PR']:
                try:

                    self.colourterm, self.colourerror, self.calibcompsused = find_comparisons_calibrated(targets=self.targets,
                                                                                    filterCode=self.filtercode,
                                                                                    paths=self.paths,
                                                                                    nopanstarrs=self.nopanstarrs,
                                                                                    nosdss=self.nosdss,                                                                                 
                                                                                    noskymapper=self.noskymapper,
                                                                                    closerejectd=self.closerejectd,
                                                                                    colourdetect=self.colourdetect,
                                                                                    linearise=self.linearise,
                                                                                    colourTerm=self.colourterm,
                                                                                    colourError=self.colourerror,
                                                                                    restrictmagbrightest=self.restrictmagbrightest,
                                                                                    restrictmagdimmest=self.restrictmagdimmest,
                                                                                    photCoordsFile=self.photCoords,
                                                                                    photFileHolder=self.photFileHolder,
                                                                                    calibSave=self.calibsave,
                                                                                    restrictcompcolourcentre=self.restrictcompcolourcentre,
                                                                                    restrictcompcolourrange=self.restrictcompcolourrange)

                    self.calibrated = True
                except AstrosourceException as e:
                    sys.stdout.write(f'⚠️ {e}\n')
            elif calib:
                sys.stdout.write(f'⚠️ filter {self.filtercode} not supported for calibration\n')
        else:
            self.calibcompsused=genfromtxt(parentPath / 'calibCompsUsed.csv', dtype=float, delimiter=',')
            self.calibrated = True
        return

    def find_variables(self):
        find_variable_stars(targets=self.targets, parentPath=self.paths['parent'], matchRadius=self.matchradius, varsearchglobalstdev=self.varsearchglobalstdev, varsearchthresh=self.varsearchthresh, varsearchstdev=self.varsearchstdev, varsearchmagwidth=self.varsearchmagwidth, varsearchminimages=self.varsearchminimages, photCoords=self.photCoords, photFileHolder=self.photFileHolder, fileList=self.usedimages)

    def photometry(self, filesave=False, targets=None):
        self.targets=targets

        data = photometric_calculations(targets=self.targets, paths=self.paths, targetRadius=self.targetradius, filesave=filesave, outliererror=self.outliererror, outlierstdev=self.outlierstdev,photCoordsFile=self.photCoords,photFileArray=self.photFileHolder, fileList=self.usedimages)
        self.output(mode='diff', data=data)
        if self.calibrated:
            self.data = calibrated_photometry(paths=self.paths, photometrydata=data, colourterm=self.colourterm,colourerror=self.colourerror,colourdetect=self.colourdetect,linearise=self.linearise,targetcolour=self.targetcolour,rejectmagbrightest=self.rejectmagbrightest,rejectmagdimmest=self.rejectmagdimmest, calibCompFile=self.calibcompsused)
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
            detrend_data(filterCode=self.filtercode, paths=self.paths, detrendfraction=self.detrendfraction)
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
