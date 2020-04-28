from astropy.io import fits
import numpy as np
import os
from pathlib import Path
import pytest

from astrosource.periodic import plot_with_period, phase_dispersion_minimization


TEST_PATH_PARENT = Path(os.path.dirname(__file__)) / 'test_files'

TEST_PATHS = {'parent': TEST_PATH_PARENT / 'period',
              'outcatPath' : TEST_PATH_PARENT / 'period',
              'periods' : TEST_PATH_PARENT / 'period'}

TEST_FILES = [
            'V1_PDMLikelihoodPlot.png',
            'V1_PDM_PhaseddiffMags.csv',
            'V1_String_PhasedCalibMags.csv',
            'V1_PDMTestPeriodPlot.png',
            'V1_StringLikelihoodPlot.png',
            'V1_String_PhasedDiffMags.csv',
            'V1_PDMTestPeriodPlot_Calibrated.png',
            'V1_StringTestPeriodPlot.png',
            'V1_Trials.csv',
            'V1_PDMTrial.csv',
            'V1_StringTestPeriodPlot_Calibrated.png',
            'V1_PDM_PhasedCalibMags.csv',
            'V1_StringTrial.csv',
]

def test_pdm():
    vardata = np.genfromtxt(TEST_PATHS['parent'] / 'V1_diffExcel.csv', dtype=float, delimiter=',')
    num = 10000
    minperiod = 0.2
    maxperiod = 1.2
    numBins= 10
    periodPath = TEST_PATHS['periods']
    variableName = 'V1'
    pdm = phase_dispersion_minimization(vardata, num,  minperiod, maxperiod, numBins, periodPath, variableName)
    assert 0.0047599999 == pytest.approx(pdm['stdev_error'])
    assert 0.0051999999 == pytest.approx(pdm['distance_error'])
    assert len(pdm['periodguess_array']) == num
    teardown_function()

def test_period_files_created():
    plot_with_period(paths=TEST_PATHS, filterCode='B')
    for t in TEST_FILES:
        assert (TEST_PATHS['periods'] / t).exists() == True
    teardown_function()

def teardown_function():
    for tf in TEST_FILES:
        f = TEST_PATHS['periods'] / tf
        if f.exists():
            os.remove(f)
