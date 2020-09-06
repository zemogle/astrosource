from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.units import degree
from numpy import array as nparray
import os
import pytest
from pathlib import Path
from unittest.mock import patch, Mock

from astrosource.identify import convert_photometry_files
from astrosource.comparison import find_comparisons, read_data_files, find_reference_frame, \
    remove_stars_targets, find_comparisons_calibrated, catalogue_call

from astrosource.test.mocks import mock_vizier_query_region_vsx, mock_vizier_apass_v, mock_vizier_apass_b, \
    mock_vizier_ps_r, mock_vizier_sdss_r


TEST_PATH_PARENT = Path(os.path.dirname(__file__)) / 'test_files'

TEST_PATHS = {'parent': TEST_PATH_PARENT / 'comparison'}


class TestSetup:
    def __init__(self):
        # Create tmp files we need
        used_files = TEST_PATHS['parent'] / 'usedImages.txt'
        if used_files.exists():
            used_files.unlink()
        files = TEST_PATHS['parent'].glob('*.psx')
        files = convert_photometry_files(files)
        with used_files.open(mode='w') as fid:
            for f in files:
                fid.write("{}\n".format(f))
        # Add targets to the TestSetup object
        self.targets = nparray([(2.92142, -1.74868, 0.00000000, 0.00000000)])

@pytest.fixture()
def setup():
    return TestSetup()

def test_read_data_files(setup):
    files = os.listdir(TEST_PATHS['parent'])
    fileslist = TEST_PATHS['parent'].glob('*.npy')
    assert 'screenedComps.csv' in files
    compFile, photFileArray = read_data_files(TEST_PATHS['parent'], fileslist)
    referenceFrame, fileRaDec = find_reference_frame(photFileArray)
    assert list(referenceFrame[0]) == [154.7583434, -9.6660181000000005, 271.47230000000002, 23.331099999999999, 86656.100000000006, 319.22829999999999]
    assert (fileRaDec[0].ra.degree, fileRaDec[0].dec.degree) == (154.7583434, -9.6660181)
    assert len(referenceFrame) == 227
    assert len(fileRaDec) == 227

def test_comparison(setup):
    # All files are present so we are ready to continue
    filelist = TEST_PATHS['parent'].glob('*.npy')
    outfile, num_cands = find_comparisons(targets=setup.targets, parentPath=TEST_PATHS['parent'], fileList=filelist)

    assert outfile == TEST_PATHS['parent'] / "compsUsed.csv"
    assert num_cands == 11

@patch('astrosource.comparison.Vizier.query_region',mock_vizier_query_region_vsx)
def test_remove_targets_calibrated(setup):
    parentPath = TEST_PATHS['parent']
    fileslist = TEST_PATHS['parent'].glob('*.npy')
    compFile, photFileArray = read_data_files(parentPath, fileslist)
    assert compFile.shape == (60,2)
    compFile_out = remove_stars_targets(parentPath, compFile, acceptDistance=5.0, targetFile=setup.targets, removeTargets=1)
    # 3 stars are removed because they are variable
    assert compFile_out.shape == (57,2)

@patch('astrosource.comparison.Vizier',mock_vizier_apass_b)
def test_find_comparisons_calibrated_b(setup):
    compFile = find_comparisons_calibrated(filterCode='B', paths=TEST_PATHS, targets=setup.targets)
    assert compFile.shape == (11,5)

@patch('astrosource.comparison.Vizier',mock_vizier_apass_v)
def test_find_comparisons_calibrated_v(setup):
    compFile = find_comparisons_calibrated(filterCode='V', paths=TEST_PATHS, targets=setup.targets)
    assert compFile.shape == (11,5)

@patch('astrosource.comparison.Vizier', mock_vizier_ps_r)
def test_catalogue_call_panstarrs(setup):
    coord=SkyCoord(ra=303.6184*degree, dec=(-13.8355*degree))
    resp = catalogue_call(coord,opt={'filter' : 'rmag', 'error' : 'e_rmag'},cat_name='PanSTARRS', targets=setup.targets, closerejectd=5.0)
    print(resp.ra.shape)
    assert resp.ra.shape == (4,)

@patch('astrosource.comparison.Vizier',mock_vizier_sdss_r)
def test_catalogue_call_sdss(setup):
    coord=SkyCoord(ra=303.6184*degree, dec=(-13.8355*degree))
    resp = catalogue_call(coord,opt={'filter' : 'rmag', 'error' : 'e_rmag'},cat_name='SDSS', targets=setup.targets, closerejectd=5.0)
    assert resp.ra.shape == (3,)
