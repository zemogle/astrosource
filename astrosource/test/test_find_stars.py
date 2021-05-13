from astrosource.identify import find_stars

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

def test_find_stars(setup):
    usedImages, comparisons = find_stars(targets, paths, fileList)
