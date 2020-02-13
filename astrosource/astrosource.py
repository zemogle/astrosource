from pathlib import Path
import logging

from astrosource.comparison import find_comparisons, find_comparisons_calibrated
from astrosource.identify import find_stars, gather_files
from astrosource.utils import folder_setup
from astrosource.utils import AstrosourceException

logger = logging.getLogger('astrosource')


class TimeSeries:
    def __init__(self,targets,indir,**kwargs):
        self.targets = targets
        self.indir = Path(indir)
        self.format = kwargs.get('format','fz')
        self.paths = folder_setup(self.indir)
        self.filelist, self.filtercode = gather_files(self.paths, filetype=self.format)

    def analyse(self):
        usedimages = find_stars(self.targets, self.paths, self.filelist)
        find_comparisons(self.indir, usedimages)
        # Check that it is a filter that can actually be calibrated - in the future I am considering calibrating w against V to give a 'rough V' calibration, but not for now.
        if self.filtercode in ['B', 'V', 'up', 'gp', 'rp', 'ip', 'zs']:
            try:
                find_comparisons_calibrated(self.filtercode, self.paths)
            except AstrosourceException as e:
                logger.warning(e)

        else:
            sys.stdout.write(f'⚠️ filter {self.filtercode} not supported for calibration')
