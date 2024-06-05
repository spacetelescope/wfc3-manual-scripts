from astropy.io import fits
from toolbox import display_message, setup_logging, make_timestamp
from phot_tools import detect_sources_scan, calc_sky
from io import StringIO
import sys
import logging
import os
from glob import glob

test_files = glob('/grp/hst/wfc3v/wfc3photom/data/uvis_scan_monitor/data/GD153/F218W/*_fcr.fits')

class Capturing(list):
    """
    Class to capture output from externally-imported
    functions. 
    """
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self
    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio
        sys.stdout = self._stdout

def test_log_wrapping(filepath):
    with Capturing() as output:
        with fits.open(filepath) as f:
            data = f[0].data
            sources = detect_sources_scan(data, snr_threshold=3.0,
                                      sigma_kernel=3,
                                      size_kernel=(3, 3),
                                      n_pixels=250,
                                      show=False,
                                      save=False,
                                      title='')
            print(f"Detected sources at {sources['xcentroid'][0]}, {sources['ycentroid'][0]}")

    display_message(verbose=False, log=True, message=output, log_type='info')

    with Capturing() as output:
        back, back_rms = calc_sky(data,
                                  sources['xcentroid'][0],
                                  sources['ycentroid'][0],
                                  source_mask_len=400,
                                  source_mask_width=300,
                                  n_pix=30,
                                  method='median')

    display_message(verbose=False, log=True, message=output, log_type='info')

if __name__ == '__main__':
    setup_logging(local=True,
                  log_dir=os.getcwd(),
                  log_name=make_timestamp(),
                  verbose=False,
                  log=True)

    for filepath in test_files:
        test_log_wrapping(filepath)
