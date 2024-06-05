"""
    This module contains a function to correct geometric
    distortion in UVIS FLT (flat-fielded) images. By
    multiplying the data from the FLT by the appropriate
    Pixel Area Map (PAM), any given source will yield a
    count-rate equivalent to what would have been achieved
    by drizzling multiple FLTs together.

    For more background, see:
        "Pixel Area Maps"
            www.stsci.edu/hst/instrumentation/wfc3/data-analysis/pixel-area-maps
        "WFC3 Pixel Area Maps"
            WFC3 Instrument Science Report 2010-08
            Kalirai et al.
        "WFC3 Chip Dependent Photometry with the UV filters"
            WFC3 Instrument Science Report 2017-07
            Deustua et al.

    Authors
    -------
        Clare Shanahan, Dec 2019
        Mariarosa Marinelli, 2022

    Use
    ---
        This script is intended to imported:

            from wfc3_phot_tools.utils import UVIS_PAM

        or:

            from wfc3_phot_tools.utils.UVIS_PAM import make_PAMcorr_image_UVIS

"""

import copy
import numpy as np
from astropy.io import fits

def make_PAMcorr_image_UVIS(data, prihdr, scihdr, pamdir):
    """
    Corrects the geometric distortion of the input image
    data by multiplying by the correct UVIS PAM.

    Parameters
    ----------
    data : array
        ****Name of FITS file.
    pri : header
        Primary header of file for data.
    scihdr : header
        Header from science extension of data.
    pamdir : str
        Path to where pixel area maps for UVIS1 and/or
        UVIS2 are located.

    Returns
    -------
    pamcorr_data : array
        PAM-corrected data
    """

    data = copy.copy(data)
    x0 = int(np.abs(scihdr['LTV1']))
    y0 = int(np.abs(scihdr['LTV2']))
    x1 = int(x0 + scihdr['NAXIS1'])
    y1 = int(y0 + scihdr['NAXIS2'])

    if scihdr['CCDCHIP'] == 1:
        pam = fits.getdata(pamdir + 'UVIS1wfc3_map.fits')
        pamcorr_data = data * pam[y0:y1, x0:x1]

    elif scihdr['CCDCHIP'] == 2:
        pam = fits.getdata(pamdir + 'UVIS2wfc3_map.fits')
        pamcorr_data = data * pam[y0:y1, x0:x1]
    else:
        raise Exception('Chip case not handled.')

    return pamcorr_data
