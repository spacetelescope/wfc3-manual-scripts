"""
deprecated

    This module contains a function to calculate flux
    errors in the same manner as IRAF/DAOPHOT.

    Authors
    -------
        Mariarosa Marinelli, 2022
        Clare Shanahan, Dec 2019

    Use
    ---
        This script is intended to imported:

            from wfc3_phot_tools.utils import daophot_err

        or:

            from wfc3_phot_tools.utils.daophot_err import compute_phot_err_daophot

"""

import numpy as np

def compute_phot_err_daophot(flux, back, back_rms, phot_ap_area,
                             sky_ap_area, gain=1.0):

    """
    Calculates flux errors in the same manner as
    IRAF/DAOphot.

    The error terms in this model represent Poisson noise
    from the source, Poisson noise in the sky and readout
    noise, and error in the sky measurement.

        err = sqrt(
                   (flux - back*phot_ap_area)/epadu +
                   phot_ap_area * backrms**2 +
                   phot_ap_area**2 * backrms**2 / sky_ap_area
                   )

    Where flux is the aperture sum, back is the per-pixel
    background level, epadu is the conversion factor
    between e- and adu (gain), phot_ap_area is the area of
    the photometric aperture, backrms is the uncertainty in
    the sky, and sky_ap_area is the sky aperture area. Note
    that the flux/background in the above equation are in
    ADU, but the input to this function is in e-. The
    conversion is done internally.

    Parameters
    ----------
    flux : float
        Non-sky subtracted flux of star, in electrons.
    back : float
        Per-pixel background level.
    back_rms : float
        Background RMS.
    phot_ap_area : int or float
        Area of photometric aperture.
    sky_ap_area : int or float
        Area of sky aperture.
    gain : float
        CCD gain, default 1.0.

     Returns
     --------
     errs : tuple
        (error in instrumental magnitudes, error in flux)
    """

    # convert values input in e- to ADU, as equation expects
    # sky subtract flux to isolate poisson noise from source
    flux = (flux - back*phot_ap_area) / gain                # ADU
    back = back / gain                                      # ADU / px^2      # we don't use this anywhere???
    back_rms = back_rms / gain                              # ADU / px^2

    err1 = (flux/gain)                                      # ADU^2 / e
    err2 = (phot_ap_area*back_rms**2)                       # ADU^2
    err3 = (phot_ap_area**2 * back_rms**2) / sky_ap_area    # ADU^2 / px^2

    flux_err_adu = np.sqrt(np.abs(err1 + err2 + err3))      # ADU * np.sqrt(np.abs(1+1/e+1/px^2))
    flux_err = flux_err_adu * gain                          # e

    return flux_err
