"""
Functions to correct photometry for IR time-dependence as defined by a
calculated slope of relative sensitivity change over time.

Functions
---------
    import_slope_file()
        Imports and formats the file containing the different slopes for
        each filter.
    correct_for_td()
        Calculates the elapsed time between starting time (global parameter
        `MJD_I`) and the observation date, then uses the slope `m` to
        calculate the flux difference over the elapsed time. Finally,
        adds the flux back in in order to correct the flux.

Author
------
Mariarosa Marinelli, 2023

"""
from ir_config import CONFIG


def correct_tds(flux, expstart, filt):
    """
    Calculates the elapsed time between starting time
    and the observation date, then corrects the flux
    for time-dependent changes.

    Parameters
    ----------
    flux : float or int
        Observed/measured flux.
    expstart : float or int
        Exposure start time in MJD.

    Returns
    -------
    mag_corr : float
        Observed/measured flux, corrected for time-
        dependence according to the provided `m` slope.
    """
    dt = expstart - CONFIG['mjd_i']   # time difference, in MJD (days)
    m = CONFIG['tds_2024'][filt]

    flux_corr = flux / (1 - (m * dt))

    return flux_corr
