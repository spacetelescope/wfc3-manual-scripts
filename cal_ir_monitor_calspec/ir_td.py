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
from ir_logging import display_message


MJD_I = 55008


def import_slope_file(filename, log=False, verbose=True):
    """
    Imports and formats the file containing the different
    slopes for each filter.

    Parameters
    ----------
    filename : str
        String representation of path to text file with
        slopes.
    verbose : Boolean
        Whether to print the message; defaults to True.
    log : Boolean
        Whether to log the message; defaults to False.

    Returns
    -------
    slopes : dict
        Dictionary where each key-value pair represents
        a WFC3/IR filter and the corresponding slope.
    """
    slopes = {}

    try:
        with open(filename) as f:
            display_message(log=log, verbose=verbose, log_type='info',
                            message=f'Using slope file at {filename}:')

            lines = [x.strip() for x in f.readlines()]
            for line in lines:
                split_line = line.split(',')
                slopes[split_line[0]] = float(split_line[1])
                display_message(log=log, verbose=verbose, log_type='info',
                                message=f'\t{split_line[0]} = '\
                                        f'{split_line[1]}% / year')

    except FileNotFoundError:
        display_message(log=log, verbose=verbose, log_type='critical',
                        message=f"Can't find slope file at {filename}")

    return slopes


def correct_for_td(flux, expstart, m, log=False, verbose=True):
    """
    Calculates the elapsed time between starting time
    (global parameter `MJD_I`) and the observation date,
    then uses the slope `m` to calculate the flux
    difference over the elapsed time. Finally, adds the
    flux back in in order to correct the flux.

    Parameters
    ----------
    flux : float or int
        Observed/measured flux.
    expstart : float or int
        Exposure start time in MJD.
    m : float
        Slope as a percent change per year. (Should already
        be negative.)
    verbose : Boolean
        Whether to print the message; defaults to True.
    log : Boolean
        Whether to log the message; defaults to False.

    Returns
    -------
    mag_corr : float
        Observed/measured flux, corrected for time-
        dependence according to the provided `m` slope.
    """
    dt = expstart - MJD_I   # time difference, in MJD (days)
    flux_diff = (dt / 365.25) * (m / 100) * flux

    if m < 0:
        flux_corr = flux - flux_diff    # subtract a negative = add flux back
    else:
        flux_corr = flux + flux_diff    # add a positive = add flux back

    return flux_corr
