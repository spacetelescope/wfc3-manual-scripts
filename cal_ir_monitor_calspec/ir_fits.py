# pylint: disable=E1101
"""
Functions for handling FITS files.

Usage
-----
    This module is designed to be imported into the
    `ir_phot_pipeline.py` script. The individual functions
    can also be imported separately.

        > import ir_fits
        > from ir_fits import get_ext_data, get_hdr_info

Author
------
    Mariarosa Marinelli, 2023
"""

from astropy.io import fits
from ir_toolbox import get_decimalyear, resolve_targnames


def get_ext_data(filepath):
    """Helper function to extract data arrays.

    Parameters
    ----------
    stare_obj : `ObsBatch`

    Returns
    -------
    data_arr, err_arr, dq_arr : array-like
        The extracted data array from each appropriate
        extension: `sci` is science data from extension 1,
        `err` is error data from extension 2, and `dq` is
        the data quality flag array from extension 3.
    """
    with fits.open(filepath) as fits_file:
        data_arr = fits_file[1].data
        err_arr = fits_file[2].data
        dq_arr = fits_file[3].data

    return data_arr, err_arr, dq_arr


def get_hdr_info(filepath):
    """Extract information from FITS headers.

    Helper function to extract needed header info into a
    dictionary.

    Parameters
    ----------
    stare_obj : `ObsBatch`

    Returns
    -------
    hdr_info : dict
        Dictionary of stripped header information, wherein
        keys correspond to header card names and values are
        the header card values.

        - cd1_1, cd1_2, cd2_1, cd2_2: partials for WCS transformations
        - photflam, photfnu, photzpt, photbw, photplam: TDS photometry
    """
    hdr_info = {}
    keywords = ['rootname', 'proposid', 'targname', 'filter', 'aperture',     # set parameters
                'expstart', 'exptime', 'linenum', 'bunit', 'scan_typ',        # observing info
                'subarray', 'subtype', 'samp_seq', 'nsamp', 'sampzero',       # ins. config. params
                'ltv1', 'ltv2', 'crpix1', 'crpix2', 'crval1', 'crval2',       # coordinates
                'cd1_1', 'cd1_2', 'cd2_1', 'cd2_2',
                'photflam', 'photfnu', 'photzpt', 'photbw', 'photplam',
                'bpixtab', 'biasfile', 'crrejtab', 'darkfile',                # calib. files
                'pfltfile', 'imphttab', 'idctab', 'mdriztab']                 # calib. files

    with fits.open(filepath) as fits_file:
        pri_hdr = fits_file[0].header
        sci_hdr = fits_file[1].header
        combined_hdr = pri_hdr + sci_hdr

        for keyword in keywords:
            if keyword == 'linenum':
                hdr_info[keyword] = str(combined_hdr[keyword.upper()])
            elif keyword == 'targname':
                hdr_info[keyword] = resolve_targnames(targname=pri_hdr[keyword],
                                                      simplify=True)
            else:
                hdr_info[keyword] = combined_hdr[keyword.upper()]

        hdr_info['expstart_decimalyear'] = get_decimalyear(hdr_info['expstart'])
        print(hdr_info['ltv1'])

        return hdr_info
