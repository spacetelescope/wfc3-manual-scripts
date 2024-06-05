

import os
import glob
import time
import shutil
import copy
#import json
import numpy as np
from astropy.convolution import convolve, Gaussian2DKernel
from astropy.io import ascii, fits
from astropy.stats import sigma_clipped_stats, gaussian_fwhm_to_sigma
from astropy.table import Table, vstack
from astropy.time import Time
from astroquery.mast import Observations
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import photutils.segmentation as phot_seg

import cr_reject as cr_reject # the scripts!
import phot_tools as phot_tools
import UVIS_PAM as uvis_pam
import daophot_err as daophot_err

from pyql.database.ql_database_interface import session
from pyql.database.ql_database_interface import Master, UVIS_flt_0, Anomalies
#from housekeeping import check_subdirectory
from toolbox import (CaptureOutput, check_subdirectory, display_message,
                     InteractiveArgs, make_timestamp, parse_args, setup_logging)

PAM_DIR = '/grp/hst/wfc3v/wfc3photom/data/pamfiles/'
MONITOR_DIR = '/grp/hst/wfc3v/wfc3photom/data/uvis_scan_monitor'

#trial_dir_name = '2023_02_21_test1' # median, sigmaclip
#trial_dir_name = '2023_02_21_test2' # median, sigma_clipped_stats
#trial_dir_name = '2023_02_21_test3' # mean, sigma_clipped_stats --> what I was using originally?
#trial_dir_name = '2023_02_21_test4' # mean, sigmaclip
#trial_dir_name = '2023_02_21_test5' # mean, sigma_clipped_stats, 30px   ---> YES!!!
#trial_dir_name = '2023_02_21_test6' # mean, sigmaclip, 30px
#trial_dir_name = '2023_02_22_test_grw70' # mean, sigmaclip, 30px
#trial_dir_name = '2023_02_27_test_grw70' # mean, sigmaclip, 30px
trial_dir_name = '2023_03_06_test1'
#trial_dir = '/grp/hst/wfc3v/wfc3photom/data/uvis_scan_monitor/refactor_old_fcrs'

def assess_scan_quality(data, ap_info, verbose, log, plot=True):
    """
    This function determines whether or not a scan should be
    used for photometry on the basis of four assessments:
        1. If a source is detected. If one is not detected,
           the `detect_sources()` function will be returned
           as a NoneType object.
        2. If the entirety of the sky background rind will
           fit on the subarray when centered on a detected
           source. Evaluated by `check_rind_parameters()`.
        3. If the angle of the detected source is within a
           5 degree offset from the vertical. If not, the
           corresponding scan may have been affected by
           loss of lock during the exposure. Evaluted by
           `check_scan_angle()`.
        4. If the shape of a 2D Gaussian function with the
           same second-order central moments as the
           detected source is not as expected, which may
           be judged by the following:
               - If the eccentricity (the fraction of
                 distance along the semimajor axis at which
                 the focus lies) of said Gaussian is less
                 than 0.98.
               - If the ratio of the 1-sigma standard
                 deviations along the semimajor axis to the
                 semiminor axis is less than 20.
           Evaluated by `check_source_shape()`.

    Parameters
    ----------
    data : array-like
        Input data array.
    ap_info : dict
        Dictionary of aperture photometry parameter
        information.
    sky_ap_dim : list of int
        The dimensions of the inner boundary of the sky
        background rind, in format [x_pixels, y_pixels].
    sky_thickness : int
        Width of the sky background rind in pixels.
    verbose : Boolean
        Whether to print the message.
    log : Boolean
        Whether to log the message.
    plot : Boolean
        Whether or not to show a source detection plot.

    Returns
    -------
    use_scan : Boolean
        If `False`, indicates that the scan should not be
        used for photometry.
    tbl : `astropy.table.table.QTable` or NoneType
        If a source is detected, this will return a QTable
        of its properties. If no source is detected, will
        return None.
    """
    use_scan = True

    threshold = phot_seg.detect_threshold(data, nsigma=5)
    seg_img = phot_seg.detect_sources(data, threshold=threshold, npixels=250)

    if seg_img is None:
        display_message(verbose=verbose,
                        log=log,
                        log_type='error',
                        message='Could not find a source. Will not use this '\
                                'observation for photometry.')
        use_scan = False
        tbl = None

    else:
        cat = phot_seg.SourceCatalog(data, seg_img)
        tbl = cat.to_table()

        rind_fits = check_rind_parameters(data,
                                          tbl['xcentroid'][0],
                                          tbl['ycentroid'][0],
                                          ap_info['sky_ap_dim'],
                                          ap_info['sky_thickness'],
                                          verbose, log)
        angle_good = check_scan_angle(tbl['orientation'][0].value, verbose, log)
        shape_good = check_source_shape(tbl['eccentricity'][0].value,
                                        tbl['semimajor_sigma'][0].value,
                                        tbl['semiminor_sigma'][0].value,
                                        verbose, log)

        if rind_fits and angle_good and shape_good:
            use_scan = True
        else:
            use_scan = False
            display_message(verbose=verbose,
                            log=log,
                            log_type='error',
                            message='\tWill not use this scan for photometry.')

        if plot:
            fig, ax = plt.subplots(1,2,figsize=(14,7))
            ax[0].imshow(data, origin='lower', norm=LogNorm())
            ax[1].imshow(seg_img.data, origin='lower')
            plt.show()
            plt.close()

    return use_scan, tbl


def calc_phot_wrapper(scan_obj, data_type, verbose, log, show=False):
    """
    Wrapper for calculating the sky-subtracted photometry
    inside a photometric aperture, as well as the error.

    Parameters
    ----------
    scan_obj :  `obsScan`
        A scan observation object constructed through the
        class `obsScan`.
    data_type : string
        Either 'flt' or 'fcr', denoting which data
        extension to use (flt_data or fcr_data, both of
        which are in counts per second).
    verbose : Boolean
        Whether to print the message.
    log : Boolean
        Whether to log the message.
    show : Boolean
        Whether to show the plot of the scan or not.

    Returns
    -------
    flux : float
        Total flux inside the photometric aperture minus
        the product of the sky background (median or mean)
        flux and the number of pixels in the photometric
        aperture (photometric aperture area), in units of
        counts per second.
    flux_err : float
        Error in the calculation of the sky-subtracted
        photometric flux, in units of counts per second.
    """
    if show:
        plt_title=f'{scan_obj.header_info["rootname"]} - {file_type.upper()}'
    else:
        plt_title=None

    if data_type == 'flt':
        data_attr = scan_obj.flt_data
    else:
        data_attr = scan_obj.fcr_data

    phot_table = phot_tools.aperture_photometry_scan(data_attr,
                                                         x_pos=scan_obj.x_pos,
                                                         y_pos=scan_obj.y_pos,
                                                         ap_width=scan_obj.ap_info['ap_dim'][0],
                                                         ap_length=scan_obj.ap_info['ap_dim'][1],
                                                         theta=scan_obj.theta,
                                                         show=show,
                                                         plt_title=plt_title)

    flux_uncorr = phot_table['aperture_sum'][0]

    display_message(verbose=args.verbose,
                    log=args.log,
                    message='Uncorrected flux in photometric aperture:\n'\
                            f'\t{flux_uncorr} electrons/second\n'\
                            f'\t{flux_uncorr*scan_obj.header_info["exptime"]} electrons',
                    log_type='info')

    # convert source sum back to electrons for error calculation
    flux_uncorr_e = scan_obj.header_info['exptime'] * flux_uncorr

    # convert background measurements back to electrons for error calculations
    if data_type == 'flt':
        back_e = scan_obj.flt_back * scan_obj.header_info['exptime']
        back_rms_e = scan_obj.flt_back_rms * scan_obj.header_info['exptime']

    else:
        back_e = scan_obj.header_info['exptime'] * scan_obj.fcr_back
        back_rms_e = scan_obj.header_info['exptime'] * scan_obj.fcr_back_rms

    display_message(verbose=args.verbose,
                    log=args.log,
                    message=f'Background {scan_obj.back_method} level in sky aperture:\n'\
                            f'\t{back_e / scan_obj.header_info["exptime"]} electrons/second\n'\
                            f'\t{back_e} electrons',
                    log_type='info')

    flux_err = daophot_err.compute_phot_err_daophot(flux=flux_uncorr_e,
                                                    back=back_e,
                                                    back_rms=back_rms_e,
                                                    phot_ap_area=scan_obj.ap_info['ap_area'],
                                                    sky_ap_area=scan_obj.ap_info['sky_ap_area'])

    # convert error to count rate
    flux_err = flux_err / scan_obj.header_info['exptime']

    display_message(verbose=args.verbose,
                    log=args.log,
                    message='Subtracting background for all pixels in sky aperture:\n'\
                            f'\t{(back_e * scan_obj.ap_info["ap_area"]) / scan_obj.header_info["exptime"]} electrons/second\n'\
                            f'\t{back_e * scan_obj.ap_info["ap_area"]} electrons',
                    log_type='info')

    # subtract background median times photometric area,
    # then convert back to count rate
    flux_e = flux_uncorr_e - (back_e * scan_obj.ap_info['ap_area'])

    flux = flux_e / scan_obj.header_info['exptime']

    display_message(verbose=args.verbose,
                    log=args.log,
                    message='Total sky-subtracted flux in photometric aperture:\n'\
                            f'\t{flux} electrons/second\n'\
                            f'\t{flux_e} electrons',
                    log_type='info')

    return flux, flux_err


def calc_sky_ap_area(sky_ap_dim, sky_thickness):
    """
    Helper function to calculate the area of the sky
    background aperture given the inner dimensions and the
    width of the frame/rind.

    Parameters
    ----------
    sky_ap_dim : list of int
        The dimensions of the inner boundary of the sky
        background rind, in format [x_pixels, y_pixels].
    sky_thickness : float or int
        Depth or width of the frame or rind for the sky
        background aperture.

    Returns
    -------
    sky_ap_area : float or int
        The total area of the sky background frame or rind
        in units of squared pixels.
    """
    inner_area = sky_ap_dim[0] * sky_ap_dim[1]
    outer_area = (sky_ap_dim[0] + (2*sky_thickness)) * (sky_ap_dim[1] + (2*sky_thickness))
    sky_ap_area = outer_area - inner_area

    return sky_ap_area


def calc_sky_wrapper(scan_obj, data_type, verbose, log):
    """
    Wrapper for calculating the background sky level per
    pixel.

    Parameters
    ----------
    scan_obj : `obsScan`
        A scan observation object constructed through the
        class `obsScan`.
    data_type : string
        Either 'flt' or 'fcr', denoting which data
        extension to use (flt_data or fcr_data, both of
        which are in counts per second).
    verbose : Boolean
        Whether to print the message.
    log : Boolean
        Whether to log the message.

    Returns
    -------
    back : float
        Either the median or mean of the pixels in the
        sky background aperture, in units of counts per
        second.
    back_rms : float
        The standard deviation of the pixel background
        level inside the sky aperture, in units of counts
        per second.
    """
    if data_type == 'flt':
        data_attr = scan_obj.flt_data
    else:
        data_attr = scan_obj.fcr_data

    with CaptureOutput() as output:
        back, back_rms = phot_tools.calc_sky(data=data_attr,
                                             x_pos=scan_obj.x_pos,
                                             y_pos=scan_obj.y_pos,
                                             source_mask_len=scan_obj.ap_info['sky_ap_dim'][1],
                                             source_mask_width=scan_obj.ap_info['sky_ap_dim'][0],
                                             n_pix=scan_obj.ap_info['sky_thickness'],   # this needs to remain n_pix
                                             method=scan_obj.back_method)

    display_message(verbose=args.verbose,
                    log=args.log,
                    message=output,
                    log_type='info')

    return back, back_rms


def check_file(filepath, verbose, log):
    """
    Helper function to check the filepath and file type of
    a given file. Raises an exception if file does not
    exist. Raises an exception if file exists but is not
    either an FCR or FLT file, as determined from the name
    of the file.

    Parameter
    ---------
    filepath : string
        String representation of path to a file.
    verbose : Boolean
        Whether to print the message
    log : Boolean
        Whether to log the message.

    Returns
    -------
    file_type : string
        Should be either `flt` or `fcr`, depending on the
        name of the filepath
    data_ext : int
        For an FCR file, this is 0 since they only have one
        data extension. For an FLT file, this will be 1,
        since the zeroth extension is the primary header.
    """
    if os.path.exists(filepath):
        file_type = os.path.basename(filepath).split('.fits')[0][-3:]
        data_exts = {'fcr': 0, 'flt': 1}
        try:
            data_ext = data_exts[file_type]
            return file_type, data_ext

        except KeyError as ke:
            display_message(verbose=verbose,
                            log=log,
                            message='File does not appear to be either an '\
                                    f'FLT or FCR file: {os.path.basename(filepath)}',
                            log_type='error')
            return None, None
    else:
        display_message(verbose,
                        log,
                        message=f'Specified filepath does not exist {filepath}',
                        log_type='error')
        return None, None


def check_rind_parameters(data, scan_x, scan_y, ap_info, sky_ap_dim, sky_thickness, verbose, log):
    """
    Helper function to make sure that the entirety of the
    sky rind is in the subarray.

    Parameters
    ----------
    data : array-like
        Input data array.
    scan_x, scan_y : floats
        The center of the detected source, in pixels.
    sky_ap_dim : list of int
        The dimensions of the inner boundary of the sky
        background rind, in format [x_pixels, y_pixels].
    sky_thickness : int
        Width of the sky background rind in pixels.
    verbose : Boolean
        Whether to print the message.
    log : Boolean
        Whether to log the message.

    Returns
    -------
    rind_fits : Boolean
        If `False`, the sky background rind cannot fit on
        the subarray when centered on the detected source,
        indicating that the scan should be rejected.
    """
    data_dims = data.shape

    ap_y = sky_ap_dim[1]/2.   # 400
    ap_x = sky_ap_dim[0]/2.   # 300

    x_l = scan_x - ap_x - sky_thickness
    x_r = scan_x + ap_x + sky_thickness

    y_b = scan_y - ap_y - sky_thickness
    y_t = scan_y + ap_y + sky_thickness


    if (x_l > 10) and \
       (y_b > 10) and \
       (x_r < (data_dims[0] - 10)) and \
       (y_t < (data_dims[1] - 10)):
        rind_fits = True

    else:
        rind_fits = False
        display_message(verbose=verbose,
                        log=log,
                        log_type='error',
                        message='\tSky background rind will not '\
                                'fit in the data array.')

    return rind_fits


def check_scan_angle(orientation, verbose, log):
    """
    Function to check the angle of the scan with respect
    to the vertical, to see if the scan failed.

    Parameter
    ---------
    orientation : float
        The angle between the x-axis and the major axis of
        a 2D Gaussian function that has the same second-
        order moments as the detected source, in degrees.
    verbose : Boolean
        Whether to print the message.
    log : Boolean
        Whether to log the message.

    Returns
    -------
    angle_good : Boolean
        If `False`, then the angle of the detected source
        indicates that the scan should be rejected.
    """
    if orientation < 0:              # negative
        offset = 90 + orientation    # positive offset
    else:                            # positive
        offset = orientation - 90    # negative offset

    if np.abs(offset) < 5:
        angle_good = True
    else:
        angle_good = False
        display_message(verbose=verbose,
                        log=log,
                        log_type='error',
                        message='\tScan angle is offset from the vertical by '\
                                f'{offset:.4f} degrees.')

    return angle_good


def check_source_shape(eccentricity, semimajor_sigma, semiminor_sigma, verbose, log):
    """
    Function to check the shape of the detected source of a
    scan, to see if the eccentricity isn't as expected.

    Parameters
    ----------
    eccentricity : float
        The eccentricity of the 2D Gaussian function that
        has the same second-order moments as the source.
    semimajor_sigma : float
        1-sigma standard deviation along the semimajor axis
        of the 2D Gaussian function with the same second-
        order central moments as the source, in pixels.
    semiminor_sigma : float
        1-sigma standard deviation along the semiminor axis
        of the 2D Gaussian function with the same second-
        order central moments as the source, in pixels.
    verbose : Boolean
        Whether to print the message.
    log : Boolean
        Whether to log the message.

    Returns
    -------
    shape_good : Boolean
        If `False`, then the shape of the detected sources
        indicates that the scan should be rejected.
    """
    if eccentricity < 0.98:
        shape_good = False
    else:
        shape_good = True

    if not shape_good:
        display_message(verbose=verbose,
                        log=log,
                        log_type='error',
                        message=f'\tSource eccentricity is {eccentricity:.6f}')

    return shape_good


def get_decimalyear(mjd):
    """
    Helper function to convert a Modified Julian Date into
    the decimalyear format.

    Parameter
    ---------
    mjd : int or float
        Some date/time in MJD format.

    Returns
    -------
    decimalyear : float
        The date/time in decimalyear format.
    """
    decimalyear = Time(mjd, format='mjd').to_value(format='decimalyear')
    return decimalyear


def get_header(data_ext, with_open_file):
    """
    Helper function to get information from header.

    Parameters
    ----------
    data_ext : int
        Either 0 (for FCR files) or 1 (for FLT files).
    with_open_file : `fits.hdu.hdulist.HDUList`
        Opened fits file.

    Returns
    -------
    hdr : `fits.header.Header`
        Header from the fits file, composed of the items in
        the zeroth header, plus, in the case of an FLT file,
        the items in the first header.
    """
    hdr = with_open_file[0].header

    if data_ext != 0:
        sci_hdr = with_open_file[data_ext].header
        hdr = sci_hdr + hdr

    return hdr


def get_header_info(scan_obj, verbose, log):
    """
    Helper function to extract needed header info into a
    dictionary. Also resolves target names as stated in
    fits file header using the `resolve_targnames()`
    function.

    Parameters
    ----------
    scan_obj :  `obsScan`
        A scan observation object constructed through the
        class `obsScan`.
    verbose : Boolean
        Whether to print the message.
    log : Boolean
        Whether to log the message.

    Returns
    -------
    header_info : dict
        Dictionary of stripped header information, wherein
        keys correspond to header card names and values are
        the header card values.
    """
    header_info = {}
    keywords = ['rootname', 'proposid', 'targname', 'filter', 'aperture',       # set parameters
                'expstart', 'exptime', 'linenum',                               # observing info
                'ccdamp', 'ccdgain', 'ccdofsta', 'ccdofstc',                    # engineering parameters
                'atodgna', 'readnsea', 'biasleva',                              # calibrated engineering parameters
                'atodgnc', 'readnsec', 'biaslevc',                              # calibrated engineering parameters
                'ang_side', 'scan_ang', 'scan_rat', 'scan_len',                 # scan keywords
                'flashdur', 'flashcur', 'flashlvl', 'shutrpos',                  # postflash parameters
                'photflam', 'phtflam1', 'phtflam2', 'phtratio',                 # time-dependent phot. cal.
                'photfnu', 'photzpt', 'photbw', 'photplam',                     # time-dependent phot. cal.
                'mdrizsky',                                                     # sky background as calc. by AstroDrizzle
                'bpixtab', 'biasfile', 'flshfile', 'darkfile',                  # calibration files
                'pfltfile', 'imphttab', 'drkcfile', 'snkcfile']                 # calibration files
    for keyword in keywords:
        if keyword == 'targname':
            header_info[keyword] = resolve_targnames(targname=scan_obj.hdr[keyword.upper()],
                                                     verbose=verbose,
                                                     log=log)
        header_info[keyword] = scan_obj.hdr[keyword.upper()]

    header_info['expstart_decimalyear'] = get_decimalyear(header_info['expstart'])

    return header_info


def resolve_targnames(targname, simplify=True, verbose=True, log=False):
    """
    Helper functions to resolve target names. Sometimes
    what is put into APT is not the simplest form of a
    target's name so this helps make sure that everything
    is consistent. Or, in the case of searching MAST, we
    want all possible versions of the target name.

    Parameter
    ---------
    targname : string
        Name of target.
    simplify : Boolean
        When set to `True`, finds the simplified version of
        the input target name. If set to `False`, returns a
        list of all possible/accepted names for the input
        target name.
    verbose : Boolean
        Whether to print the message.
    log : Boolean
        Whether to log the message.

    Returns
    -------
    resolved : string or list
        Resolved target name(s). If unable to resolve the
        name (ex. it's a different target altogether, or a
        weird spelling), the original input `targname` will
        be returned instead. In `simplify` mode, it will
        return either GD153 or GRW70, the simplest versions
        of the two main target names for this monitor. If
        not in `simplify` mode, it will return a list of
        possible names for searching in MAST.
    """
    targnames = {'GD153': 'GD153',
                 'GD-153': 'GD153',
                 'GRW+70D5824': 'GRW70',
                 'GRW+70D': 'GRW70',
                 'GRW70': 'GRW70'}
    if simplify:
        try:
            resolved_targname = targnames[targname]
            resolved = resolved_targname
        except KeyError as ke:
            display_message(verbose=verbose,
                            log=log,
                            message=f'Warning! Unable to resolve name for {ke}',
                            log_type='warning')
            resolved = targname
    else:
        resolved_targnames = [k for k, v in targnames.items() if v == targname]
        if len(resolved_targnames) > 0:
            resolved = resolved_targnames
        else:
            resolved = targname

    return resolved

def remove_non_scans(prods_p_t_f, proposal_ql_roots):
    """
    Helper function that removes observations that are not
    scans from the table of data products that will be
    downloaded.

    Parameters
    ----------
    prods_p_t_f :  `astropy.table.table.Table`
        Table of data products of a particular program,
        target, and filter subset.
    proposal_ql_roots : list
        List of `ql_root`s (rootname without transmission
        character) corresponding to scans in the given proposal.

    Returns
    -------
    prods_p_t_f :  `astropy.table.table.Table`
        The original table of data products, with any
        products corresponding to non-scan observations
        removed.
    """
    # make a new column in table for the observation mode
    prods_p_t_f['obsmode'] = ['scan'
                              if prod['productFilename'].split('_')[0][:-1]
                              in proposal_ql_roots
                              else 'stare'
                              for prod in prods_p_t_f]

    # remove non-scan products from table
    prods_p_t_f = prods_p_t_f[prods_p_t_f['obsmode'] == 'scan']

    return prods_p_t_f


def redownload_wrapper(prods_p_t_f, dir_p_t_f, redownload_data_flag,
                       verbose=True, log=False):
    """
    Function to remove already-existing data products from
    MAST data products table if redownloading files is not
    desired.

    Parameters
    ----------
    prods_p_t_f :  `astropy.table.table.Table`
        Table of data products of a particular program,
        target, and filter subset.
    dir_p_t_f : str
        String representation of directory path for a
        particular program, target, and filter subset of
        data.
    redownload_data_flag : Boolean
        Whether or not to redownload the data.
    verbose : Boolean
        Whether to print the message.
    log : Boolean
        Whether to log the message.

    Returns
    -------
    prods_p_t_f :  `astropy.table.table.Table`
        Table of data products of a particular program,
        target, and filter subset, with already-existing
        data products removed if `redownload_data_flag` is
        set to `False`.
    continue_download : Boolean
        Whether or not to continue with the data download.
        Set to `False` if there are no data products left
        (`prods_p_t_f` is empty).
    """
    planned_filenames = prods_p_t_f['productFilename']

    # if you don't want to redownload existing files:
    if not redownload_data_flag:
        for planned_filename in planned_filenames:
            planned_path = os.path.join(dir_p_t_f,
                                        os.path.basename(planned_filename))
            if os.path.exists(planned_path):
                display_message(verbose=verbose,
                                log=log,
                                message=f'Found existing file at {planned_path}.',
                                log_type='info')
                prods_p_t_f = prods_p_t_f[prods_p_t_f['productFilename'] != planned_filename]

    number_removed = len(planned_filenames) - len(prods_p_t_f)
    if number_removed == 0:
        display_message(verbose=verbose,
                        log=log,
                        message=f'Commencing download of {len(prods_p_t_f)} files...',
                        log_type='info')
        continue_download = True
    else:
        if len(prods_p_t_f) == 0:
            display_message(verbose=verbose,
                            log=log,
                            message='All files in download queue already exist.',
                            log_type='info')
            continue_download = False
        else:
            display_message(verbose=verbose,
                            log=log,
                            message=f'Removed {number_removed} files. '\
                                    f'Commencing download of {len(prods_p_t_f)} files...',
                            log_type='info')
            continue_download = True

    return prods_p_t_f, continue_download


def retrieve_scan_data(dirs, verbose, log, **params):
    """
    Function to query MAST and download observations to the
    proper location.

    Parameters
    ----------
    dirs : dict
        Dictionary of directories.
    verbose : Boolean
        Whether to print the message.
    log : Boolean
        Whether to log the message.

    **params
        proposal_id : int, list of int
            Proposal IDs or list of proposal IDs desired.
        filters : str, list of str
            Filter name or list of filter names desired.
        target_name : str, list of str
            Target name or list of target names desired.

    Returns
    -------
    download_manifest :
    """
    display_message(verbose=verbose,
                    log=log,
                    message=f'Querying MAST for data matching specified parameters:\n{params}',
                    log_type='info')

    download_manifest = Table()
    obs = Observations.query_criteria(**params)
    display_message(verbose=verbose,
                    log=log,
                    log_type='info',
                    message=f'Found {len(obs)} matching observations.')

    if len(obs) > 0:
        # 3 levels of organization:
        # make list of tables by proposal
        proposals = sorted(list(set(obs['proposal_id'])))
        obs_ps = [obs[obs['proposal_id'] == p] for p in proposals]

        for proposal, obs_p in zip(proposals, obs_ps):
            # check quicklook for all scans in that proposal:
            proposal_ql = session.query(UVIS_flt_0.ql_root).\
                                  filter(UVIS_flt_0.proposid == proposal).\
                                  filter(UVIS_flt_0.scan_typ == 'C').all()

            proposal_ql_roots = [r.ql_root for r in proposal_ql]

            dir_p = check_subdirectory(parent_dir=dirs['data'],
                                       sub_name=proposal,
                                       verbose=verbose,
                                       log=log)

            # resolve (simplify) target names:
            obs_p['resolved_target_name'] = [resolve_targnames(t) for t in obs_p['target_name']]

            # make list of tables by target for each proposal
            targets = sorted(list(set(obs_p['resolved_target_name'])))
            obs_p_ts = [obs_p[obs_p['resolved_target_name'] == t] for t in targets]

            for target, obs_p_t in zip(targets, obs_p_ts):
                dir_p_t = check_subdirectory(parent_dir=dir_p,
                                             sub_name=target,
                                             verbose=verbose,
                                             log=log)

                # make list of tables by filter for each proposal/target
                filters = sorted(list(set(obs_p_t['filters'])))
                obs_p_t_fs = [obs_p_t[obs_p_t['filters'] == f] for f in filters]

                for filt, obs_p_t_f in zip(filters, obs_p_t_fs):
                    dir_p_t_f = check_subdirectory(parent_dir=dir_p_t,
                                                   sub_name=filt,
                                                   verbose=verbose,
                                                   log=log)

                    # get all products in the proposal/target/filter table
                    all_prods_p_t_f = Observations.get_product_list(obs_p_t_f)

                    # filter to only the FLT files
                    prods_p_t_f = Observations.filter_products(all_prods_p_t_f,
                                                               productSubGroupDescription='FLT')
                    display_message(verbose=verbose,
                                    log=log,
                                    log_type='info',
                                    message='Filtered to just FLTs')

                    # remove non scans
                    prods_p_t_f = remove_non_scans(prods_p_t_f, proposal_ql_roots)

                    # If the file already exists in desired location,
                    # remove from product list if redownload is set to False
                    prods_p_t_f, continue_download = redownload_wrapper(prods_p_t_f,
                                                                        dir_p_t_f,
                                                                        args.redownload_data,
                                                                        args.verbose,
                                                                        args.log)

                    if continue_download:
                        manifest = Observations.download_products(prods_p_t_f)

                        for prod in manifest:
                            new_path = os.path.join(dir_p_t_f,
                                                    os.path.basename(prod['Local Path']))
                            current_path = os.path.join(os.getcwd(),
                                                        prod['Local Path'][2:])
                            shutil.move(current_path, new_path)

                        download_manifest = vstack([download_manifest, manifest])

    else:
        download_manifest = Table()

    return download_manifest


def check_ql_gsfail(scan_ql_roots):
    """
    Checks the Quicklook Database for any tagged guidestar
    failures under a list of ql_root IDs, and returns a
    list of any known failed observations.

    Parameter
    ---------
    scan_ql_roots : list
        List of first 8 characters of scan rootnames.

    Returns
    -------
    known_scan_failures : list
        List of first 8 characters of scan rootnames
        that have been tagged as a guidestar failure in
        Quicklook
    """
    all_gs_fails = session.query(Anomalies.ql_root).\
                           filter(Anomalies.guidestar_failure == 1).all()
    all_gs_fail_roots = [result.ql_root for result in all_gs_fails]

    known_scan_failures = [ql_root for ql_root in scan_ql_roots
                           if ql_root in all_gs_fail_roots]

    return known_scan_failures


def get_new_data(args, dirs):
    """
    Executes MAST search based on specified target(s),
    filter(s), and proposal(s). Resolves target names if
    necessary.

    Parameters
    ----------
    args : `argparse.Namespace` or `InteractiveArgs`
        Arguments.
    dirs : dict
        Dictionary of directories.
    """
    if args.targets == 'core':
        if args.filters == 'core':
            download_manifest = retrieve_scan_data(dirs,
                                                   verbose=args.verbose,
                                                   log=args.log,
                                                   proposal_id=args.proposals)
        else:
            download_manifest = retrieve_scan_data(dirs,
                                                   verbose=args.verbose,
                                                   log=args.log,
                                                   proposal_id=args.proposals,
                                                   filters=filters)
    else:
        search_targets = []
        for targname in args.targets:
            search_targets.extend(resolve_targnames(targname, simplify=False))

        if args.filters == 'core':
            download_manifest = retrieve_scan_data(dirs,
                                                   verbose=args.verbose,
                                                   log=args.log,
                                                   proposal_id=args.proposals,
                                                   target_name=search_targets)
        else:
            download_manifest = retrieve_scan_data(dirs,
                                                   verbose=args.verbose,
                                                   log=args.log,
                                                   proposal_id=args.proposals,
                                                   target_name=search_targets,
                                                   filters=args.filters)

    return download_manifest


def run_process_wrapper(args, dirs):
    """
    Wrapper for verifying file structure and existence
    before cosmic ray rejection/interpolation and/or
    photometry are attempted.

    Parameters
    ----------
    args : `argparse.Namespace` or `InteractiveArgs`
        Arguments.
    dirs : dict
        Dictionary of directories.
    """
    run_flags = [args.run_cr_reject, args.run_ap_phot]

    process_names = {[False, False]: None,
                     [True, False]: 'cosmic ray rejection',
                     [False, True]: 'photometry',
                     [True, True]: 'cosmic ray rejection and photometry'}

    process_name = process_names[run_flags]

    if process_name == None:
        display_message(verbose=args.verbose,
                        log=args.log,
                        log_type='info',
                        message='Cosmic ray rejection and aperture photometry '\
                                'flags are both set to `False`.')
    else:
        proposals = sorted([os.path.basename(x)
                            for x in glob.glob(f'{dirs["data_dir"]}/*')])

        for proposal in proposals:
            if int(proposal) in args.proposals:
                display_message(verbose=args.verbose,
                                log=args.log,
                                log_type='info',
                                message=f'Beginning {process_name} for Program {proposal}')

                targets = sorted([os.path.basename(x)
                                  for x in glob.glob(f'{dirs["data_dir"]}/{proposal}/*')])

                for target in targets:
                    if target in args.targets:
                        display_message(verbose=args.verbose,
                                        log=args.log,
                                        log_type='info',
                                        message=f'Beginning {process_name} for {target}')

                        filters = sorted([os.path.basename(x)
                                          for x in glob.glob(f'{dirs["data_dir"]}/{proposal}/{target}/*')])

                        for filt in filters:
                            if filt in args.filters:
                                display_message(verbose=args.verbose,
                                                log=args.log,
                                                log_type='info',
                                                message=f'Beginning {process_name} for {filt}')

                                if args.file_type == 'flt':   # if we want anything from FLT files, then we'll start with the FLT files
                                    display_message(verbose=args.verbose,
                                                    log=args.log,
                                                    log_type='info',
                                                    message='Starting with FLT files.')
                                    filepaths = glob.glob(f'{dirs["data_dir"]}/{proposal}/{target}/{filt}/*flt.fits')
                                else: # otherwise, start with the FCR files
                                    display_message(verbose=args.verbose,
                                                    log=args.log,
                                                    log_type='info',
                                                    message='Starting with FCR files.')
                                    filepaths = glob.glob(f'{dirs["data_dir"]}/{proposal}/{target}/{filt}/*fcr.fits')

                                if len(filepaths) > 0:
                                    display_message(verbose=args.verbose,
                                                    log=args.log,
                                                    log_type='info',
                                                    message='Will process the following files: ')
                                    filepaths = np.roll(sorted(filepaths), -10)
                                    for filepath in filepaths:
                                        display_message(verbose=args.verbose,
                                                        log=args.log,
                                                        log_type='info',
                                                        message=f'\t{filepath.split("/")[-1]}')

                                    run_process(filepaths=filepaths,
                                                args=args,
                                                dirs=dirs,
                                                phot_table_name=f'{proposal}_{target}_{filt}.csv',
                                                write_loc=f'{dirs["output_dir"]}',
                                                write=True,
                                                overwrite=True)


                                else:
                                    display_message(verbose=args.verbose,
                                                    log=args.log,
                                                    log_type='info',
                                                    message=f'Did not find any matching files in {dirs["data_dir"]}/{proposal}/{target}/{filt}')
                            else:
                                display_message(verbose=args.verbose,
                                                log=args.log,
                                                log_type='info',
                                                message=f"Skipping {proposal}/{target}/{filt} in data directory since it's not specified in the parameters.")
                    else:
                        display_message(verbose=args.verbose,
                                        log=args.log,
                                        log_type='info',
                                        message=f"Skipping {proposal}/{target} in data directory since it's not specified in the parameters.")
                else:
                    display_message(verbose=args.verbose,
                                    log=args.log,
                                    log_type='info',
                                    message=f"Skipping {proposal} in data directory since it's not specified in the parameters")


def set_tbl_path(phot_table_name, write_loc, write, overwrite,
                 verbose=True, log=False):
    """
    Helper function to set up table location. This is only run if
    `write` is True.

    Parameters
    ----------
    write : Boolean
    write_loc : str
    phot_table_name : str
    overwrite : Boolean
    """
    if write_loc[-1] == '/':
        write_loc = write_loc[:-1]

    if phot_table_name[-4:] != '.csv':
        phot_table_name = f'{phot_table_name}.csv'

    if not os.path.exists(write_loc):
        display_message(verbose=verbose,
                        log=log,
                        log_type='warning',
                        message=f'Warning: Nonexistent path {write_loc}\n'\
                                'Using current working directory instead.')
        write_loc = os.getcwd()

    full_tbl_path = f'{write_loc}/{phot_table_name}'

    if os.path.exists(full_tbl_path):
        if overwrite:
            display_message(verbose=verbose,
                            log=log,
                            log_type='warning',
                            message=f'Warning: Existing table at {full_tbl_path} and '\
                                    '`overwrite` is set to True.')
        else:
            exception_message = f'Existing table at {full_tbl_path} but '\
                                '`overwrite` is set to False.\n'\
                                'Aborting run. Please try again with '\
                                'compatible arguments.'
            display_message(verbose=verbose,
                            log=log,
                            log_type='critical',
                            message=exception_message)
            raise Exception(exception_message)

    return full_tbl_path

def remove_failed_scans(filepaths, verbose, log):
    """
    Helper function to remove known guidestar failures from
    a list of files, based on flags in the Quicklook
    Database.

    Parameters
    ----------
    filepaths : list
        List of filepaths.
    verbose : Boolean
        Whether to print the message.
    log : Boolean
        Whether to log the message.

    Returns
    -------
    filepaths : list
        List of filepaths from which any scans with known
        guidestar failures have been been removed.
    """

    ql_roots = [os.path.basename(f).split('_')[0][:-1] for f in filepaths]
    failures = check_ql_gsfail(ql_roots)
    filepaths = [f for f in filepaths
                 if os.path.basename(f).split('_')[0][:-1] not in failures]

    display_message(verbose=verbose,
                    log=log,
                    log_type='info',
                    message=f'Removed {len(failures)} scans affected by guide star failures.')

    display_message(verbose=verbose,
                    log=log,
                    log_type='info',
                    message=f'Beginning processing set of {len(filepaths)} files.')

    return filepaths


def run_process(filepaths,
                args,
                dirs,
                phot_table_name='table.csv',
                write_loc=os.getcwd(),
                write=True,
                overwrite=False):
    """
    Parameters
    ----------
    filepaths : list
        List of filepaths for FLT or FCR files.
    args : `argparse.Namespace` or `InteractiveArgs`
        Arguments.
    dirs : dict
        Dictionary of directories.
    phot_table_name : str
        File name to save the resulting photometry table.
    write_loc : str
        Location for saving the photometry table. Defaults
        to current working directory.
    write : Bool
        Whether to save the photometry table. Defaults to
        `True`.
    overwrite : Bool
        Whether to overwrite existing photometry table.
        Defaults to `False`.

    Returns
    -------
    phot_table : `astropy.table.table.Table`
        Table of photometry corresponding to the files
        indicated by `filepaths`.
    """
    if write:
        full_tbl_path = set_tbl_path(phot_table_name=phot_table_name,
                                     write_loc=write_loc,
                                     write=True,
                                     overwrite=overwrite)   # aborts if overwrite is False but table exists!
    rows = []
    filepaths = remove_failed_scans(filepaths,
                                    verbose=args.verbose,
                                    log=args.log)

    for i, filepath in enumerate(filepaths):
        if filepath == f'{dirs["data_dir"]}/15398/GD153/F225W/ids0f0vpq_flt.fits':
            display_message(verbose=args.verbose,
                            log=args.log,
                            log_type='warning',
                            message='skipping over ids0f0vpq_flt.fits since it causes a critical error:')
            display_message(verbose=args.verbose,
                            log=args.log,
                            log_type='error',
                            message='\tv = data[:, j]\n\t\tIndexError: index 513 is out  of bounds for axis 1 with size 513')

        else:
            display_message(verbose=args.verbose,
                            log=args.log,
                            log_type='info',
                            message=f'***File {i+1}/{len(filepaths)}***: ')
            file_scan = obsScan(filepath)

            if args.run_cr_reject:
                file_scan.apply_crrej(args)

            if args.run_ap_phot:
                if args.ap_phot_fcr:
                    find_sources_data = file_scan.fcr_data
                else:
                    find_sources_data = file_scan.flt_data

                detected = file_scan.detect_sources(data=find_sources_data)

                if detected:
                    file_scan.ap_info = {'ap_dim': args.ap_dim,
                                         'sky_ap_dim': args.sky_ap_dim,
                                         'sky_thickness': args.sky_thickness,
                                         'sky_ap_area': calc_sky_ap_area(args.sky_ap_dim, args.sky_thickness),
                                         'ap_area': args.ap_dim[0]*args.ap_dim[1]}

                    use_scan, tbl = assess_scan_quality(find_sources_data,
                                                        file_scan.ap_info,
                                                        verbose, log,
                                                        plot=False)
                    if use_scan:
                        file_scan.calculate_phot(args,
                                                 flt_data=args.ap_phot_flt,
                                                 fcr_data=args.ap_phot_fcr)

                        scan_row = file_scan.make_scan_row(flt_data=args.ap_phot_flt,
                                                           fcr_data=args.ap_phot_fcr)
                        rows.append(scan_row)

            del file_scan

    if args.run_ap_phot:
        phot_table = Table(rows)

        if write:
            display_message(verbose=args.verbose,
                            log=args.log,
                            log_type='info',
                            message='Saving photometry table '\
                                    f'to {full_tbl_path}...')

            phot_table.write(full_tbl_path, format='csv', overwrite=overwrite)

            display_message(verbose=args.verbose,
                            log=args.log,
                            log_type='info',
                            message='Table successfully saved.')


class obsScan:
    """
    A class to represent a UVIS scan observation. Requires
    two attributes to initialize, and has four methods to
    enable reducing, analyzing, and compiling data.

    Attributes
    ----------
    filepath : str or path-like
        String representation of path to scan
        observation FITS file.
    args : `argparse.Namespace` or `InteractiveArgs`
        Arguments.

    Methods
    -------
    apply_crrej(args, output_dir=None, write_mask=True):
        Applies cosmic ray rejection and interpolation to
        input file.
    detect_sources(args, data):
        Constructs a table of information about any sources
        that can be detected in the input data array.
    calculate_phot(args, flt_data, fcr_data):
        Function to calculate sky-subtracted photometry for
        FLT data, FCR data, or both.
    make_scan_row(flt_data, fcr_data):
        Assembles a table row from previously-extracted
        header information and other essential attributes
        of the scan object.
    """
    def __init__(self, filepath, args):
        """
        Parameters
        ----------
        self : `obsScan` object
            Scan.
        filepath : str or path-like
            String representation of path to scan
            observation FITS file.
        args : `argparse.Namespace` or `InteractiveArgs`
            Arguments.
        """
        self.filepath = filepath
        self.file_type, self.data_ext = check_file(self.filepath,
                                                   verbose=args.verbose,
                                                   log=args.log)

        try:
            with fits.open(self.filepath) as f:
                self.hdr = get_header(self.data_ext, f)

                if self.file_type == 'flt':
                    self.flt_data = f[self.data_ext].data
                    self.flt_units = f[self.data_ext].header['BUNIT']
                else:
                    self.fcr_data = f[self.data_ext].data
                    self.fcr_units = f[self.data_ext].header['BUNIT']
            self.header_info = get_header_info(self,
                                               verbose=args.verbose,
                                               log=args.log)

        except OSError as oe:
            display_message(verbose=args.verbose,
                            log=args.log,
                            log_type='error',
                            message=f'{filepath} triggered OSError:\n{oe}')


    def apply_crrej(self, args, output_dir=None, write_mask=True):
        """
        Applies cosmic ray rejection and interpolation to
        input file. If files are to be created, also wraps
        the `wfc3_phot_tools` commands so generated output
        messages are directed appropriately:
            - if `args.log` is set to True, will log the
              messages;
            - if `args.verbose` is set to True, will print
              the messages to the output (if running from
              the command line, prints to terminal; if
              running interactively in a notebook, prints
              to notebook cell).

        Parameters
        ----------
        self : `obsScan` object
            Scan.
        args : `argparse.Namespace` or `InteractiveArgs`
            Arguments.
        output_dir : str or NoneType
            String representation of directory where output
            file(s) should be saved. If None, then files
            will be saved to directory where input file is
            located. Default is `None`.
        write_mask : Boolean
            Whether to write the CR rejection mask to a
            separate file. Default is `True`.
        """
        if output_dir == None:
            output_dir = self.filepath.split(os.path.basename(self.filepath))[0]

        fcr_filename = os.path.basename(self.filepath).replace('flt.fits', 'fcr.fits')
        self.fcr_filepath = os.path.join(output_dir, fcr_filename)

        if os.path.exists(self.fcr_filepath):  # if the FCR already exists
            if args.reprocess_fcr:             # and we said to reprocess
                make_fcr = True                # then we'll run CR rejection
            else:                              # and we said not to reprocess
                make_fcr = False               # then we'll just grab the existing data
        else:                                  # if the FCR doesn't exist
            if args.ap_phot_fcr:               # and we'll need to do photometry on FCR data
                make_fcr = True                # then we'll run CR rejection


        if make_fcr:
            with CaptureOutput() as outputs:
                cr_reject.make_crcorr_file_scan_wfc3(self.filepath, mult=4,
                                                     output_dir=output_dir,
                                                     ext=self.data_ext,
                                                     write_mask=write_mask)

            for output in outputs:
                display_message(verbose=args.verbose,
                                log=args.log,
                                log_type='info',
                                message=output)
        else:
            display_message(verbose=args.verbose,
                            log=args.log,
                            log_type='info',
                            message=f'Using FCR file found at: {self.fcr_filepath}')

        with fits.open(self.fcr_filepath) as f:
            self.fcr_hdr = f[0].header
            self.fcr_data = f[0].data
            self.fcr_units = self.fcr_hdr['BUNIT']

            if write_mask:
                self.mask_filepath = os.path.join(output_dir,
                                                  os.path.basename(self.filepath).\
                                                  replace('flt.fits', 'mask.fits'))
                with fits.open(self.mask_filepath) as m:
                    self.mask_data = m[0].data
            else:
                self.mask_filepath = None
                self.mask_data = None


    def detect_sources(self, args, data):
        """
        Helper function, wrapping the `detect_sources_scan`
        function from the `spatial_scan.phot_tools` module
        from the `wfc3_phot_tools` package. Attempts to
        create a source table, and returns error message
        if no sources are detected. If multiple sources
        are detected, it will use the first row of the
        table (which should correspond with the greatest
        source that `photutils` can detect).

        Parameters
        ----------
        self : `obsScan` object
            Scan.
        args : `argparse.Namespace` or `InteractiveArgs`
            Arguments.
        data : array-like
            Data array to use.

        Returns
        -------
        detected : Boolean
            Whether a source has been detected.
        """
        self.source_tbl = phot_tools.detect_sources_scan(data,
                                                         snr_threshold=3.0,
                                                         n_pixels=1000,
                                                         show=False)

        try:
            sources_detected = len(self.source_tbl)
            detected = True
        except TypeError:
            display_message(verbose=args.verbose,
                            log=args.log,
                            log_type='error',
                            message='No sources detected.')
            detected = False

        if detected:
            self.x_pos = self.source_tbl['xcentroid'][0]
            self.y_pos = self.source_tbl['ycentroid'][0]
            self.theta = -(90 - self.source_tbl['orientation'][0].value) * (np.pi / 180)

            if sources_detected == 1:
                message = f'Detected {sources_detected} source at '\
                          f'{self.x_pos}, {self.y_pos}.'
            else:
                message = f'Detected {sources_detected} sources at:'
                for i, row in enumerate(self.source_tbl):
                    message = f'{message}\n\t{self.source_tbl["xcentroid"][i]}, '\
                              f'{self.source_tbl["ycentroid"][i]}'
                    if i == 0:
                        message = f'{message} <--- using this one'
            display_message(verbose=args.verbose,
                            log=args.log,
                            log_type='info',
                            message=message)

        return detected


    def calculate_phot(self, args, flt_data, fcr_data):
        """
        Function to calculate the photometry. Performs
        photometry for the FLT data array, FCR data array,
        or both. Each data array is corrected for geometric
        distortion by multiplying the array by the
        requisite pixel area map (PAM). The background sky
        median count rate and its rms are calculated, as
        well as the total count rate and error within the
        photometric aperture. The median sky is then
        multiplied by the area of the photometric aperture,
        and this total sky is then subtracted from the
        photometric sum, thus producing sky-subtracted
        photometric count rate.

        Parameters
        ----------
        self : `obsScan` object
            Scan.
        args : `argparse.Namespace` or `InteractiveArgs`
            Arguments.
        flt_data : Boolean
            Whether to calculate photometry for FLT data.
        fcr_data : Boolean
            Whether to calculate photometry for FCR data.
        """
        self.back_method = args.back_method

        if flt_data:
            self.flt_data = uvis_pam.make_PAMcorr_image_UVIS(self.flt_data,
                                                             self.hdr,
                                                             self.hdr,
                                                             PAM_DIR)
            display_message(verbose=args.verbose,
                            log=args.log,
                            log_type='info',
                            message='Applied PAM correction to FLT data')
            self.flt_data = self.flt_data / self.header_info['exptime']                 # counts/second
            self.flt_units = f'{self.flt_units}/s'
            display_message(verbose=args.verbose,
                            log=args.log,
                            log_type='info',
                            message='Converted FLT data into count-rates')

            display_message(verbose=args.verbose,
                            log=args.log,
                            log_type='info',
                            message='Performing FLT photometry.....')
            self.flt_back, self.flt_back_rms = calc_sky_wrapper(self,
                                                                data_type='flt',
                                                                verbose=args.verbose,
                                                                log=args.log)
            self.flt_phot, self.flt_phot_rms = calc_phot_wrapper(self,
                                                                 data_type='flt',
                                                                 verbose=args.verbose,
                                                                 log=args.log,
                                                                 show=args.show_ap_plot)


        if fcr_data:
            try:
                self.fcr_data = uvis_pam.make_PAMcorr_image_UVIS(self.fcr_data,
                                                                 self.hdr,
                                                                 self.hdr,
                                                                 PAM_DIR)
                display_message(verbose=args.verbose,
                                log=args.log,
                                log_type='info',
                                message='Applied PAM correction to FCR data')
                self.fcr_data = self.fcr_data / self.header_info['exptime']              # counts/second
                self.fcr_units = f'{self.fcr_units}/s'
                display_message(verbose=args.verbose,
                                log=args.log,
                                log_type='info',
                                message='Converted FCR data into count-rates')

                display_message(verbose=args.verbose,
                                log=args.log,
                                log_type='info',
                                message='Converted FCR photometry.....')
                self.fcr_back, self.fcr_back_rms = calc_sky_wrapper(self,
                                                                    data_type='fcr',
                                                                    verbose=args.verbose,
                                                                    log=args.log)
                self.fcr_phot, self.fcr_phot_rms = calc_phot_wrapper(self,
                                                                     data_type='fcr',
                                                                     verbose=args.verbose,
                                                                     log=args.log,
                                                                     show=args.show_ap_plot)


            except AttributeError:
                display_message(verbose=args.verbose,
                                log=args.log,
                                log_type='critical',
                                message="Cannot calculate FCR photometry"\
                                        "because cosmic ray rejection has not "\
                                        "been performed yet.\nWhy don't you "\
                                        "try calling apply_crrej() and maybe "\
                                        "you'll calm down.")


    def make_scan_row(self, flt_data, fcr_data):
        """
        Helper function to assemble the table row from the
        existing header_info dictionary and various other
        essential attributes of the scan object. She's a
        little ugly but she works.

        Parameters
        ----------
        self : `obsScan` object
            Scan.
        flt_data : Boolean
            Whether to look for FLT photometry data.
        fcr_data : Boolean
            Whether to look for FCR photometry data.

        Returns
        -------
        row_dict : dict
            Dictionary where each key represents a column
            name, and the value is the corresponding value
            of the column for this particular scan.
        """
        colnames = list(self.header_info.keys())
        row_vals = list(self.header_info.values())

        self_dict = {prop: val for prop, val in vars(self).items()}

        phot_info_colnames = ['x_pos', 'y_pos', 'theta']
        phot_info_vals = [self_dict[key] for key in phot_info_colnames]
        colnames.extend(phot_info_colnames)
        row_vals.extend(phot_info_vals)

        ap_colnames = ['ap_dim', 'sky_ap_dim', 'sky_thickness', 'sky_ap_area', 'ap_area']
        ap_vals = [str(self.ap_info[key]) for key in ap_colnames]
        colnames.extend(ap_colnames)
        row_vals.extend(ap_vals)

        phot_colnames = ['back', 'back_rms', 'phot', 'phot_rms']

        if flt_data:
            flt_colnames = [f'flt_{col}' for col in phot_colnames]
            flt_vals = [self_dict[key] for key in flt_colnames]
            colnames.extend(flt_colnames)
            row_vals.extend(flt_vals)

        if fcr_data:
            fcr_colnames = [f'fcr_{col}' for col in phot_colnames]
            fcr_vals = [self_dict[key] for key in fcr_colnames]
            colnames.extend(fcr_colnames)
            row_vals.extend(fcr_vals)

        row_dict = {col: val for col, val in zip(colnames, row_vals)}

        return row_dict


def uvis_scan_pipeline(args, dirs):
    """
    Runs the UVIS scan calibration pipeline. There are two
    main components. The first is getting new data, and
    execution of this component of the module is determined
    by the Boolean flag `args.get_new_data`. The second
    component is reducing and/or analyzing data, governed
    primarily by the Boolean flags `args.run_cr_reject` and
    `args.run_ap_phot`. If none of these three attributes
    are set to `True`, then this function will not do
    anything.

    Parameters
    ----------
    args : `argparse.Namespace` or `InteractiveArgs`
        Arguments.
    dirs : dict
        Dictionary of three items - data, bad, and output
        directories.
    """
    if args.get_new_data:
        new_data = get_new_data(args, dirs)

    if args.run_cr_reject or args.run_ap_phot:
        run_process_wrapper(args, dirs)


def initialize_directories(args):
    """
    If run in trial mode (`args.trial` is True), then this
    creates the trial directory (named `args.name` in the
    UVIS scan monitor directory), then creates the three
    needed directories: `/data`, `/bad`, & `/output`, as
    well as the proposal, target, and filter sub-
    directories in `/data`. If `args.trial` is False,
    then the existence of the three directories is verified
    and the proposal, target, and filter sub-directories,
    if they do not already exist in `/data`, are created.

    Parameters
    ----------
    args : `argparse.Namespace` or `InteractiveArgs`
        Arguments.

    Returns
    -------
    dirs : dict
        Dictionary of directories.
    """
    if args.trial:
        trial_dir_name = args.name
        trial_dir = check_subdirectory(parent_dir=MONITOR_DIR,
                                       sub_name=trial_dir_name,
                                       verbose=args.verbose,
                                       log=args.log)

    else:
        trial_dir = MONITOR_DIR

    dir_names = ['data', 'bad', 'output']
    dirs = {}

    for dir_name in dir_names:
        dir = check_subdirectory(parent_dir=trial_dir,
                                 sub_name=dir_name,
                                 verbose=args.verbose,
                                 log=args.log)
        dirs[f'{dir_name}_dir'] = dir

        if dir_name == 'data':
            props = [str(x) for x in args.proposals]
            for prop in props:
                prop_dir = check_subdirectory(parent_dir=dir,
                                              sub_name=prop,
                                              verbose=args.verbose,
                                              log=args.log)

                for targ in args.targets:
                    targ_dir = check_subdirectory(parent_dir=prop_dir,
                                                  sub_name=targ,
                                                  verbose=args.verbose,
                                                  log=args.log)

                    for filt in args.filters:
                        filt_dir = check_subdirectory(parent_dir=targ_dir,
                                                      sub_name=filt,
                                                      verbose=args.verbose,
                                                      log=args.log)

    return dirs


if __name__ == '__main__':
    args = parse_args()
    dirs = initialize_directories(trial_dir_name, args)

    if args.log:
        setup_logging(log_dir=od.path.join(MONITOR_DIR, 'logs'),
                      log_name=args.name)

    if args.verbose:
        display_args(args)

    uvis_scan_pipeline(args, dirs)
