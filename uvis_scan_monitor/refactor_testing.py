# outdated

import os
import glob
import time
import shutil
import copy
import json
import numpy as np
from astropy.io import ascii, fits
from astropy.table import Table, vstack
from astropy.time import Time
from astroquery.mast import Observations

import cr_reject as cr_reject # the scripts!
import phot_tools as phot_tools
import UVIS_PAM as uvis_pam
import daophot_err as daophot_err

from pyql.database.ql_database_interface import session
from pyql.database.ql_database_interface import Master, UVIS_flt_0, Anomalies
from scan_quality import assess_scan_quality
from housekeeping import check_subdirectory

PAM_DIR = '/grp/hst/wfc3v/wfc3photom/data/pamfiles/'
monitor_dir = '/grp/hst/wfc3v/wfc3photom/data/uvis_scan_monitor'
trial_dir_name = '2023_03_06_test1'

def calc_phot_wrapper(scan_obj, data_type, show=False):
    """
    Wrapper for calculating the sky-subtracted photometry
    inside a photometric aperture, as well as the error.

    Parameters
    ----------
    scan_obj :
    data_type : string
        Either 'flt' or 'fcr', denoting which data
        extension to use (flt_data or fcr_data, both of
        which are in counts per second).
    show : Boolean
        Whether to show the plot of the scan or not.

    Returns
    -------
    flux : float
        Total flux inside the photometric aperture minus
        the product of the sky background (median or mean)
        flux and the number of pixels in the photometric
        aperture (photometric aperture area). In units of
        counts per second.
    flux_err : float
        Error in the calculation of the sky-subtracted
        photometric flux. In units of counts per second.
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

    print('Uncorrected flux in photometric aperture:\n'\
          f'\t{flux_uncorr} electrons/second\n'\
          f'\t{flux_uncorr*scan_obj.header_info["exptime"]} electrons')

    # convert source sum back to electrons for error calculation
    flux_uncorr_e = scan_obj.header_info['exptime'] * flux_uncorr

    # convert background measurements back to electrons for error calculations
    if data_type == 'flt':
        back_e = scan_obj.flt_back * scan_obj.header_info['exptime']
        back_rms_e = scan_obj.flt_back_rms * scan_obj.header_info['exptime']

    else:
        back_e = scan_obj.header_info['exptime'] * scan_obj.fcr_back
        back_rms_e = scan_obj.header_info['exptime'] * scan_obj.fcr_back_rms

    print(f'Background {scan_obj.back_method} level in sky aperture:\n'\
          f'\t{back_e / scan_obj.header_info["exptime"]} electrons/second\n'\
          f'\t{back_e} electrons')

    flux_err = daophot_err.compute_phot_err_daophot(flux=flux_uncorr_e,
                                                    back=back_e,
                                                    back_rms=back_rms_e,
                                                    phot_ap_area=scan_obj.ap_info['ap_area'],
                                                    sky_ap_area=scan_obj.ap_info['sky_ap_area'])

    # convert error to count rate
    flux_err = flux_err / scan_obj.header_info['exptime']

    print('Subtracting background for all pixels in sky aperture:\n'\
          f'\t{(back_e * scan_obj.ap_info["ap_area"]) / scan_obj.header_info["exptime"]} electrons/second\n'\
          f'\t{back_e * scan_obj.ap_info["ap_area"]} electrons')

    # subtract background median times photometric area,
    # then convert back to count rate
    flux_e = flux_uncorr_e - (back_e * scan_obj.ap_info['ap_area'])

    flux = flux_e / scan_obj.header_info['exptime']

    print('Total sky-subtracted flux in photometric aperture:\n'\
          f'\t{flux} electrons/second\n'\
          f'\t{flux_e} electrons')
#    print(f'Converting sky-subtracted count-rate from {flux_e} electrons into count rate')

#    print('Background-subtracted countrate in '\
#          f'{scan_obj.ap_info["ap_dim"]} is {flux}.')

    return flux, flux_err


def calc_sky_ap_area(sky_ap_dim, n_pix):
    """
    Helper function to calculate the area of the sky
    background aperture given the inner dimensions and the
    width of the frame/rind.

    Parameters
    ----------
    sky_ap_dim : tuple
        The measurement of the inner boundary of the sky
        aperture in format (total width, total height).
    n_pix : float or int
        Depth or width of the frame or rind for the sky
        background aperture.

    Returns
    -------
    sky_ap_area : float or int
        The total area of the sky background frame or rind
        in units of squared pixels.
    """
    inner_area = sky_ap_dim[0] * sky_ap_dim[1]
    outer_area = (sky_ap_dim[0] + (2*n_pix)) * (sky_ap_dim[1] + (2*n_pix))
    sky_ap_area = outer_area - inner_area

    return sky_ap_area


def calc_sky_wrapper(scan_obj, data_type):
    """
    Wrapper for calculating the background sky level per
    pixel.

    Parameters
    ----------
    scan_obj :
    data_type : string
        Either 'flt' or 'fcr', denoting which data
        extension to use (flt_data or fcr_data, both of
        which are in counts per second).

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

    back, back_rms = phot_tools.calc_sky(data=data_attr,
                                         x_pos=scan_obj.x_pos,
                                         y_pos=scan_obj.y_pos,
                                         source_mask_len=scan_obj.ap_info['sky_ap_dim'][1],
                                         source_mask_width=scan_obj.ap_info['sky_ap_dim'][0],
                                         n_pix=scan_obj.ap_info['n_pix'],
                                         method=scan_obj.back_method)
    return back, back_rms


def check_file(filepath):
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
            #print(f'For {filepath}:\nfile_type is {file_type}\ndata_ext is {data_ext}')
            return file_type, data_ext
        except KeyError as ke:
            raise Exception('File does not appear to be either an FLT '\
                            f'or FCR file: {os.path.basename(filepath)}')
    else:
        raise Exception(f'Specified filepath does not exist: {filepath}')


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
#        print(sci_hdr['NAXIS1'])
        hdr = sci_hdr + hdr
#        print(hdr['NAXIS1'])   # I think this worked!

        #with open('example_header.txt', 'w') as txtfile:
        #    txtfile.write(json.dumps(hdr))

    return hdr

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

def resolve_targnames(targname, simplify=True):
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
        possible names, for searching in MAST.
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
            print(f'Warning! Unable to resolve name for {ke}.')
            resolved = targname
    else:
        resolved_targnames = [k for k, v in targnames.items() if v == targname]
        if len(resolved_targnames) > 0:
            resolved = resolved_targnames
        else:
            resolved = targname

    return resolved


def get_header_info(scan_obj):
    """
    Helper function to extract needed header info into a dictionary
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
            header_info[keyword] = resolve_targnames(scan_obj.hdr[keyword.upper()])
        header_info[keyword] = scan_obj.hdr[keyword.upper()]
    header_info['expstart_decimalyear'] = get_decimalyear(header_info['expstart'])
    return header_info


"""
1. Download/obtain files (FLT)
2. Get header info
3. PAM correct
4. CR-reject
5. Detect sources
6. Perform photometry
"""


def remove_non_scans(prods_p_t_f, proposal_ql_roots):
    # remove any non-scan products
    prods_p_t_f['obsmode'] = ['scan'
                              if prod['productFilename'].split('_')[0][:-1]
                              in proposal_ql_roots
                              else 'stare'
                              for prod in prods_p_t_f]

    prods_p_t_f = prods_p_t_f[prods_p_t_f['obsmode'] == 'scan']

    return prods_p_t_f


def redownload_data(prods_p_t_f, dir_p_t_f, redownload_data_flag):
    """
    redownload_data_flag : Boolean
    """
    planned_filenames = prods_p_t_f['productFilename']

    # if you don't want to redownload existing files:
    if not redownload_data_flag:
        for planned_filename in planned_filenames:
            planned_path = os.path.join(dir_p_t_f,
                                        os.path.basename(planned_filename))
            if os.path.exists(planned_path):
                print(f'Found existing file at {planned_path}')
                prods_p_t_f = prods_p_t_f[prods_p_t_f['productFilename'] != planned_filename]

    number_removed = len(planned_filenames) - len(prods_p_t_f)
    if number_removed == 0:
        print(f'Commencing download of {len(prods_p_t_f)} files...')
        continue_download = True
    else:
        if len(prods_p_t_f) == 0:
            print('All files in download queue already exist.')
            continue_download = False
        else:
            print(f'Removed {number_removed} files. '\
                  f'Commencing download of {len(prods_p_t_f)} files...')
            continue_download = True

    return prods_p_t_f, continue_download


def retrieve_scan_data(data_dir, **params):
    """
    ex :
    target_name=resolve_targnames('GD153', simplify=False)

    Parameters
    ----------
    data_dir : str

    **params
        proposal_id : int, list of int
            Proposal IDs or list of proposal IDs desired.
        filters : str, list of str
            Filter name or list of filter names desired.
        target_name : str, list of str
            Target name or list of target names desired.
    """
    download_manifest = Table()

    print(f'Querying MAST for data matching specified parameters:\n{params}')
    obs = Observations.query_criteria(**params)
    print(f'Found {len(obs)} matching observations.')

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

            dir_p = check_subdirectory(data_dir, proposal)

            # resolve (simplify) target names:
            obs_p['resolved_target_name'] = [resolve_targnames(t) for t in obs_p['target_name']]

            # make list of tables by target for each proposal
            targets = sorted(list(set(obs_p['resolved_target_name'])))
            obs_p_ts = [obs_p[obs_p['resolved_target_name'] == t] for t in targets]

            for target, obs_p_t in zip(targets, obs_p_ts):
                dir_p_t = check_subdirectory(dir_p, target)

                # make list of tables by filter for each proposal/target
                filters = sorted(list(set(obs_p_t['filters'])))
                obs_p_t_fs = [obs_p_t[obs_p_t['filters'] == f] for f in filters]

                for filt, obs_p_t_f in zip(filters, obs_p_t_fs):
                    dir_p_t_f = check_subdirectory(dir_p_t, filt)

                    # get all products in the proposal/target/filter table
                    all_prods_p_t_f = Observations.get_product_list(obs_p_t_f)

                    # filter to only the FLT files
                    prods_p_t_f = Observations.filter_products(all_prods_p_t_f,
                                                               productSubGroupDescription='FLT')
                    print('Filtered to just FLTs')


                    # remove non scans
                    prods_p_t_f = remove_non_scans(prods_p_t_f, proposal_ql_roots)

                    # If the file already exists in desired location,
                    # remove from product list if redownload is set to False
                    prods_p_t_f, continue_download = redownload_data(prods_p_t_f,
                                                                     dir_p_t_f,
                                                                     flags['redownload_data'])

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

def quality_check_scan(data, sky_ap_dim, n_pix, plot=False):
    use_scan, tbl = assess_scan_quality(data, sky_ap_dim, n_pix, plot=plot)

def get_new_data(data_dir, flags, params):
    """
    Test docstring
    """
    if params['targets'] == 'all':
        if params['filters'] == 'all':
            download_manifest = retrieve_scan_data(data_dir,
                                                   proposal_id=params['proposals'])
        else:
            download_manifest = retrieve_scan_data(data_dir,
                                                   proposal_id=params['proposals'],
                                                   filters=filters)
    else:
        search_targets = []
        for targname in params['targets']:
            search_targets.extend(resolve_targnames(targname, simplify=False))

        if params['filters'] == 'all':
            download_manifest = retrieve_scan_data(data_dir,
                                                   proposal_id=params['proposals'],
                                                   target_name=search_targets)
        else:
            download_manifest = retrieve_scan_data(data_dir,
                                                   proposal_id=params['proposals'],
                                                   target_name=search_targets,
                                                   filters=params['filters'])

    return download_manifest

def run_process_wrapper(flags, params, dirs):
    """
    Wrapper to aid with file sorting/location
    """
    if flags['run_cr_reject'] and not flags['run_ap_phot']:
        process_name = 'cosmic ray rejection'
    if flags['run_cr_reject'] and flags['run_ap_phot']:
        process_name = 'cosmic ray rejection and photometry'
    if flags['run_ap_phot'] and not flags['run_cr_reject']:
        process_name = 'photometry'

    proposals = sorted([os.path.basename(x)
                        for x in glob.glob(f'{dirs["data_dir"]}/*')])

    for proposal in proposals:
        if int(proposal) in params['proposals']:
            print(f'Beginning {process_name} for Program {proposal}')

            targets = sorted([os.path.basename(x)
                              for x in glob.glob(f'{dirs["data_dir"]}/{proposal}/*')])

            for target in targets:
                if target in params['targets']:
                    print(f'Beginning {process_name} for {target}')

                    filters = sorted([os.path.basename(x)
                                      for x in glob.glob(f'{dirs["data_dir"]}/{proposal}/{target}/*')])

                    for filt in filters:
                        if filt in params['filters']:
                            print(f'Beginning {process_name} for {filt}')

                            if params['file_type'] == 'flt':   # if we want anything from FLT files, then we'll start with the FLT files
                                print('Starting with FLT files.')
                                filepaths = glob.glob(f'{dirs["data_dir"]}/{proposal}/{target}/{filt}/*flt.fits')
                            else: # otherwise, start with the FCR files
                                print('Starting with FCR files.')
                                filepaths = glob.glob(f'{dirs["data_dir"]}/{proposal}/{target}/{filt}/*fcr.fits')

                            if len(filepaths) > 0:
                                print('Will process the following files: ')
                                filepaths = np.roll(sorted(filepaths), -10)
                                for filepath in filepaths:
                                    print(f'\t{filepath.split("/")[-1]}')

                                run_process(filepaths=filepaths,
                                            flags=flags,
                                            params=params,
                                            dirs=dirs,
                                            phot_table_name=f'{proposal}_{target}_{filt}.csv',
                                            write_loc=f'{dirs["output_dir"]}',
                                            write=True,
                                            overwrite=True)


                            else:
                                print(f'Did not find any matching files in {dirs["data_dir"]}/{proposal}/{target}/{filt}')
                        else:
                            print(f"Skipping {proposal}/{target}/{filt} in data directory since it's not specified in the parameters.")
                else:
                    print(f"Skipping {proposal}/{target} in data directory since it's not specified in the parameters.")
            else:
                print(f"Skipping {proposal} in data directory since it's not specified in the parameters")


def set_tbl_path(phot_table_name, write_loc, write, overwrite):
    """
    Helper function to set up table location. Only run if
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
        print(f'Warning: Nonexistent path {write_loc}\n'\
              'Using current working directory instead.')
        write_loc = os.getcwd()

    full_tbl_path = f'{write_loc}/{phot_table_name}'

    if os.path.exists(full_tbl_path):
        if overwrite:
            print(f'Warning: Existing table at {full_tbl_path} and '\
                  '`overwrite` is set to True.')
        else:
            raise Exception(f'Existing table at {full_tbl_path} but '\
                            '`overwrite` is set to False.\n'\
                            'Aborting run. Please try again with '\
                            'compatible arguments.')

    return full_tbl_path

def remove_known_gs_fails(filepaths, ):
    """
    Remove known guidestar failures from a list of files.
    """

    ql_roots = [os.path.basename(f).split('_')[0][:-1] for f in filepaths]
    failures = check_ql_gsfail(ql_roots)
    filepaths = [f for f in filepaths
                 if os.path.basename(f).split('_')[0][:-1] not in failures]

    print(f'Removed {len(failures)} scans affected by guide star failures.')
    print(f'Beginning processing set of {len(filepaths)} files.')

    return filepaths


def run_process(filepaths,
                flags,
                params,
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
    flags : dict
    params : dict
    phot_table_name : str
        File name to save the resulting photometry table.
    write_loc : str
        Location for saving the photometry table. Defaults
        to current working directory.
    write : Bool
        Whether to save the photometry table. Defaults to
        `True`.
    overwrite : Bool

    Returns
    -------
    phot_table : `astropy.table.table.Table`

    """
#    print(f'beginning: {filepaths[0]}')
    if write:
        full_tbl_path = set_tbl_path(phot_table_name=phot_table_name,
                                     write_loc=write_loc,
                                     write=True,
                                     overwrite=overwrite)
    rows = []
    filepaths = remove_known_gs_fails(filepaths)
    for i, filepath in enumerate(filepaths):
        if filepath == f'{dirs["data_dir"]}/15398/GD153/F225W/ids0f0vpq_flt.fits':
            print('skipping over ids0f0vpq_flt.fits since it causes a critical error:')
            print('\tv = data[:, j]\n\t\tIndexError: index 513 is out  of bounds for axis 1 with size 513')

        else:
            print(f'***File {i+1}/{len(filepaths)}***: ')
            file_scan = Scan(filepath)

            if flags['run_cr_reject']:
                file_scan.apply_crrej(flags)

            if flags['run_ap_phot']:
                if params['ap_phot_fcr']:
                    find_sources_data = file_scan.fcr_data
                else:
                    find_sources_data = file_scan.flt_data

                detected = file_scan.detect_sources(data=find_sources_data)

                if detected:
                    n_pix = 30

                    use_scan, tbl = assess_scan_quality(find_sources_data,
                                                        params['sky_ap_dim'],
                                                        n_pix,
                                                        plot=False)
                    if use_scan:
                        file_scan.calculate_phot(params,
                                                 n_pix=n_pix,
                                                 flt_data=params['ap_phot_flt'],
                                                 fcr_data=params['ap_phot_fcr'])
                        scan_row = file_scan.make_scan_row(flt_data=params['ap_phot_flt'],
                                                           fcr_data=params['ap_phot_fcr'])
                        rows.append(scan_row)

            del file_scan

    if flags['run_ap_phot']:
        phot_table = Table(rows)

        if write:
            print(f'Saving photometry table to {full_tbl_path}...')
            phot_table.write(full_tbl_path, format='csv', overwrite=overwrite)
            print('Table successfully saved.')

        #return phot_table


class Scan:
    def __init__(self, filepath):
        self.filepath = filepath
        self.file_type, self.data_ext = check_file(self.filepath)

        try:
            with fits.open(self.filepath) as f:
                self.hdr = get_header(self.data_ext, f)

                if self.file_type == 'flt':
                    self.flt_data = f[self.data_ext].data
                    self.flt_units = f[self.data_ext].header['BUNIT']
                else:
                    self.fcr_data = f[self.data_ext].data
                    self.fcr_units = f[self.data_ext].header['BUNIT']
            self.header_info = get_header_info(self)
        except OSError as oe:
            print(f'{filepath} triggered OSError:\n{oe}')


    def apply_crrej(self, flags, output_dir=None, write_mask=True):
        """
        def make_crcorr_file_scan_wfc3(input_file, mult=4, output_dir=None, ext=1,
                                       write_mask=True):

            _write_fcr(input_file, output_dir, corrected_data, ext, file_type)

            if write_mask:
                _write_mask(input_file, output_dir, mask, ext, file_type)

        """
        if output_dir == None:
            output_dir = self.filepath.split(os.path.basename(self.filepath))[0]

        fcr_filename = os.path.basename(self.filepath).replace('flt.fits', 'fcr.fits')
        self.fcr_filepath = os.path.join(output_dir, fcr_filename)

        if os.path.exists(self.fcr_filepath):  # if the FCR already exists
            if flags['reprocess_fcr']:         # and we said to reprocess
                make_fcr = True                # then we'll run CR rejection
            else:                              # and we said not to reprocess
                make_fcr = False               # then we'll just grab the existing data
        else:                                  # if the FCR doesn't exist
            if params['ap_phot_fcr']:          # and we'll need to do photometry on FCR data
                make_fcr = True                # then we'll run CR rejection


        if make_fcr:
            cr_reject.make_crcorr_file_scan_wfc3(self.filepath, mult=4,
                                                 output_dir=output_dir,
                                                 ext=self.data_ext,
                                                 write_mask=write_mask)
        else:
            print(f'Using FCR file found at: {self.fcr_filepath}')

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

    def detect_sources(self, data):
        """
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
            print('No sources detected.')
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
            print(message)

        return detected


    def calculate_phot(self,
                       params, #ap_dim, sky_ap_dim,
                       n_pix, #back_method='median',
                       flt_data,
                       fcr_data):
        """
        Here we make sure to divide by exposure time first.
        """
        self.ap_info = {'ap_dim': params['ap_dim'],
                        'sky_ap_dim': params['sky_ap_dim'],
                        'n_pix': n_pix,
                        'sky_ap_area': calc_sky_ap_area(params['sky_ap_dim'], n_pix),
                        'ap_area': params['ap_dim'][0]*params['ap_dim'][1]}

        self.back_method = params['back_method']

        if flt_data:
            self.flt_data = uvis_pam.make_PAMcorr_image_UVIS(self.flt_data,
                                                             self.hdr,
                                                             self.hdr,
                                                             PAM_DIR)
            print('Applied PAM correction to FLT data')
            self.flt_data = self.flt_data / self.header_info['exptime']                 # counts/second
            self.flt_units = f'{self.flt_units}/s'
            print('Converted FLT data into count-rates')

            print('Performing FLT photometry.....')
            self.flt_back, self.flt_back_rms = calc_sky_wrapper(self, data_type='flt')
            self.flt_phot, self.flt_phot_rms = calc_phot_wrapper(self,
                                                                 data_type='flt',
                                                                 show=flags['show_ap_plot'])


        if fcr_data:
            try:
                self.fcr_data = uvis_pam.make_PAMcorr_image_UVIS(self.fcr_data,
                                                                 self.hdr,
                                                                 self.hdr,
                                                                 PAM_DIR)
                print('Applied PAM correction to FCR data')
                self.fcr_data = self.fcr_data / self.header_info['exptime']              # counts/second
                self.fcr_units = f'{self.fcr_units}/s'
                print('Converted FCR data into count-rates')

                print('Converted FCR photometry.....')
                self.fcr_back, self.fcr_back_rms = calc_sky_wrapper(self, data_type='fcr')
                self.fcr_phot, self.fcr_phot_rms = calc_phot_wrapper(self,
                                                                     data_type='fcr',
                                                                     show=flags['show_ap_plot'])


            except AttributeError:
                print("Cannot calculate FCR photometry because "\
                      "cosmic ray rejection has not been performed yet.\n"\
                      "Why don't you try calling apply_crrej() and "\
                      "maybe you'll calm down.")


    def make_scan_row(self, flt_data, fcr_data):
        """
        Helper function to assemble the table row from the
        existing header_info dictionary and various other
        essential attributes of the scan object. She's a
        little ugly but she works.

        Parameters
        ----------
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

        ap_colnames = ['ap_dim', 'sky_ap_dim', 'n_pix', 'sky_ap_area', 'ap_area']
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


def main(flags, params, dirs):
    """
    Parameters
    ----------
    flags : dict
        Dictionary of execution flags.
            flags = {'get_new_data': False,
                     'run_cr_reject': False,
                     'reprocess_fcr': False,
                     'remove_gs_fails': False,
                     'run_ap_phot': False,
                     'show_ap_plot': False}
    params : dict
        Dictionary of pipeline parameter specifications.
            params = {'proposals': [14878, 15398, 15583, 16021, 16416, 16580, 17016],
                      'targets': 'all',
                      'filters': 'all',
                      'file_type': 'flt',
                      'ap_dim': (44, 268),
                      'sky_ap_dim': (300, 400),
                      'back_method': 'median',
                      'ap_phot_flt': False,
                      'ap_phot_fcr': False}
    dirs : dict
        Dictionary of three items - data, bad, and output
        directories.
    """
    if flags['get_new_data']:
        new_data = get_new_data(dirs['data_dir'], flags, params)

    if flags['run_cr_reject'] or flags['run_ap_phot']:
        run_process_wrapper(flags, params, dirs)

def initialize_directories(trial_dir_name, params):
    trial_dir = check_subdirectory(monitor_dir, trial_dir_name)
    dir_names = ['data', 'bad', 'output']
    dirs = {}

    for dir_name in dir_names:
        dir = check_subdirectory(trial_dir, dir_name)
        dirs[f'{dir_name}_dir'] = dir

        if dir_name == 'data':
            props = [str(x) for x in params['proposals']]
            for prop in props:
                prop_dir = check_subdirectory(dir, prop)

                for targ in params['targets']:
                    targ_dir = check_subdirectory(prop_dir, targ)

                    for filt in params['filters']:
                        filt_dir = check_subdirectory(targ_dir, filt)



#def set_input():
#    flags = {'get_new_data': True,
#             'redownload_data': False,
#             'run_cr_reject': True,
#             'reprocess_fcr': False,
#             'run_ap_phot': True,
#             'show_ap_plot': False}

    params = {'proposals':[14878, 15398, 15583, 16021, 16416, 16580, 17016], #[15583],#'proposals': ,
              'targets': ['GRW70', 'GD153'],
              'filters': ['F218W', 'F225W', 'F275W', 'F336W', 'F438W', 'F606W', 'F814W'],#['F814W'],#'filters': ['F225W', 'F438W', 'F814W'], 15583/GRW70/F814W
              'file_type': 'flt',
              'ap_dim': (44, 268),
              'sky_ap_dim': (300, 400),
              'back_method': 'mean',
              'ap_phot_fcr': True,
              'ap_phot_flt': False}

    trial_dir = check_subdirectory(monitor_dir, trial_dir_name)

    dir_names = ['data', 'bad', 'output']
    dirs = {}

    for dir_name in dir_names:
        dir = check_subdirectory(trial_dir, dir_name)
        dirs[f'{dir_name}_dir'] = dir

        if dir_name == 'data':
            props = [str(x) for x in params['proposals']]
            for prop in props:
                prop_dir = check_subdirectory(dir, prop)

                for targ in params['targets']:
                    targ_dir = check_subdirectory(prop_dir, targ)

                    for filt in params['filters']:
                        filt_dir = check_subdirectory(targ_dir, filt)

    return flags, params, dirs



#def parse_cl_args():
#    """
#    In progress
#    """
#    argParser = argparse.ArgumentParser()
#    argParser.add_argument("-n", "--name", help="your name")

#    args = argParser.parse_args()

from test_cl import parse_args

if __name__ == '__main__':

#    flags, params, dirs = set_input()
    args = parse_args()

    #main(flags, params, dirs)
