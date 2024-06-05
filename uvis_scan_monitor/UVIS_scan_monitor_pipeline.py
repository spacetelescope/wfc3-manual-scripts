#!/usr/bin/env python

"""Runs WFC3 UVIS spatial scan monitoring pipeline.

This module enables downloading, inspection, and processing
of WFC3 UVIS spatial scan data and produces photometry catalogs.

Authors
-------
    Mariarosa Marinelli, 2021
    Debopam Som, 2020
    Clare Shanahan, 2018

Use
---
    This script can be executed directly via the command line, and
    will prompt user for input as needed.

        python UVIS_scan_monitor_pipeline.py

    This script can also be imported as a module, ex. in a Jupyter
    notebook.

        import UVIS_scan_monitor_pipeline as pipeline

"""
# imports
import os
import glob
import time
import shutil
import copy

import numpy as np
from astropy.io import ascii, fits
from astropy.table import Table, vstack
from astropy.time import Time

import wfc3_phot_tools.spatial_scan.cr_reject as cr_reject
import wfc3_phot_tools.spatial_scan.phot_tools as phot_tools
import wfc3_phot_tools.data_tools.sort_data as sort_data
import wfc3_phot_tools.data_tools.get_wfc3_data_astroquery as get_data_aq
import wfc3_phot_tools.utils.UVIS_PAM as uvis_pam
import wfc3_phot_tools.utils.daophot_err as daophot_err

from pyql.database.ql_database_interface import session
from pyql.database.ql_database_interface import Master
from pyql.database.ql_database_interface import UVIS_flt_0, UVIS_spt_0

import matplotlib
import matplotlib.pyplot as plt
plt.style.use('bmh')

import warnings
warnings.filterwarnings('ignore')

import UVIS_scan_monitor_utilities as scan_utils

# Define global variable paths.
DATA_DIR = '/grp/hst/wfc3v/WFC3Library/uvis_scan_monitor/'
PHOT_TABLE_DIR = '/grp/hst/wfc3v/WFC3Library/uvis_scan_monitor/output'
PAM_DIR = '/grp/hst/wfc3p/cshanahan/phot_group_work/pixel_area_maps/'

# Define global variables.
TS = scan_utils.get_timestamp()
LOG_FILE = DATA_DIR+TS+'_log.txt'
LOG_DIR = os.path.join(DATA_DIR, TS)


def _setup_log(LOG_FILE, LOG_DIR):
    """Sets up pipeline run log.

    Parameters
    ----------
    LOG_FILE : str
        String name of logging file.
    LOG_DIR : str
        String name of logging directory path.
    """
    if not os.path.exists(LOG_DIR):
        os.mkdir(LOG_DIR)
        print(f'Directory {LOG_DIR} created.')
    if not os.path.exists(LOG_FILE):
        log_title = scan_utils.make_log_header()
        scan_utils.add_to_log(LOG_FILE, log_title)
        print(f'Log file {LOG_FILE} created.')

def _setup_dirs():
    """ Creates output directories, if they don't exist.
    """

    subdirs = ['new', 'bad', 'data']
    for s in subdirs:
        if not os.path.isdir(os.path.join(DATA_DIR, s)):
            os.makedirs(os.path.join(DATA_DIR, s))
            print(f'Making {os.path.join(DATA_DIR, s)}')

def _get_existing_filenames(data_dir, fits_file_type):
    """
    Returns a list of rootnames of already retrieved
    file that are in the sorted directory files, the 'bad'
    data directory and the 'new' data directory.

    Parameters
    ----------
    data_dir : str
        String name of data directory path.
    fits_file_type : str
        Input file type to be processed.
        Set via `params` dictionary.

    Returns
    -------
    all_filenames : list
        List of all filenames.
    """

    new_data_dir = os.path.join(data_dir, 'new/')
    new_data_files = glob.glob(new_data_dir+'*{}.fits'.format(fits_file_type))
    new_data_filenames = [os.path.basename(f) for f in new_data_files]

    bad_data_dir = os.path.join(data_dir, 'bad/')
    bad_data_files = glob.glob(bad_data_dir+'*{}.fits'.format(fits_file_type))
    bad_data_filenames = [os.path.basename(f) for f in bad_data_files]

    existing_data_dir = os.path.join(data_dir, 'data/')
    existing_data_files = glob.glob(existing_data_dir + \
                         '*/*/*{}.fits'.format(fits_file_type))
    existing_data_filenames = [os.path.basename(f) for f in existing_data_files]

    all_filenames = bad_data_filenames + existing_data_filenames + new_data_filenames

    return all_filenames

def _retrieve_scan_data_astroquery(prop_id, fits_file_type, data_dir):

    """
    First queries pyql for spatial scan files matching
    proposal ID, and then finds and downloads the data from
    Astroquery using `query_by_data_id`.

    When querying through pyql, results can include jitter
    files (suffixed with '-j' rather than '-q' or '-s'),
    which will not be found with Astroquery.

    To capture the actual exposures corresponding to the
    jitter files, we construct and pass their '-q' and '-s'
    counterparts to Astroquery. For each instance of a
    jitter file, there will only be one exposure file
    found, either a '-q' or '-s' fits file.

    Therefore, if there are jitter files returned with the
    initial pyql query, the number of file names passed to
    `query_by_data_id` will be greater than the number of
    files returned in `query_results`; the difference
    should be twice the number of jitter files originally
    returned. The number of files found in the QL database
    should be exactly equal to the number of files returned
    in `query_results`.

    Parameters
    ----------
    prop_id : list
        List of proposal IDs from which to download data.
        Set via `params` dictionary.
    fits_file_type : str
        Input file type to be processed.
        Set via `params` dictionary.
        Can be 'flt', 'flc', 'raw'.
    data_dir : str
        Data directory.
    """

    print('Retrieving all data from proposal {}'.format(str(prop_id)))

    results = session.query(Master.rootname).join(UVIS_flt_0).join(UVIS_spt_0).\
              filter(UVIS_flt_0.proposid == prop_id).\
              filter(UVIS_spt_0.scan_typ != 'N').all()

    all_scan_rootnames = [item[0] for item in results]

    # Compare list against files already retrieved/sorted.
    existing_filenames = [os.path.basename(x)[0:9]
                          for x in _get_existing_filenames(DATA_DIR,
                                                           fits_file_type)]

    new_file_rootnames = list(set(all_scan_rootnames) - set(existing_filenames))
    print(f'Found {len(new_file_rootnames)} un-ingested files in QL database.')

    # QL may return jitter files (suffixed '-j') which will not be found
    # by Astroquery. We want to look for the counterparts that will either
    # have a 'q' or 's' suffix (solid-state recorder and retransmitted from
    # PACket processOR (PACOR), respectively).

    jitter = []
    for each in new_file_rootnames:
        if each[-1] == 'j':
            jitter.append(each)
        else:
            pass
    print(f'Number of jitter files: {len(jitter)}')

    if len(jitter) > 0:
        look_for_q = [x[0:8]+'q' for x in jitter]
        look_for_s = [x[0:8]+'s' for x in jitter]
        query_all = new_file_rootnames + look_for_q + look_for_s
    else:
        query_all = new_file_rootnames

    print(f'\nNumber of files in query_all: {len(query_all)}')

    query_results = get_data_aq.query_by_data_id(query_all,
                                                 file_type=fits_file_type)

    print(f'\nFound {len(query_results)} results in Astroquery. Downloading...')

    # Log summary and results.
    su = f"\nquery_results:\n--------------\n" \
         f"_retrieve_scan_data_astroquery(prop_id={prop_id},\n" \
         f"                               fits_file_type='{fits_file_type}',\n"\
         f"                               data_dir={data_dir})\n" \
         f"{len(query_results)} results found in Astroquery.\n"

    scan_utils.add_to_log(LOG_FILE, su)
    for query_result in query_results:
        scan_utils.add_to_log(LOG_FILE, query_result['productFilename'])

    # Download data.
    get_data_aq.download_products(query_results,
                                  output_dir=os.path.join(data_dir, 'new'))

def get_header_info(hdr, keywords=['rootname', 'proposid', 'date-obs',
                                   'expstart', 'exptime', 'ccdamp',
                                   'aperture', 'flashlvl', 'targname',
                                   'filter', 'photflam', 'PHTRATIO']):
    """Extracts information from the fits header.

    Parameters
    ----------
    hdr : `FITS header`
        Header for the fits file being processed.
    keywords : list
        List of column names for which to extract
        information.

    Returns
    -------
    header_info : list
        List of header information.
    """
    header_info = []
    for keyword in keywords:
        header_info.append(hdr[keyword])
    return header_info

def _wrapper_make_phot_table(input_files, show_ap_plot, data_ext, ap_dim,
                             sky_ap_dim, back_method):
    """
    Runs full photometry process (PAM image, source
    detection, etc...) on input_files, returns an astropy
    table with photometry info.

    Parameters
    ----------
    input_files : list
        List of files to be processed.
    show_ap_plot : bool
        Set via `flags` dictionary.
        If True, the image will be displayed with all of
        the identified sources marked.
    data_ext : int
        Determined in `main_process_scan_UVIS`.
        If the `crrej` execution flag is True, `data_ext`
        will be 0.
    ap_dim : tuple
        Aperture used for photometry of scan.
        Set via `params` dictionary.
    sky_ap_dim : tuple
        Aperture used for sky background subtraction.
        Set via `params` dictionary.
    back_method : str
        Method used to calculated background.
        Set via `params` dictionary.

    Returns
    -------
    phot_tab : `astropy.table.table.Table`
        Table of photometry data.
    """

    phot_tab_colnames = []
    all_phot_info = []

    for f in input_files:
        print('\nRunning photometry on {}'.format(f))
        hdu = fits.open(f)
        data = fits.open(f)[data_ext].data
        hdr = fits.open(f)[data_ext].header
        if data_ext != 0:
            pri_hdr = fits.open(f)[0].header
            hdr = hdr + pri_hdr

        phot_info_cols = ['rootname', 'proposid', 'date-obs', 'expstart',
                          'exptime', 'ccdamp', 'aperture', 'flashlvl',
                          'targname', 'filter', 'photflam', 'PHTRATIO']

        phot_info = get_header_info(hdr, keywords=phot_info_cols)

        targnames = {'GD153': ['GD153', 'GD-153'],
                     'GRW70': ['GRW+70D5824', 'GRW+70D']}
        for standard, possible in targnames.items():
            for each in possible:
                if phot_info[-3] == each:
                    phot_info[-3] = standard

        expstart_date = Time(str(hdr['expstart']), format='mjd').to_value('iso')
        phot_info += [expstart_date, TS]
        phot_info_cols += ['expstart_date', 'pipeline_ts']

        source_tbl = phot_tools.detect_sources_scan(copy.deepcopy(data),
                                                    nsigma=3.0,
                                                    n_pixels=1000,
                                                    show=False)

        x_pos = source_tbl['xcentroid'][0].value
        y_pos = source_tbl['ycentroid'][0].value
        theta = -(90 - source_tbl['orientation'][0].value) * (np.pi / 180)

        print('Detected {} sources at {}, {}.'.format(len(source_tbl),
                                                      x_pos, y_pos))

        print('Making PAM corrected image in memory.')
        data = uvis_pam.make_PAMcorr_image_UVIS(copy.deepcopy(data),
                                                hdr, hdr, PAM_DIR)

		# Divide by exposure time to get countrate.
        data = data / hdr['EXPTIME']

        back, back_rms = phot_tools.calc_sky(copy.deepcopy(data), x_pos, y_pos,
                                             sky_ap_dim[1], sky_ap_dim[0],
                                             50, method = back_method)

        sky_ap_area = sky_ap_dim[0] * sky_ap_dim[1]
        phot_info += [back, back_rms]
        phot_info_cols += ['back', 'back_rms']

        # Background-subtracted data -  units are now e-/s, sky-subtracted.
        data = data - back

        w, l = ap_dim
        phot_ap_area = w * l

        phot_table = phot_tools.aperture_photometry_scan(data, x_pos, y_pos,
                                                         ap_width=w,
                                                         ap_length=l,
                                                         theta=theta,
                                                         show=show_ap_plot,
                                                         plt_title=os.path.basename(f))

        ap_sum = phot_table['aperture_sum'][0]
        print(f'Background-subtracted countrate in {ap_dim} is {ap_sum}.')

        # Convert source sum to countrate for error calculation
        flux_err = daophot_err.compute_phot_err_daophot(ap_sum*hdr['EXPTIME'],
                                                        back*hdr['EXPTIME'],
                                                        back_rms*hdr['EXPTIME'],
                                                        phot_ap_area,
                                                        sky_ap_area)

        flux_err = flux_err / hdr['EXPTIME']
        phot_info += [ap_sum, flux_err]
        phot_info_cols += ['countrate_{}_{}'.format(w,l), \
                           'err_{}_{}'.format(w,l)]

        phot_sens = ap_sum * hdr['PHOTFLAM']
        phot_sens_err = flux_err * hdr['PHOTFLAM']

        phtratio = hdr['PHTRATIO']

        phot_info += [phot_sens, phot_sens_err, phtratio]
        phot_info_cols += ['phot_sens_{}_{}'.format(w,l), \
                           'phot_sens_err_{}_{}'.format(w,l), \
                           'PHTRATIO']

        phot_tab_colnames = phot_info_cols
        all_phot_info.append(phot_info)

    phot_tab = Table()

    for i, val in enumerate(phot_tab_colnames):
        phot_tab[val] = [item[i] for item in all_phot_info]

    return phot_tab

def main_process_scan_UVIS(flags, params):
    """
    Runs photometry pipeline.

    Parameters
    ----------
    flags : dict
        Execution flags, set by user.
    params : dict
        Pipeline run parameters, set by user.
    """

    # Unpack flags:
    get_new_data = flags['newdat']
    sort_new_data = flags['sortdat']
    run_cr_reject = flags['crrej']
    cr_reprocess = flags['crrepr']
    run_ap_phot = flags['run_apphot']
    show_ap_plot = flags['show_ap']

    # Unpack params
    proposals = params['proposals']
    targets = params['targets']
    filters = params['filters']
    file_type = params['file_type']
    ap_dim = params['ap_dim']
    sky_ap_dim = params['sky_ap_dim']
    back_method = params['back_method']
    ap_phot_file_type = params['ap_phot_file_type']

    _setup_dirs()

    scan_utils.add_dict_to_log('flags', flags, LOG_FILE)
    scan_utils.add_dict_to_log('params', params, LOG_FILE)

    if get_new_data:
        for id in proposals:
            _retrieve_scan_data_astroquery(id, file_type, DATA_DIR)

    if sort_new_data:
        sort_data.sort_data_targname_filt(os.path.join(DATA_DIR, 'new'),
                                          os.path.join(DATA_DIR, 'data'),
                                          file_type=file_type,
                                          targname_mappings={'GD153' :
                                                            ['GD153', 'GD-153'],
                                                            'GRW70' :
                                                            ['GRW+70D5824',
                                                            'GRW+70D']})

    if targets == 'all':
        objss=[os.path.basename(x) \
               for x in glob.glob(os.path.join(DATA_DIR, 'data/*'))]
    else:
        objss = targets if type(targets) in [list, tuple] \
                else [targets]
    if filters == 'all':
        filtss=list(set([os.path.basename(x) for x in \
                    glob.glob(os.path.join(DATA_DIR, 'data/*/*'))]))
    else:
        filtss = filters if type(filters) in [list, tuple] \
                 else [filters]

    for objj in objss:
        for filtt in filtss:
            dirr = os.path.join(DATA_DIR, f'data/{objj}/{filtt}/')

            if run_cr_reject:
                print(f'CR rejection {filtt}, {objj}')
                cr_input = glob.glob(dirr + f'*{file_type}.fits')

                if cr_reprocess is False:
                    existing_fcr = glob.glob(dirr + '*fcr.fits')
                    cr_input = [f for f in cr_input  \
                                if f.replace(file_type, 'fcr') not in \
                                existing_fcr]

                if len(cr_input) > 0:
                    print('{} new files to cr reject.'.format(len(cr_input)))
                    for f in cr_input:
                        print(f)
                        cr_reject.make_crcorr_file_scan_wfc3(f)
                else:
                    print('All files already CR rejected.')

            if run_ap_phot:

                if ap_phot_file_type == 'fcr':
                    data_ext = 0
                elif (ap_phot_file_type == 'flt') or \
                     (ap_phot_file_type == 'flc'):
                    data_ext = 1
                else:
                    ve_mess = 'ap_phot_file_type must be fcr, flc, or flt.'
                    return ValueError(ve_mess)

                print(f'\n\n *** Photometry {filtt}, {objj} *** \n\n')

                fcrs = glob.glob(dirr + f'/*{ap_phot_file_type}.fits')
                print(f'{len(fcrs)} files for obj = {objj} filter = {filtt}.')

                phot_table = _wrapper_make_phot_table(fcrs,
                                                      show_ap_plot=show_ap_plot,
                                                      data_ext=data_ext,
                                                      ap_dim=ap_dim,
                                                      sky_ap_dim=sky_ap_dim,
                                                      back_method=back_method)

                if not os.path.isdir(PHOT_TABLE_DIR):
                    os.makedirs(PHOT_TABLE_DIR)
                    print(f'Making {PHOT_TABLE_DIR}')

                output_path = os.path.join(PHOT_TABLE_DIR,
                                           '{}_{}_phot.dat'.format(objj, filtt))
                print(output_path)
                if os.path.isfile(output_path):
                    os.remove(output_path)
                print(f'Writing {output_path}\n\n')
                ascii.write(phot_table, output_path, format = 'csv')

_setup_log(LOG_FILE, LOG_DIR)

def make_timestamp():
    today = datetime.now()
    timestamp = today.strftime('%Y-%m-%d_%H-%M-S')

    return timestamp


def parse_args():
    """Parses `UVIS_scan_monitor_pipeline.py` command line arguments.

    TO DO
    -----
        finish adapting and then test implementation


    Parses command line arguments for the UVIS spatial scan
    monitor pipeline. In addition to the default `--help`
    flag, there are a total of # configurable arguments:
        # pipeline settings
        # pipeline flags
        # pipeline parameters

    Pipeline Settings
    -----------------
    name : str
        Name of the directory for this pipeline run in
        central store. Defaults to the timestamp returned
        by `make_timestamp()`.
    trial : Boolean
        Run in 'trial' mode. Defaults to False if pipeline
        is run from command line (opposite default when
        pipeline is run through an `InteractiveArgs`
        object).
    verbose : Boolean
        Whether to print the message; defaults to False.
    log : Boolean
        Whether to log the message; defaults to False.

    Pipeline Execution Flags
    ------------------------
    get_new_data : Boolean
        If set, will query MAST and download data. If set
        to `True` and `redownload` is set to `False`,
        will only download new data. Default is `False`.
    ...
    run_ap_phot : Boolean
        When set, indicates that aperture photometry should
        be calculated. Default is `False`.

    Pipeline Parameters
    -------------------
    proposals : str or list of str
        Program ID(s) to examine. If none are provided,
        this will default to the list of all IR staring
        mode calibration programs, `MONITOR_PROGRAMS`.
    targets : str or list of str
        Target(s) to select. If `download_new_data` is set
        to `True`, will download data for these targets;
        names will be resolved such that providing `GRW70`
        will also download data from MAST matching
        `GRW+70D` and `GRW+70D5824`. Otherwise, will only
        process (drizzle and/or perform photometry) data in
        the defined directory for given targets. If no
        targets are defined, all targets available will be
        processed.
    filters : str or list of str
        WFC3/IR filters to select. If `download_new_data`
        is set to `True`, will only download data in these
        filters. Otherwise, will only process (drizzle
        and/or perform photometry) data in the defined
        directory in these filters. If no filters are
        defined, all filters available will be processed.
    file_type : str
        File type with which to begin pipeline operations.
        Possible options are `flt` or `drz`; defaults to
        `flt`.
    radius : int
        Radius, in pixels, of the photometric aperture;
        defaults to 3.
    annulus: int
        Inner radius, in pixels, of the background annulus;
        default is 14.
    dannulus : int
        Width, in pixels, of the background annulus;
        default is 5.
    back_method : str
        Method to calculate the background from the sigma-
        clipped data. Options are `mean`, `median`, and
        `mode`; defaults to `median`. AKA `salgorithm`.

    Returns
    -------
    args : `argparse.Namespace`
        Namespace class object that has as attributes the
        20 configurable arguments.
    """
    parser = ArgumentParser(prog='ir_phot_pipeline',
                            description='WFC3/IR standard staring mode  '\
                                        'photometry monitor pipeline',
                            epilog = 'Author: Mariarosa Marinelli')

    # Settings:
    parser.add_argument("-n", "--name",
                        help="name for pipeline run/log; defaults to timestamp",
                        default=make_timestamp())
    parser.add_argument("--local",
                        help='when set, runs pipeline to download stuff locally',
                        action='store_true')
    #parser.add_argument("-v", "--verbose",
    #                    help="when set, prints statements to command line",
    #                    action="store_true")
    #parser.add_argument("-l", "--log",
    #                    help="when set, logs statements to log file",
    #                    action="store_true")

    # Execution Flags:
    parser.add_argument("--get_new_data",                                 # keep the same
                        help="when set, get new data",
                        action='store_true')

    parser.add_argument("--sort_new_data",                                   # new
                        help="when set, sort new data (??)",
                        action='store_true')
    parser.add_argument("--run_cr_reject",                                      # new
                        help="when set, run cosmic ray rejection",
                        action='store_true')
    parser.add_argument("cr_reprocess",                                     # new
                        help="when set, re-run cosmic ray rejection on existing files",
                        action='store_true')
    parser.add_argument("--run_ap_phot",                                  # keep the same
                        help="when set, run aperture photometry",
                        action='store_true')
    parser.add_argument("--show_ap_plot",                                  # keep the same
                        help="when set, show aperture photometry plots (??)",
                        action='store_true')

    # Pipeline Parameters:
    parser.add_argument("--proposals",
                        help="calibration proposal or list of proposals",
                        nargs="+",
                        type=int)
    parser.add_argument("--targets",
                        help="target or list of targets (default is 'all')",
                        nargs="+",
                        default="all")
    parser.add_argument("--filters",
                        help="filter or list of filters (default is 'all')",
                        nargs="+",
                        default='all')
    parser.add_argument("--file_type",                                          # changed fcr to drz
                        help="file type to begin with (flt or fcr)",
                        default="flt",
                        choices=["flt", "fcr"])
    parser.add_argument("--ap_dim",                                             # new
                        help="photometric aperture dimensions",
                        type=list,
                        default=[36, 240])
    parser.add_argument("--sky_ap_dim",                                            # new
                        help="sky background rind outer dimensions",
                        type=list,
                        default=[75, 350])
    parser.add_argument("--back_method",                                        # new
                        help="method to calculate background from sigma-clipped data",
                        default="median",
                        choices=["mean", "median", "mode"])

    args = parser.parse_args()

    if args.run_cr_reject:
        args.ap_phot_file_type = 'fcr'

    else:
        args.ap_phot_file_type = args.file_type

    return args


if __name__ == '__main__':

    flags = FLAGS_DEF
    params = PARAMS_DEF

    print('\nBoolean flags for executing this pipeline will appear\n', \
    'with the default value in parentheses.\n', \
    '  To keep default, hit ENTER.\n', \
    '  To change, hit any other key.\n')
    for each in flags:
        default = flags.get(each)
        inp = input(f'{each} ({default})')
        if inp != '':
            flags[each] = not default

    print('\nSpecify datasets, data-handling, and analysis parameters.\n', \
    '  Default values, when applicable, will be stated.\n', \
    '  To keep default values, hit ENTER.\n')

    for each in params:
        default = params.get(each)
        default_type = type(params[each])
        inp = input(f'{each} (currently set to {default}): ')
        if inp != '':
            if default_type == str:
                inp = str(inp)
            elif default_type == int:
                inp = int(inp)
            elif default_type == tuple:
                inp = tuple(map(int, inp.split(',')))
            else:
                inp = list(map(int, inp.split(',')))
            params[each] = inp

    if flags['crrej']:
        params['ap_phot_file_type'] = 'fcr'
    else:
        params['ap_phot_file_type'] = params['file_type']

    print('\nExecuting UVIS spatial scan monitor script...\n')
    main_process_scan_UVIS(flags, params)
