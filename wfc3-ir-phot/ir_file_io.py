# pylint: disable=E1101
"""
Functions to manage file I/O, including checking and creating directories,
filtering and moving files, and setting paths.

Author
------
    Mariarosa Marinelli, 2023

Functions
---------
    check_for_raw()
    check_subdirectory()
        Creates subdirectory if it doesn't already exist.
    initialize_directories()
        Initalizes data and pipeline output directories.
    download_raws()
        Checks for the corresponding RAW files for an input
        list of paths to files (FLTs or DRZs). If a RAW doesn't
        already exist in the same location as the FLT/DRZ file,
        the file is downloaded from MAST and moved to the
        appropriate location.
    filter_file_type()
        Filters file type for a batch of products.
    get_raw_product()
        Queries MAST for all observations matching IPPPSS and
        narrows product list to only the RAW file matching the
        full IPPPSSOOT rootname. Returns the subset of the
        product list Astropy table (really just 1 row).
    locate_data()
        Locate data in the data directory to be processed.
    move_downloaded_files()
        Moves downloaded files from their default MAST location to the
        correct directory, according to the program, target, and filter
        of the observation.
    rename_file()
        Archives calibrated files.
    set_tbl_path()
        Helper function to set up table location. This is only run if
        `write` in the main pipeline is True.
"""

import os
import shutil
from glob import glob

from astropy.table import Table, vstack
from astroquery.mast import Observations

from ir_logging import display_message, MONITOR_DIR, make_timestamp


def rename_file(to_rename, file_type, append_string):
    """
    To replace archive_calibrated_files().

    Parameters
    ----------
    to_rename : str or list
        What should be renamed.
    """
    if not isinstance(to_rename, list):
        to_rename = [to_rename]

    for file in to_rename:
        new_name = file.replace(f'_{file_type.lower()}.fits',
                                f'_{append_string}_{file_type.lower()}.fits')
        os.rename(file, new_name)


def archive_calibrated_files(group, current_dirname, verbose, log):
    """
    Parameter
    ---------
    group : str
    current_dirname : str
        String representation of the directory path.
    """
    # Using *.* to make sure we're only grabbing files (.fits and .tra)
    # Use `group` key from `groups` to only get the stuff from a specific visit
    calib_files = [f for f in glob(os.path.join(current_dirname, f'{group}*.*'))
                   if not f.endswith('raw.fits')]

    if len(calib_files) == 0:
        display_message(log=log, verbose=verbose, log_type='info',
                        message=f'No existing calibrated files in {current_dirname}')
    else:
        display_message(log=log, verbose=verbose, log_type='info',
                        message=f'Archiving {len(calib_files)} calibrated files'\
                                f' in {current_dirname}')

        archive_timestamp = make_timestamp()
        #archive_filename = os.path.join(current_dirname,
        #                               f'archive_{archive_timestamp}')

        #if not os.path.exists(archive_dirname):
        #    os.mkdir(archive_dirname)
        #    display_message(log=log, verbose=verbose, log_type='info',
        #                    message='Created archive directory at '\
        #                            f'{archive_dirname}')

        for current_path in calib_files:
            current_name = os.path.basename(current_path)
            new_name = f'archive_{archive_timestamp}_{current_name}'
            new_path = os.path.join(os.path.dirname(current_path), new_name)

            shutil.move(current_path, new_path)
            display_message(verbose=verbose, log=log, log_type='info',
                            message=f'Calibrated file moved to {new_path}')


def check_for_raw(filepath, verbose, log):
    """
    Checks to see if the RAW file corresponding to the path
    to the FLT or DRZ file (`filepath`) exists already in
    the same location as the input file. Will return the
    anticipated RAW filepath and a Boolean indicating
    whether or not the RAW file exists.

    Parameter
    ---------
    filepath : str
        String representation of a file path to an FLT or
        DRZ file.

    Returns
    -------
    raw_filepath : str
        String representation of a file path to a RAW file,
        regardless of if it actually exists.
    raw_exists : Boolean
        Whether the corresponding RAW exists in the same
        location.
    """
    raw_filepath = filepath.replace('flt.fits', 'raw.fits')
    raw_exists = os.path.exists(raw_filepath)

    if raw_exists:
        display_message(log=log,
                        verbose=verbose,
                        log_type='info',
                        message=f'Found RAW file at {raw_filepath}')

    else:
        display_message(log=log,
                        verbose=verbose,
                        log_type='info',
                        message=f'No RAW file found with {filepath}')

    return raw_filepath, raw_exists


def check_subdirectory(parent_dir, sub_name, verbose=True, log=False):
    """Creates subdirectory if it doesn't already exist.

    Helper function to check if a subdirectory exists.
    If it doesn't exist, the subdirectory is created.

    Parameters
    ----------
    parent_dir : str or path-like
        String representation of path or path-like object
        of parent directory (can be either relative or
        absolute path).
    sub_name : str
        Name of subdirectory.
    verbose : Boolean
        Whether to print the message; defaults to True.
    log : Boolean
        Whether to log the message; defaults to False.

    Returns
    -------
    sub_dir : str or None
        String representation of path to subdirectory.
        Returns `None` if parent directory does not exist.
    """
    sub_dir = os.path.join(parent_dir, sub_name)
    if os.path.exists(parent_dir):
        if os.path.exists(sub_dir):
            display_message(verbose=verbose,
                            log=log,
                            log_type='info',
                            message=f'  Found existing directory at {sub_dir}')
        else:
            display_message(verbose=verbose,
                            log=log,
                            log_type='info',
                            message=f'  Making new directory at {sub_dir}')
            os.mkdir(sub_dir)

    else:
        critical_messages = [f'  Nonexistent parent directory: {parent_dir}',
                             f'  Cannot make new directory at {sub_dir}']
        for critical_message in critical_messages:
            display_message(verbose=verbose,
                            log=log,
                            log_type='critical',
                            message=critical_message)

        sub_dir = None

    return sub_dir



def download_raws(filepaths, verbose, log):
    """
    Checks for the corresponding RAW files for an input
    list of paths to files (FLTs or DRZs). If a RAW doesn't
    already exist in the same location as the FLT/DRZ file,
    the file is downloaded from MAST and moved to the
    appropriate location.

    Parameters
    ----------
    filepaths : list of str
        List of paths to FLT/DRZ files.
    verbose :
    log :

    Returns
    -------
    existing_raw_filepaths : list
    """
    needed_raw_prods = Table()
    intended_dir = os.path.dirname(filepaths[0])
    raw_filepaths = []

    for filepath in filepaths:
        raw_filepath, raw_exists = check_for_raw(filepath, verbose, log)
        raw_filepaths.append(raw_filepath)

        if not raw_exists:
            display_message(verbose=verbose, log=log, log_type='info',
                            message=f'File {raw_filepath} does not exist '\
                                    'and will be downloaded')

            rootname = raw_filepath.split('/')[-1].split('_')[0]
            raw_prod = get_raw_product(rootname)

            needed_raw_prods = vstack([needed_raw_prods, raw_prod])

    if len(needed_raw_prods) == 0:
        display_message(verbose=verbose, log=log, log_type='info',
                        message=f'All expected RAW files exist.')
    else:
        display_message(verbose=verbose, log=log, log_type='info',
                        message=f'Downloading {len(needed_raw_prods)} files...')
        manifest = Observations.download_products(needed_raw_prods)

        manifest, _ = move_downloaded_files(manifest,
                                            intended_dir,
                                            verbose=verbose,
                                            log=log)

    display_message(verbose=verbose, log=log, log_type='info',
                    message='Verifying that downloaded RAWs are in '\
                            'the correct location...')
    raw_missing = [not os.path.exists(r) for r in raw_filepaths]

    if any(raw_missing):
        messages = ["Was not able to download RAW file to correct location:"]

        missing_files = [raw_filepath for i, raw_filepath
                         in enumerate(raw_filepaths)
                         if raw_missing[i] is True]
        existing_raw_filepaths = [raw_filepath for raw_filepath in raw_filepaths
                                  if raw_filepath not in missing_files]

        for missing_file in missing_files:
            messages.append(f'\t{missing_file}')

        for message in messages:
            display_message(verbose=verbose, log=log, message=message,
                            log_type='warning')

    else:
        existing_raw_filepaths = raw_filepaths

    return existing_raw_filepaths


def filter_file_type(obs_table, helium_corr, verbose, log):
    """Filters file type for a batch of products.

    Parameters
    ----------
    obs_table : `astropy.table.table.Table`
        Observation table returned by `query_criteria()`
        and filtered to each proposal, target, and filter.
    helium_corr : Boolean
        Whether to apply helium correction.
    verbose :
    log :

    Returns
    -------
    filtered_products : `astropy.table.table.Table`
        Table of data products to be downloaded.
    """
    filt = obs_table['filters'][0]

    if (filt in ['F105W', 'F110W']) and helium_corr:
        #file_types = ['FLT', 'RAW']
        # For now, just download the RAWs so there's no ambiguity
        file_types = ['IMA', 'ASN', 'FLT', 'RAW']
        display_message(verbose=verbose, log=log, log_type='info',
                        message='Will download ASN, FLT, IMA, and RAW files '\
                                f'to enable helium correction in {filt}')

    else:
        file_types = ['FLT']

    # Get all products in the proposal/target/filter table.
    all_products = Observations.get_product_list(obs_table)

    filtered_products = Observations.filter_products(all_products,
                                                     project='CALWF3',
                                                     productSubGroupDescription=file_types)

    messages = ['Product table filtered to']

    for file_type in file_types:
        prod_count = len(filtered_products[\
                         filtered_products[\
                         "productSubGroupDescription"] == file_type])
        messages.append(f'\t{prod_count} {file_type} files')

    for message in messages:
        display_message(verbose=verbose,
                        log=log,
                        log_type='info',
                        message=message)

    return filtered_products


def get_raw_product(rootname):
    """
    Queries MAST for all observations matching IPPPSS and
    narrows product list to only the RAW file matching the
    full IPPPSSOOT rootname. Returns the subset of the
    product list Astropy table (really just 1 row).

    Parameter
    ---------
    rootname : str
        HST observation ID.

    Returns
    -------
    raw_prod : `Astropy.table.table.Table`
        Table with a single row/data product, the RAW file
        corresponding to the FLT or DRZ being examined.
    """
    # Get all observations from this visit
    visit_obs = Observations.query_criteria(instrument_name='WFC3/IR',
                                            provenance_name='CALWF3',
                                            obs_id=f'{rootname[:-3]}*')

    visit_prods = Observations.get_product_list(visit_obs)

    raw_prod = Observations.filter_products(visit_prods,
                                            productFilename=f'{rootname}_raw.fits')

    return raw_prod


def initialize_directories(args):
    """Initalizes data and pipeline output directories.

    If run in trial mode (`args.trial` is True), then this
    creates the trial directory (named `args.name` in the
    IR standard star photometry directory), then creates
    the three needed directories: `/data`, `/bad`, &
    `/output`, as well as the proposal, target, and filter
    sub-directories in `/data`. If `args.trial` is False,
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
    function_desc = 'Initializing needed directories...'
    dashes = '-'*len(function_desc)
    for message in [dashes, function_desc]:
        display_message(verbose=args.verbose,
                        log=args.log,
                        log_type='info',
                        message=message)

    if args.trial:
        trial_dir_name = args.name

        if args.local:
            trial_parent_dir = os.getcwd()
        else:
            trial_parent_dir = MONITOR_DIR

        trial_dir = check_subdirectory(parent_dir=trial_parent_dir,
                                       sub_name=trial_dir_name,
                                       verbose=args.verbose,
                                       log=args.log)

    else:
        trial_dir = MONITOR_DIR

    dir_names = ['data', 'bad', 'output', 'plots']
    dirs = {}

    for dir_name in dir_names:
        actual_dir = check_subdirectory(parent_dir=trial_dir,
                                        sub_name=dir_name,
                                        verbose=args.verbose,
                                        log=args.log)
        dirs[f'{dir_name}_dir'] = actual_dir

    return dirs


def locate_data(args, data_dir):
    """Locate data in the data directory to be processed.

    Parameters
    ----------
    args : `argparse.Namespace` or `InteractiveArgs`
        Arguments.
    data_dir : str
        If not in `trial` mode, then this will just be
        the monitor data directory. Otherwise, will be
        the subdirectory indicated by `args.name`.

    Returns
    -------
    filepaths_batches : dict
        Dictionary where keys are unique combinations in
        the format 'proposal/target/filter', corresponding
        to the directories in which data exists, and values
        are the lists of files in those subdirectories.
    """
    status = 'Locating data to process...'
    dashes = '-'*len(status)
    for message in [dashes, status]:
        display_message(verbose=args.verbose,
                        log=args.log,
                        log_type='info',
                        message=message)

    skipping = "Skipping _ because + = was not specified."
    filepaths_batches = {}

    proposals_dirs = sorted(glob(os.path.join(data_dir, '*')))
    proposals = [int(os.path.basename(proposal_dir))
                 for proposal_dir in proposals_dirs]

    for proposal, proposal_dir in zip(proposals, proposals_dirs):
        if proposal in args.proposals:
            targets_dirs = sorted(glob(os.path.join(proposal_dir, '*')))
            targets = [os.path.basename(target_dir)
                       for target_dir in targets_dirs]

            for target, target_dir in zip(targets, targets_dirs):
                if (target in args.targets) or (args.targets == 'all'):
                    filters_dirs = sorted(glob(os.path.join(target_dir, '*')))
                    filters = [os.path.basename(filter_dir)
                               for filter_dir in filters_dirs]

                    for filt, filter_dir in zip(filters, filters_dirs):
                        if (filt in args.filters) or (args.filters == 'all'):

                            if args.helium_corr and filt in ['F105W', 'F110W']:
                                search_for = os.path.join(filter_dir,
                                                          f'*raw.fits')
                            else:
                                search_for = os.path.join(filter_dir,
                                                          f'*{args.file_type}.fits')

                            # Avoid picking up any of the original MAST or
                            # no-ramp-fitting files.
                            filepaths = [path for path in sorted(glob(search_for))
                                         if path.split('/')[-1].split('.')[0][9:]
                                         not in ['_mast_flt', '_nrf_flt']]


                            if len(filepaths) > 0:
                                batch_key = f'{proposal}/{target}/{filt}'
                                filepaths_batches[batch_key] = filepaths

                                messages = [f'{len(filepaths)} files found '\
                                            f'for {batch_key}:']

                                messages.extend([f'{" "*4}- {f.split("/")[-1]}'
                                                 for f in filepaths])

                                for message in messages:
                                    display_message(verbose=args.verbose,
                                                    log=args.log,
                                                    log_type='info',
                                                    message=message)

                            else:
                                message = 'Did not find any matching files in '\
                                          f'{filter_dir}'
                                display_message(verbose=args.verbose,
                                                log=args.log,
                                                log_type='warning',
                                                message=message)
                        else:
                            skip_message = skipping.replace('_', filter_dir).\
                                                    replace('+', 'filter').\
                                                    replace('=', filt)
                            display_message(verbose=args.verbose,
                                            log=args.log,
                                            log_type='info',
                                            message=skip_message)
                else:
                    skip_message = skipping.replace('_', target_dir).\
                                            replace('+', 'target').\
                                            replace('=', target)
                    display_message(verbose=args.verbose,
                                    log=args.log,
                                    log_type='info',
                                    message=skip_message)
        else:
            skip_message = skipping.replace('_', proposal_dir).\
                                    replace('+', 'proposal').\
                                    replace('=', str(proposal))
            display_message(verbose=args.verbose,
                            log=args.log,
                            log_type='info',
                            message=skip_message)

    return filepaths_batches


def move_bad_files(filepaths_to_move, verbose, log):
    """
    Moves a bad file into the `bad` data directory.

    Parameters
    ----------
    filepaths_to_move : list
        List of string representations of file paths for
        observations that were not deemed appropriate for
        photometry.
    """
    for filepath in filepaths_to_move:
        split_filepath = filepath.split('/')
        components = [i for i, string in enumerate(split_filepath)
                      if string == 'data']
        index = components[-1]  # Ensure it's the last directory called 'data'
        split_filepath[index] = 'bad'

        # Put file directly into 'bad' folder.
        # Add one, since stop point is not inclusive.
        bad_filedir = '/'.join(split_filepath[:index+1])
        bad_filepath = os.path.join(bad_filedir, split_filepath[-1])

        shutil.move(filepath, bad_filepath)
        display_message(verbose=verbose, log=log, log_type='info',
                        message=f'Bad file {split_filepath[-1]} has been '\
                                f'moved to {os.path.dirname(bad_filepath)}')


def move_downloaded_files(manifest, intended_dir, verbose, log):
    """
    Moves downloaded files from their default MAST location to the
    correct directory, according to the program, target, and filter
    of the observation.

    Parameters
    ----------
    manifest : `astropy.table.table.Table`
        Table of downloaded files.
    intended_dir : str
        String representation of directory path for a
        particular program, target, and filter subset of
        data.
    verbose
    log

    Returns
    -------
    manifest : `astropy.table.table.Table`
    error_tbl : `astropy.table.table.Table`
        Table of rows from the manifest that posed an issue.
    """
    display_message(verbose=verbose,
                    log=log,
                    log_type='info',
                    message='Moving downloaded files...')

    problem_indices, problem_rows = [], []

    for index, prod in enumerate(manifest):
        new_path = os.path.join(intended_dir,
                                os.path.basename(prod['Local Path']))
        current_path = os.path.join(os.getcwd(),
                                    prod['Local Path'].split('../')[-1])

        try:
            shutil.move(current_path, new_path)
            display_message(verbose=verbose,
                            log=log,
                            log_type='info',
                            message=f'File moved to {new_path}')

        except FileNotFoundError:
            problem_indices.append(index)
            problem_rows.append(prod)

            for message in ['FileNotFoundError:',
                            f'{" "*4} {current_path}',
                            f'{" "*4} {new_path}']:
                display_message(verbose=verbose,
                                log=log,
                                log_type='error',
                                message=message)

    if len(problem_indices) > 0:
        error_tbl = Table(rows=problem_rows, names=manifest.colnames)
        manifest = manifest.remove_rows(problem_indices)

    else:
        error_tbl = Table()

    return manifest, error_tbl


def set_tbl_path(filename, write_dir, overwrite, verbose, log):
    """
    Helper function to set up table location. This is only run if
    `write` in the main pipeline is True.

    Parameters
    ----------
    filename: str
        Name for file to be saved. Should either not have
        no extension or extension '.csv'.
    write : Boolean
        Whether to write (save) the table.
    write_dir : str
        The parent dictionary in which to write the table.
    overwrite : Boolean
        Whether to overwrite existing table.
    verbose : Boolean
        Whether to print the message.
    log : Boolean
        Whether to log the message.
    """
    if write_dir[-1] == '/':
        write_dir = write_dir[:-1]

    if filename[-4:] != '.csv':
        filename = f'{filename}.csv'

    if not os.path.exists(write_dir):
        warning_messages = [f'Nonexistent path {write_dir}',
                            'Using current working directory instead.']
        for warning_message in warning_messages:
            display_message(verbose=verbose,
                            log=log,
                            log_type='warning',
                            message=warning_message)
        write_dir = os.getcwd()

    tbl_path = os.path.join(write_dir, filename)
    overwrite = True

    if os.path.exists(tbl_path):
        if overwrite:
            overwrite_warning = f'Existing table at {tbl_path} and '\
                                '`overwrite` is set to True.'
            display_message(verbose=verbose,
                            log=log,
                            log_type='warning',
                            message=overwrite_warning)
        else:
            critical_messages = [f'Existing table at {tbl_path} but '\
                                 '`overwrite` is set to False.',
                                 'Aborting pipeline run. Please try again '\
                                 'with compatible arguments.']
            for critical_message in critical_messages:
                display_message(verbose=verbose,
                                log=log,
                                log_type='critical',
                                message=critical_message)

    return tbl_path
