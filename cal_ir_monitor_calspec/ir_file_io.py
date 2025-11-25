# pylint: disable=E1101
"""
Functions to manage file I/O, including checking and creating directories,
filtering and moving files, and setting paths.

This is also where we call `batch_reprocess()` from the reprocessing module,
so when we are locating data, we are only using the FLTs.

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

from astropy.table import Table
from astroquery.mast import Observations

from ir_reprocess import batch_reprocess
from ir_toolbox import MONITOR_DIR, make_timestamp


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


def archive_calibrated_files(group, current_dirname):
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
        print(f'No existing calibrated files in {current_dirname}')
    else:
        print(f'Archiving {len(calib_files)} calibrated files in '\
                     f'{current_dirname}')

        archive_timestamp = make_timestamp()

        for current_path in calib_files:
            current_name = os.path.basename(current_path)
            new_name = f'archive_{archive_timestamp}_{current_name}'
            new_path = os.path.join(os.path.dirname(current_path), new_name)

            shutil.move(current_path, new_path)
            print(f'Calibrated file moved to {new_path}')


def check_for_raw(filepath):
    """
    not in use?

    Checks to see if the RAW file corresponding to the path
    to the FLT file (`filepath`) exists already in the same
    location as the input file. Will return the anticipated
    RAW filepath and a Boolean indicating whether or not
    the RAW file exists.

    Parameter
    ---------
    filepath : str
        String representation of the path to an FLT file.

    Returns
    -------
    raw_filepath : str
        String representation of the path to a RAW file,
        regardless of if it actually exists.
    raw_exists : Boolean
        Whether the corresponding RAW exists in the same
        location.
    """
    raw_filepath = filepath.replace('flt.fits', 'raw.fits')
    raw_exists = os.path.exists(raw_filepath)

    if raw_exists:
        print(f'Found RAW file at {raw_filepath}')

    else:
        print(f'No RAW file found with {filepath}')

    return raw_filepath, raw_exists


def check_subdirectory(parent_dir, sub_name):
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

    Returns
    -------
    sub_dir : str or None
        String representation of path to subdirectory.
        Returns `None` if parent directory does not exist.
    """
    sub_dir = os.path.join(parent_dir, sub_name)
    if os.path.exists(parent_dir):
        if os.path.exists(sub_dir):
            print(f'  Found existing directory at {sub_dir}')
        else:
            print(f'  Making new directory at {sub_dir}')
            os.mkdir(sub_dir)

    else:
        print(f'  Nonexistent parent directory: {parent_dir}\n'\
              f'  Cannot make new directory at {sub_dir}')

        sub_dir = None

    return sub_dir


def filter_file_type(obs_table, reproc_hel, reproc_lin):
    """Filters file type for a batch of products.

    Parameters
    ----------
    obs_table : `astropy.table.table.Table`
        Observation table returned by `query_criteria()`
        and filtered to each proposal, target, and filter.
    helium_corr : Boolean
        Whether to apply helium correction.

    Returns
    -------
    filtered_products : `astropy.table.table.Table`
        Table of data products to be downloaded.
    """
    filt = obs_table['filters'][0]

    if reproc_lin or (reproc_hel and (filt in ['F105W', 'F110W'])):
        file_types = ['ASN', 'RAW', 'SPT']
        print('Will download ASN, RAW, and SPT files for reprocessing.')

    else:
        file_types = ['FLT', 'IMA']
        print('Will download FLT and IMA files.')

    # Get all products in the proposal/target/filter table.
    all_prods = Observations.get_product_list(obs_table)

    prods = Observations.filter_products(all_prods, project='CALWF3',
                                         productSubGroupDescription=file_types)


    print('Product table filtered to')

    for file_type in file_types:
        len_prod = len(prods[prods["productSubGroupDescription"] == file_type])
        print(f'\t{len_prod} {file_type} files')

    return prods


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
    print(f'{"-"*34}\nInitializing needed directories...')

    if args.trial:
        trial_dir_name = args.name

        if args.local:
            trial_parent_dir = os.getcwd()
        else:
            trial_parent_dir = MONITOR_DIR

        trial_dir = check_subdirectory(trial_parent_dir, trial_dir_name)

    else:
        trial_dir = MONITOR_DIR

    dir_names = ['data', 'bad', 'output', 'plots']
    dirs = {}

    for dir_name in dir_names:
        actual_dir = check_subdirectory(trial_dir, dir_name)
        dirs[f'{dir_name}_dir'] = actual_dir

    return dirs


def find_flts(file_dir, lin_bool, he_bool, nlinfile):
    """
    find files
    """
    filt = file_dir.split('/')[-1]
    flt_files = sorted(glob(os.path.join(file_dir, '*flt.fits')))

    if lin_bool or (he_bool and filt in ['F105W', 'F110W']):
        # Make sure to remove any FLT files that already exist
        # otherwise calwf3 will crash.
        if len(flt_files) > 0:
            for flt_file in flt_files:
                os.remove(flt_file)
            print(f'Removed {len(flt_files)} existing FLT files')

        flt_files = batch_reprocess(file_dir, lin_bool, he_bool, nlinfile)

    return flt_files


def get_dirs_and_attrs(parent_dir, attr_arg, make_int=False):
    """
    get directories and check against attributes
    """
    cat_dirs = sorted(glob(os.path.join(parent_dir, '*')))
    category = [os.path.basename(cat_dir) for cat_dir in cat_dirs]
    if make_int:
        category = [int(c) for c in category]

    use_dirs = [cat_dirs[i] for i, cat in enumerate(category) if cat in attr_arg]

    return use_dirs


def map_subdirectories(parent_dir, args):
    """
    map subdirectories
    """
    mapped_dirs = []

    p_dirs = get_dirs_and_attrs(parent_dir, attr_arg=args.proposals,
                                make_int=True)

    for p_dir in p_dirs:
        t_dirs = get_dirs_and_attrs(p_dir, attr_arg=args.targets)

        for t_dir in t_dirs:
            f_dirs = get_dirs_and_attrs(t_dir, attr_arg=args.filters)

            mapped_dirs.extend(f_dirs)

    return mapped_dirs


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
    print(f'{"-"*27}\nLocating data to process...')

    batches = {}

    ptf_dirs = map_subdirectories(data_dir, args)

    for ptf_dir in ptf_dirs:
        ptf = ptf_dir.split('/')[-3:]
        files = find_flts(ptf_dir, args.linearity, args.helium, args.nlinfile)

        if len(files) > 0:
            key = f'{ptf[0]}/{ptf[1]}/{ptf[2]}'
            print(f' {key}: {len(files)} files found')

            batches[key] = files

        else:
            print('Did not find any matching files in '\
                  f'{ptf_dir}')

    return batches


def move_bad_files(filepaths_to_move):
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
        print(f'Bad file {split_filepath[-1]} has been moved to '\
              f'{os.path.dirname(bad_filepath)}')


def move_downloaded_files(manifest, intended_dir):
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

    Returns
    -------
    manifest : `astropy.table.table.Table`
    error_tbl : `astropy.table.table.Table`
        Table of rows from the manifest that posed an issue.
    """
    print('Moving downloaded files...')

    problem_indices, problem_rows = [], []

    for index, prod in enumerate(manifest):
        new_path = os.path.join(intended_dir,
                                os.path.basename(prod['Local Path']))
        current_path = os.path.join(os.getcwd(),
                                    prod['Local Path'].split('../')[-1])

        try:
            shutil.move(current_path, new_path)
            print(f'File moved to {new_path}')

        except FileNotFoundError:
            problem_indices.append(index)
            problem_rows.append(prod)

            print(f'FileNotFoundError:\n    {current_path}\n    {new_path}')

    if len(problem_indices) > 0:
        error_tbl = Table(rows=problem_rows, names=manifest.colnames)
        manifest = manifest.remove_rows(problem_indices)

    else:
        error_tbl = Table()

    return manifest, error_tbl


def set_tbl_path(filename, write_dir, overwrite):
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
    """
    if write_dir[-1] == '/':
        write_dir = write_dir[:-1]

    if filename[-4:] != '.csv':
        filename = f'{filename}.csv'

    if not os.path.exists(write_dir):
        print(f'Nonexistent path {write_dir}\n'\
              'Using current working directory instead.')
        write_dir = os.getcwd()

    tbl_path = os.path.join(write_dir, filename)
    overwrite = True

    if os.path.exists(tbl_path):
        if overwrite:
            print(f'Table exists at {tbl_path} and `overwrite` is set to True.')
        else:
            print(f'Table exists at {tbl_path} and `overwrite` is set to False.'\
                  '\nAborting pipeline run.\nTry again with valid arguments.')

    return tbl_path
