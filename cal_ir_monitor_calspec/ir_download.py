# pylint: disable=E1101
"""
Functions to enable downloading of IR standard star
staring mode calibration data.


python cal_ir_monitor_calspec.py --name 2025-11-12_test_vblin_alldata --trial --get_new_data --proposals 12699 12702 13088 13089 13092 13094 13573 13575 13576 13579 13711 14021 14024 14384 14386 14544 14883 14992 14994 15113 15582 16030 16415 16579 17015 17361 17681 17961 --helium --linearity
Author
------
    Mariarosa Marinelli, 2023
"""

import os

from astropy.table import Table, vstack
from astroquery.mast import Observations

from ir_file_io import check_subdirectory, filter_file_type, move_downloaded_files
from ir_toolbox import resolve_targnames, SIMPLE_TARGS


def query_for_data(**params):
    """
    Essentially a wrapper function for the astroquery.mast
    Observations class. Also removes any blanks or grisms
    that may be returned due to search parameters.

    Returns
    -------
    obs_all : `astropy.table.table.Table`
        Table of all observations.
    """
    obs_all = Observations.query_criteria(instrument_name='WFC3/IR',
                                          provenance_name='CALWF3',
                                          **params)
    for filter_to_remove in ['G102', 'G141', 'BLANK']:
        obs_all = obs_all[obs_all['filters'] != filter_to_remove]

    print(f'Found {len(obs_all)} matching observations.')

    return obs_all


def make_subdirs_from_obs(obs_tbl, data_dir):
    """
    sorts observation table
    """
    obs_tbls = {}

    proposals = sorted(list(set(obs_tbl['proposal_id'])))
    obs_ps = [obs_tbl[obs_tbl['proposal_id'] == p] for p in proposals]

    for (p, proposal) in enumerate(proposals):
        dir_p = check_subdirectory(parent_dir=data_dir, sub_name=proposal)

        obs_ps[p]['resolved_target_name'] = [resolve_targnames(t, simplify=True)
                                             for t in obs_ps[p]['target_name']]

        # Use list of resolved targets to build second layer of directories.
        targets = sorted(list(set(obs_ps[p]['resolved_target_name'])))
        obs_p_ts = [obs_ps[p][obs_ps[p]['resolved_target_name'] == t]
                    for t in targets]

        for (t, target) in enumerate(targets):
            dir_p_t = check_subdirectory(parent_dir=dir_p, sub_name=target)

            # Use list of filters to build third layer of directories.
            filters = sorted(list(set(obs_p_ts[t]['filters'])))
            obs_p_t_fs = [obs_p_ts[t][obs_p_ts[t]['filters'] == f]
                          for f in filters]

            for (f, filt) in enumerate(filters):
                dir_p_t_f = check_subdirectory(parent_dir=dir_p_t,
                                               sub_name=filt)

                obs_tbls[dir_p_t_f] = obs_p_t_fs[f]

    return obs_tbls


def retrieve_data(args, dirs, **params):
    """Queries MAST and downloads data.

    This function queries MAST (with optional parameters)
    and downloads observations to the proper location.

    Parameters
    ----------
    args : `argparse.Namespace` or `InteractiveArgs`
        Arguments.
    dirs : dict
        Dictionary of directories.
    **params
        proposal_id : int, list of int
            Proposal IDs or list of proposal IDs desired.
        filters : str, list of str
            Filter name or list of filter names desired.
        target_name : str, list of str
            Target name or list of target names desired.

    Returns
    -------
    download_manifest : `astropy.table.table.Table`
    """
    print(f'\nQuerying MAST for data matching parameters: {params}')

    download_manifest = Table()

    obs_all = query_for_data(**params)

    if len(obs_all) > 0:
        obs_tbls = make_subdirs_from_obs(obs_all, dirs['data_dir'])

        for sub_dir, sub_obs in obs_tbls.items():
            sub_prods = filter_file_type(sub_obs, args.helium,
                                         args.linearity)
            sub_prods, continue_download = redownload_wrapper(sub_prods,
                                                              sub_dir,
                                                              args)

            if continue_download:
                manifest = Observations.download_products(sub_prods)
                manifest, _ = move_downloaded_files(manifest, sub_dir)

                download_manifest = vstack([download_manifest, manifest])

    else:
        download_manifest = Table()

    return download_manifest


def redownload_wrapper(prods_p_t_f, dir_p_t_f, args):
    """Removes existing files if not redownloading.

    Function to remove already-existing data products from
    MAST data products table if redownloading files is not
    desired. Uses the `args.redownload` flag.

    Parameters
    ----------
    prods_p_t_f :  `astropy.table.table.Table`
        Table of data products of a particular program,
        target, and filter subset.
    dir_p_t_f : str
        String representation of directory path for a
        particular program, target, and filter subset of
        data.
    args :

    Returns
    -------
    prods_p_t_f :  `astropy.table.table.Table`
        Table of data products of a particular program,
        target, and filter subset, with already-existing
        data products removed if `redownload_flag` is
        set to `False`.
    continue_download : Boolean
        Whether or not to continue with the data download.
        Set to `False` if there are no data products left
        (`prods_p_t_f` is empty).
    """
    planned_filenames = prods_p_t_f['productFilename']

    # If you don't want to redownload existing files.
    if not args.redownload:
        for planned_filename in planned_filenames:
            planned_path = os.path.join(dir_p_t_f,
                                        os.path.basename(planned_filename))
            if os.path.exists(planned_path):
                print(f'Found existing file: {planned_path}')
                prods_p_t_f = prods_p_t_f[prods_p_t_f['productFilename'] != planned_filename]

    number_removed = len(planned_filenames) - len(prods_p_t_f)

    if number_removed == 0:
        print(f'Downloading {len(prods_p_t_f)} files...')
        continue_download = True

    else:
        if len(prods_p_t_f) == 0:
            print('All files in download queue already exist.')
            continue_download = False

        else:
            print(f'Removed {number_removed} files.\n'\
                  f'Downloading {len(prods_p_t_f)} files...')
            continue_download = True

    return prods_p_t_f, continue_download



def get_new_data_wrapper(args, dirs):
    """Wraps data retrieval function.

    This function serves as a wrapper for `retrieve_data()`
    and parses the variable parameters that are passed
    ultimately to the MAST query.

    First, if 'all' targets are desired, all targets in the
    global variable `SIMPLE_TARGS` are expanded into all
    possible target names (using the `resolve_targnames()`
    function). Otherwise, if input target(s) are given as a
    list, the list is expanded to incude every possible
    name for each specified target. If the input target is
    not given as a list (i.e. as a string), then the single
    target is expanded into a list that includes all
    variations on the target name.

    Next, if 'all' filters are desired, data is retrieved
    that matches the list of proposals and expanded list of
    targets (`search_targets`). Otherwise, only data that
    matches the proposal list, expanded list of targets,
    and the filter/list of filters is retrieved.

    Parameters
    ----------
    args : `argparse.Namespace` or `InteractiveArgs`
        Arguments.
    dirs : dict
        Dictionary of directories.

    Returns
    -------
    download_manifest : `astropy.table.table.Table`
        Download manifest of new data.
    """
    # "All" targets is not all targets, since we also use GO data.
    # In reality, we only care about the 5 targets in `SIMPLE_TARGS`.
    if args.targets == 'all':
        search_targets = []
        for targ in SIMPLE_TARGS:
            resolved = resolve_targnames(targname=targ, simplify=False)
            search_targets.extend(resolved)

    else:
        # If we have a list of targets, resolve them one by one.
        if isinstance(args.targets, list):
            search_targets = []
            for targ in args.targets:
                resolved_targnames = resolve_targnames(targname=targ,
                                                       simplify=False)
                search_targets.extend(resolved_targnames)

        # Only have one target? Much simpler.
        else:
            search_targets = resolve_targnames(targname=args.targets,
                                               simplify=False)

    if args.filters == 'all':
        _ = retrieve_data(args, dirs, proposal_id=args.proposals,
                          target_name=search_targets)
    else:
        _ = retrieve_data(args, dirs, proposal_id=args.proposals,
                          target_name=search_targets, filters=args.filters)
