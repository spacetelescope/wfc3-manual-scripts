"""
Functions to enable downloading of IR
standard star staring mode calibration
data.

Author
------
    Mariarosa Marinelli, 2023
"""

import os

from astropy.table import Table, vstack
from astroquery.mast import Observations

from ir_file_io import check_subdirectory, filter_file_type, move_downloaded_files
from ir_logging import display_message
from ir_toolbox import resolve_targnames, SIMPLE_TARGS


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
    for message in ['', 'Querying MAST for data matching specified parameters:',
                    f'{params}']:
        display_message(verbose=args.verbose, log=args.log, log_type='info',
                        message=message)

    download_manifest = Table()
    obs_all = Observations.query_criteria(instrument_name='WFC3/IR',
                                          provenance_name='CALWF3',
                                          **params)
    for filter_to_remove in ['G102', 'G141', 'BLANK']:
        obs_all = obs_all[obs_all['filters'] != filter_to_remove]

    display_message(verbose=args.verbose, log=args.log, log_type='info',
                    message=f'Found {len(obs_all)} matching observations.')

    if len(obs_all) > 0:
        # Use list of proposals to build first layer of directories.
        proposals = sorted(list(set(obs_all['proposal_id'])))
        obs_ps = [obs_all[obs_all['proposal_id'] == p] for p in proposals]

        for proposal, obs_p in zip(proposals, obs_ps):
            dir_p = check_subdirectory(parent_dir=dirs['data_dir'],
                                       sub_name=proposal,
                                       verbose=args.verbose, log=args.log)

            resolved_col = [resolve_targnames(t, simplify=True,
                                              verbose=args.verbose, log=args.log)
                            for t in obs_p['target_name']]
            obs_p['resolved_target_name'] = resolved_col

            # Use list of resolved targets to build second layer of directories.
            targets = sorted(list(set(obs_p['resolved_target_name'])))
            obs_p_ts = [obs_p[obs_p['resolved_target_name'] == t]
                        for t in targets]

            for target, obs_p_t in zip(targets, obs_p_ts):
                dir_p_t = check_subdirectory(parent_dir=dir_p,
                                             sub_name=target,
                                             verbose=args.verbose,
                                             log=args.log)

                # Use list of filters to build third layer of directories.
                filters = sorted(list(set(obs_p_t['filters'])))
                obs_p_t_fs = [obs_p_t[obs_p_t['filters'] == f] for f in filters]

                for filt, obs_p_t_f in zip(filters, obs_p_t_fs):
                    dir_p_t_f = check_subdirectory(parent_dir=dir_p_t,
                                                   sub_name=filt,
                                                   verbose=args.verbose,
                                                   log=args.log)

                    prods_p_t_f = filter_file_type(obs_p_t_f, args.helium_corr,
                                                   args.verbose, args.log)

                    prods_p_t_f, continue_download = redownload_wrapper(prods_p_t_f,
                                                                        dir_p_t_f,
                                                                        args)

                    if continue_download:
                        manifest = Observations.download_products(prods_p_t_f)
                        manifest, _ = move_downloaded_files(manifest,
                                                            dir_p_t_f,
                                                            verbose=args.verbose,
                                                            log=args.log)

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
                display_message(verbose=args.verbose, log=args.log,
                                message=f'Found existing file: {planned_path}',
                                log_type='info')
                prods_p_t_f = prods_p_t_f[prods_p_t_f['productFilename'] != planned_filename]

    number_removed = len(planned_filenames) - len(prods_p_t_f)

    if number_removed == 0:
        display_message(verbose=args.verbose, log=args.log, log_type='info',
                        message=f'Downloading {len(prods_p_t_f)} files...')
        continue_download = True

    else:
        if len(prods_p_t_f) == 0:
            display_message(verbose=args.verbose, log=args.log, log_type='info',
                            message='All files in download queue already exist.')
            continue_download = False

        else:
            display_message(verbose=args.verbose, log=args.log, log_type='info',
                            message=f'Removed {number_removed} files. '\
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
            resolved = resolve_targnames(targname=targ, simplify=False,
                                         verbose=args.verbose, log=args.log)
            search_targets.extend(resolved)

    else:
        # If we have a list of targets, resolve them one by one.
        if isinstance(args.targets, list):
            search_targets = []
            for targ in args.targets:
                resolved_targnames = resolve_targnames(targname=targ,
                                                       simplify=False,
                                                       verbose=args.verbose,
                                                       log=args.log)
                search_targets.extend(resolved_targnames)

        # Only have one target? Much simpler.
        else:
            search_targets = resolve_targnames(targname=args.targets,
                                               simplify=False,
                                               verbose=args.verbose,
                                               log=args.log)

    if args.filters == 'all':
        _ = retrieve_data(args, dirs, proposal_id=args.proposals,
                          target_name=search_targets)
    else:
        _ = retrieve_data(args, dirs, proposal_id=args.proposals,
                          target_name=search_targets, filters=args.filters)
