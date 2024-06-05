#!/usr/bin/env python

"""
Utilities to use in the UVIS spatial scan monitor pipeline.

Authors
-------
    Mariarosa Marinelli, 2021

Use
---
    This module is intended to be imported and used as part
    of the UVIS spatial scan monitor pipeline.

        import UVIS_scan_monitor_utilities as scan_utils

Notes
-----
    This script uses the user-set dictionary `params`,
    which is defined by the user programmatically after
    import of `UVIS_scan_monitor_pipeline.py`. Brief
    definitions for this dictionary's keys are provided
    below, but more detail is given in the Jupyter notebook
    `wfc3_uvis_scan_monitor.ipynb`.

    `params`
    ''''''''
        'prop_ids' : list of int
            List of proposal IDs.
        'proc_objs': list of str
            List of objects.
        'proc_filts': list of str
            List of filters.
        'file_type': str
            Fits file type.
        'ap_dim': tuple
            Aperture used for photometry of scan.
        'sky_ap_dim': tuple
            Aperture used for sky background subtraction.
        'back_method': str
            Method used to calculated background.

"""


import os
import time
from astropy.io import fits
from astropy.table import Table, vstack
from astropy.time import Time
from astropy.stats import sigma_clip

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.patches import Rectangle

import numpy as np

def get_timestamp(dir_mode=True):
    """ Gets the current time in GMT and returns formatted
    timestamp.

    Parameters
    ----------
    dir_mode : bool (default)
        If True, returns a timestamp as a string. If False,
        returns a tuple from the ISO string and the MJD
        float timestamp.

    Returns
    -------
    ts : str or tuple
        Timestamp either as a string or as a tuple of
        two items, one string and one float.

    """
    if dir_mode:
        ts = time.strftime("%Y-%m-%d_%H%M", time.gmtime())
    else:
        t_iso = Time(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))
        t_mjd = t_iso.to_value('mjd', subfmt='decimal')
        ts = (t_iso.value, float(t_mjd))
    return ts

def add_to_log(filename, add_text):
    """ Add a line to the log file. If log file does not
    exist, create it.

    Parameters
    ----------
    filename : str
        Name of logging file.
    add_text : str
        Text to add to logging file.
    """
    with open(filename, 'a+') as file:
        file.seek(0)
        fdata = file.read(100)
        if len(fdata) > 0:
            file.write('\n')
        file.write(add_text)

def add_dict_to_log(label, dictionary, filename):
    """Formats dictionary in order to add to the log file.

    Parameters
    ----------
    label : str
        Label for dictionary.
    dictionary : dict
        Dictionary to be added to logging file.
    filename : str
        Name of logging file.
    """
    underline = '-'*(len(label)+1)
    add_to_log(filename, f'\n{label}:\n{underline}')
    for key, value in dictionary.items():
        line = key+' : '+str(value)
        add_to_log(filename, line)

def make_log_header():
    """Formats header for new logging file.

    Returns
    -------
    title : str
        Formatted string title to be added to the top of a
        new logging file.
    """

    desc = 'UVIS Spatial Scan Contamination Monitor Pipeline'
    lines = '='*len(desc)
    title = f'{lines}\n{desc}\n{lines}\n'
    return title

def sad_window(sky_ap_dim):
    """
    Returns a tuple defining the bounds for the sky
    aperture dimensions relative to the full exposure
    dimensions, for plotting purposes.

    Parameters
    ----------
    sky_ap_dim : tuple
        Tuple for sky aperture dimensions, set in `params`.

    Returns
    -------
    (lb, rb, bb, tb) : tuple
        Tuple of left boundary, right boundary, bottom
        boundary, and top boundary for plotting the sky
        aperture.
    """
    sci_shape = (513, 512)

    lb = int(0.5*(sci_shape[0] - sky_ap_dim[0]))
    rb = int(0.5*(sci_shape[0] + sky_ap_dim[0]))

    bb = int(0.5*(sci_shape[1] + sky_ap_dim[1]))
    tb = int(0.5*(sci_shape[1] - sky_ap_dim[1]))

    return (lb, rb, bb, tb)

def plot_exposure(file, w, params, save_figs, save_dir):
    """
    Plots three views of the same spatial scan exposure.

    Parameters
    ----------
    file : str
        Symbolic string representation of file location.
    w : tuple
        Tuple of 4 integers, defined by `sad_window()`.
    params : dict
        Dictionary of user-set params.
    save_figs : bool
        Whether to save the figures plotted during visual
        inspection.
    save_dir : str
        Where to save figures, if save_figs is True.
    """

    img_data = fits.getdata(file, ext=1)

    fig, ax = plt.subplots(1,3,figsize=(10,4),
                                   gridspec_kw={'width_ratios': [3, 3, 1]})
     # First image - unnormalized
    img1 = ax[0].imshow(img_data, cmap='gray_r')
    ax[0].add_patch(Rectangle((w[0], w[3]),
                              params['sky_ap_dim'][0],
                              params['sky_ap_dim'][1],
                              edgecolor='#414487FF',
                              fill=False,
                              lw=3))
    ax[0].set_title('Full exposure, inverted color', fontsize=9)

    # Second image: normalized from 50% percentile to 99.5% percentile
    img2 = ax[1].imshow(img_data, cmap='gray',
                        norm=Normalize(vmin=int(np.percentile(img_data, 50)),
                                       vmax=int(np.percentile(img_data, 99.5))))
    ax[1].add_patch(Rectangle((w[0], w[3]), params['sky_ap_dim'][0],
                              params['sky_ap_dim'][1],
                              edgecolor='#414487FF',
                              fill=False,
                              lw=3))
    ax[1].set_title('Normalized - 50% to 99.5% percentile)', fontsize=9)

    # Set up sky aperture window:
    window = img_data[w[3]:w[2], w[0]:w[1]]

    # Draw third image and colorbar
    img3 = ax[2].imshow(window, cmap='viridis', norm=LogNorm())
    cb3 = plt.colorbar(img3)
    cb3.set_label('electrons', fontsize=9)
    cb3.ax.tick_params(labelsize=9)
    ax[2].grid(None)
    ax[2].set_yticks([])
    ax[2].set_xticks([])
    ax[2].set_title('LogNorm\nsky aperture', fontsize=9)
    # Final plotting
    plt.suptitle(file)
    if save_figs == True:
        plt.savefig(f'{save_dir}/{file[7:-5]}.png', dpi=300, transparent=False)
    plt.show()
    plt.close()

def inspect_exposures(list_of_files, list_to_move, params,
                      save_figs=False, save_dir=None):
    """
    Allows user to manually inspect a list of files.
    Appends flagged file names to empty list, defined
    before calling function.

    Parameters
    ----------
    list_of_files : list
        List of files to inspect.
    list_to_move : list
        List to append names of flagged files.
    params : dict
        Dictionary of user-set params.
    save_figs : bool
        Whether to save the figures plotted during visual
        inspection. Default is False.
    save_dir : str
        Where to save figures, if save_figs is True.
        Default is None.
    """

    if save_figs == True:
        try:
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)
        except TypeError as t:
            print(f'Default value for save_dir is None.\nTypeError: {t}')

    message = 'To keep exposure, press ENTER.' \
              '\nTo remove exposure, press any other key.\n'

    w = sad_window(params['sky_ap_dim'])
    total = len(list_of_files)

    for count, i in enumerate(list_of_files):
        with fits.open(i) as img:
            plot_exposure(i, w, params, save_figs, save_dir)
        move_flag = input(f'Exposure {count+1}/{total}:\n{message}')
        if move_flag == '':
            pass
        else:
            list_to_move.append(i)

def review_bad(bad_list, params):
    """
    Review flagged exposures before removal from the
    processing pipeline, in case any were flagged by
    mistake.

    Parameters
    ----------
    bad_list : list
        List of flagged files.
    params : dict
        Dictionary of user-set params.

    Returns
    -------
    bad_list : list
        List of flagged files, from which any files that
        were subsequently identified as good quality have
        been removed.
    """
    rem_from_bad = []

    inspect_exposures(bad_list, rem_from_bad, params=params)

    if len(rem_from_bad) > 0:
        bad_list = [x for x in bad_list if x not in rem_from_bad]

    return bad_list

def review_good(list_of_files, bad_list, params):
    """
    Review un-flagged exposures, so that any exposures that
    were accidentally not flagged but should not be used
    can be removed from the processing pipeline.

    Parameters
    ----------
    list_of_files : list
        Input list of files.
    bad_list : list
        List of flagged files.
    params : dict
        Dictionary of user-set params.

    Returns
    -------
    bad_list : list
        List of flagged files, appended with any additional
        files flagged during subsequent review.
    """
    rem_from_good = []
    good_list = [x for x in list_of_files if x not in bad_list]

    inspect_exposures(good_list, rem_from_good, params=params)

    print(len(rem_from_good))

    if len(rem_from_good) > 0:
        bad_list = bad_list + rem_from_good

    return bad_list

def move_bad_data(list_of_bad_files, data_dir, log_file):
    """
    Moves bad files out of 'new' directory so that they
    will not be processed in the next step of the pipeline.
    Adds a record to the log_file of which files were
    removed.

    Parameters
    ----------
    list_of_bad_files : list
        List of flagged files.
    data_dir : str
        String pointing to location of data directory.
    log_file : str
        String name of logging file.
    """
    print(f'Moving {len(list_of_bad_files)} bad files...')
    add_to_log(log_file, '\nrejected exposures:\n-------------------')
    loc = f'{data_dir}/bad'
    for i in list_of_bad_files:
        os.system(f'mv {i} {loc}')
        entry = i.strip(data_dir)[4:]
        add_to_log(log_file, entry)
    print('Done.')

def archive_data(list_of_files, log_file, log_dir):
    """
    Moves photometry output table to log directory
    to archive it.

    Parameters
    ----------
    list_of_files : list
        List of csv files that will be moved.
    log_file : str
        String name of logging file
    log_dir : str
        String name of logging directory path.
    """
    output_header = 'photometry catalogs:\n--------------------'
    add_to_log(log_file, output_header)

    log_output_dir = os.path.join(log_dir, 'output')
    if not os.path.exists(log_output_dir):
        os.mkdir(log_output_dir)

    for each in list_of_files:
        os.system(f'cp {each} {log_output_dir}')
        add_to_log(log_file, each)

    print(f'Copied {len(list_of_files)} photometry files to {log_dir}.')

def aggregate_data(list_of_files, log_file, archive=False, log_dir=None):
    """
    Aggregates data files created from the photometry
    pipeline into one Astropy table.

    Parameters
    ----------
    list_of_files : list
        List of csv files from which to extract data.
    log_file : str
        String name of logging file.
    archive : bool
        Whether to archive files into the log directory.
    log_dir : str
        String name of logging directory path.

    Returns
    -------
    table : `astropy.table.table.Table`
        Table of all data from list_of_files.
    """
    table = Table()
    for each in list_of_files:
        t = Table.read(each, format='csv')
        table = vstack([table, t])

    if archive == True:
        try:
            archive_data(list_of_files, log_file, log_dir)
        except Exception as e:
            print('Missing log directory.')

    return table
