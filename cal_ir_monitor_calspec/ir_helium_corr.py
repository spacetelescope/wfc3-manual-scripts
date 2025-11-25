# pylint: disable=E1101
"""
Enables helium correction in the F105W and F110W filters
for the IR staring mode standard star pipeline.


Functions
---------
_get_ipppss(rootname)
group_rootnames_by_ipppss(rootnames)
setup_calwf3_environs(verbose, log)
run_bestrefs(raw_filepaths, verbose, log)
run_calwf3(filepaths, verbose, log)
remove_ima_median_bg(ima_filepath, verbose, log)
rerun_calwf3(ima_filepaths, verbose, log)
"""
import os
from itertools import groupby
import numpy as np

import wfc3tools
from astropy.io import fits
import matplotlib.pyplot as plt

#from ir_file_io import move_downloaded_files
from ir_toolbox import display_message
from ir_logging import CaptureOutput
import time

def perform_helium_correction(files, verbose, log):
    print('--------FILEPATHS--------')
    for file in files:
        print(f'\f{file}')

    rootnames = [f.split('/')[-1].split('_')[0] for f in files]
    groups = group_rootnames_by_ipppss(rootnames)

    if verbose:
        print(f"Number of visits in ObsBatch: {len(groups)}")

    for i, group in enumerate(groups):
        filepaths = [f for f in files
                     if f.split('/')[-1].split('_')[0] in groups[group]]

        if verbose:
            print(f"Visit group {i+1}/{len(groups)}: {len(groups[group])} "\
                  f"rootnames and {len(filepaths)} files...")

        run_bestrefs(filepaths, verbose, log)

        for file_type in ['ima', 'flt']:
            files = [f.replace('_raw', f'_{file_type}') for f in filepaths]

            for file in files:
                new_name = file.replace(f'_{file_type}.', f'_mast_{file_type}.')
                os.rename(file, new_name)

                if verbose:
                    print(f'Renamed {file} to {new_name}')

        if verbose:
            print(f'Running calwf3 on {len(filepaths)} RAWs...')

        run_calwf3_nrf(filepaths, verbose, log)


def _get_ipppss(rootname):
    """Helper function to isolate IPPPSS in rootname.
    """
    return rootname[:-3]


def group_rootnames_by_ipppss(rootnames):
    """
    Groups a list of rootnames by the first six characters,
    representing the IPPPSS component of the full HST ID.
    Returns a dictionary wherein each item is a key-value
    pair. Every key is a unique IPPPSS identifier and the
    corresponding value is a list of rootnames that match
    the IPPPSS.

    Parameter
    ---------
    rootnames : list of str
        List of rootnames to group.

    Returns
    -------
    groups : dict
        Each item in this dictionary corresponds to the
        instrument/program/visit component of an IPPPSSOOT
        and a list of all rootnames that match those first
        six characters.
    """
    groups = {ipppss: list(match)
              for ipppss, match in groupby(rootnames, _get_ipppss)}

    return groups


def setup_calwf3_environs(verbose, log):
    """
    Parameters
    ----------
    verbose : Boolean
        Whether to print the message.
    log : Boolean
        Whether to log the message.
    """
    display_message(verbose=verbose, log=log, log_type='info',
                    message='Checking calwf3 configuration...')

    environment_variables = {'CRDS_SERVER_URL': 'https://hst-crds.stsci.edu',
                             'CRDS_SERVER': 'https://hst-crds.stsci.edu',
                             'CRDS_PATH': './crds_cache',
                             'iref': './crds_cache/references/hst/wfc3/'}

    for env_key, env_value in environment_variables.items():
        os.environ[env_key] = env_value
        display_message(verbose=verbose, log=log, log_type='info',
                        message=f"os.environ[{env_key}] has been set to "\
                                f"{os.environ[env_key]}")

    with CaptureOutput() as outputs:
        os.system('crds list --status')

    for output in outputs:
        display_message(verbose=verbose, log=log, log_type='info',
                        message=f'\t{output}')



def run_bestrefs(raw_filepaths, verbose, log):
    """
    Parameters
    ----------
    raw_filepaths : list of str
        List of strings representing the full filepaths to
        the newly-downloaded RAW files.
    verbose : Boolean
        Whether to print the message.
    log : Boolean
        Whether to log the message.
    """
    display_message(verbose=verbose, log=log, log_type='info',
                    message='Updating RAW files with bestrefs')

#    with CaptureOutput() as outputs:
    for raw_filepath in raw_filepaths:

            # #RuntimeError: calwf3.e exited with code ERROR_RETURN:
#            os.system(f"crds bestrefs --files {raw_filepath}
#                     --sync_references=1 --update_bestrefs")
        os.system(f"crds bestrefs --files {raw_filepath} -s 1 --update-bestrefs")

#    for output in outputs:
#        display_message(verbose=verbose, log=log, log_type='info',
#                        message=f'\t{output}')

def run_calwf3_nrf(raw_files, verbose, log):
    """
    Parameters
    ----------
    raw_files : list
        List of strings representing the full filepaths to
        the newly-downloaded RAW files.
    """
#    raw_files = glob('my_files/*raw.fits')

    for raw_file in raw_files:
        raw = fits.open(raw_file, mode='update')
        raw[0].header['CRCORR'] ='OMIT'
        raw.flush()

    for raw_file in raw_files:
        #wfc3tools.calwf3(raw_file)
        wfc3tools.wf3ir(raw_file, verbose=True);#, save_tmp=True);

        time.sleep(5)

    for raw_file in raw_files:
        flt_file = raw_file.replace('_raw', '_flt')

        new_name = flt_file.replace('_flt.fits', '_nrf_flt.fits')
        os.rename(flt_file, new_name)
        if verbose:
            print(f'renamed {os.path.basename(flt_file)} to {new_name}')


def rename_flts(raw_files, verbose, log):
    for raw_file in raw_files:
        flt_file = raw_file.replace('_raw', '_flt')

        new_name = flt_file.replace('_flt.fits', '_nrf_flt.fits')
        os.rename(flt_file, new_name)

        display_message(verbose=verbose, log=log, log_type='info',
                        message=f'Renamed {os.path.dirname(flt_file)} to '\
                                f'{new_name}')

# def run_calwf3_omit_crcorr(group, raw_filepaths, verbose, log):
#     """
#     Parameters
#     ----------
#     group : str
#         String representing the "ipppss" group.
#     raw_filepaths : list of str
#         List of strings representing the full filepaths to
#         the RAW files.
#     verbose : Boolean
#         Whether to print the message.
#     log : Boolean
#         Whether to log the message.
#     """
#     display_message(verbose=verbose, log=log, log_type='info',
#                     message=f'Now attempting to run calwf3 for the first time...')
#     for raw_filepath in raw_filepaths:
#         display_message(verbose=verbose, log=log, log_type='info',
#                         message=f'Working on RAW: {raw_filepath}')
#         # This is the thing that's failing:
#         wfc3tools.calwf3(raw_filepath, debug=True, verbose=True, save_tmp=True)
#
#         time.sleep(5) # add a time delay to see if that's why IMAs aren't being written
#
#
# def change_crcorr_val(filepaths, verbose, log):
#
#     for filepath in filepaths:
#         #with CaptureOutput() as outputs:
#         with fits.open(filepath, mode='update') as raw:
#             raw[0].header['CRCORR'] = 'OMIT'
#             display_message(verbose=verbose, log=log, log_type='info',
#                             message=f'Header updated for {os.path.basename(filepath)}')


def remove_ima_median_bg(raw_filepaths, verbose, log, plot=False,
                         plot_settings={'show': False,
                                        'save': False,
                                        'save_dir': os.getcwd()}):
    """
    Adapted from TVB_flattenramp_notebook.ipynb

    Parameters
    ----------
    filepath : str
        String representation of the full path to the RAW
        file.
    verbose : Boolean
        Whether to print the message.
    log : Boolean
        Whether to log the message.
    plot : Boolean
        Whether to plot the median background by science
        extension. Defualt is False.
    plot_settings : dict
        Dictionary of plotting settings. Contains three
        items:
            'show' : Boolean
                Whether to show the plot. Default is False.
            'save' : Boolean
                Whether to save the plot. Default is False.
            'save_dir' : str
                Where to save the plot. Default is the
                current working directory.
    """
    for raw_filepath in raw_filepaths:
        ima_filepath = raw_filepath.replace('_raw.', '_ima.')

        with fits.open(ima_filepath, mode='update') as ima:
            ima_size = ima[1].data.shape
            rootname = ima[0].header['ROOTNAME']
            target = ima[0].header['TARGNAME']
            ir_filt = ima[0].header['FILTER']
            nsamp = ima[0].header['NSAMP']

            # Define a subregion for stats, using the entire
            # image (or subarray, if applicable), minus a
            # margin of 5 pixels for the overscan regions.
            stats_region =[[5,ima_size[0]-5], [5,ima_size[0]]]
            slice_x = slice(stats_region[0][0], stats_region[0][1])
            slice_y = slice(stats_region[1][0], stats_region[1][1])

            # Subtract the median countrate from each read and add back
            # the full exposure countrate to preserve pixel statistics.
            total_countrate = np.median(ima['SCI',1].data[slice_x, slice_y])
            medians = []
            sci_exts = []

            for i in range(nsamp - 1):
                sci_ext = i+1
                med = np.median(ima['SCI',sci_ext].data[slice_y, slice_x])
                ima['SCI',sci_ext].data += total_countrate - med
                display_message(verbose=verbose, log=log, log_type='info',
                                message=f'{ima_filepath} [SCI,{sci_ext}] median '\
                                        f'background = {med:.3f}')
                sci_exts.append(sci_ext)
                medians.append(med)

            if plot:
                obs_info={'rootname': rootname, 'ir_filt': ir_filt, 'target': target}

                plot_helium_bg(sci_exts, medians, obs_info, plot_settings)

                if plot_settings['save']:
                    display_message(verbose=verbose, log=log, log_type='info',
                                    message=f'Saved plot for {obs_info["rootname"]}')

            # Turn back on the ramp-fitting for running calwf3 in the next step.
            ima[0].header['CRCORR'] ='PERFORM'
            display_message(verbose=verbose, log=log, log_type='info',
                            message=f'Set CRCORR to PERFORM for modified {ima_filepath}')

        display_message(verbose=verbose, log=log, log_type='info',
                        message=f'Closed {ima_filepath}')


def plot_helium_bg(sci_exts, medians, obs_info, plot_settings):
    """
    Parameters
    ----------
    sci_exts : list of int
        List of the numbered science extensions.
    medians : list of floats
        List of median background value in electrons.
    obs_info : dict
        Dictionary of observation information, including
        exposure rootname, filter name, and target name.
    plot_settings : dict
        Dictionary of Boolean plot settings.
    """
    fig, _ax = plt.subplots(figsize=(8,5))
    _ax.plot(sci_exts, medians, c='red', label=obs_info["rootname"])
    _ax.set_xlabel('SCI Exposure Number')
    _ax.set_ylabel('Median Background Value (e-)')
    _ax.legend(loc=2)
    _ax.set_title(f'WFC3/IR {obs_info["ir_filt"]}, {obs_info["target"]}')
    fig.tight_layout()

    if plot_settings['save']:
        filename = f'he_{obs_info["ir_filt"]}_{obs_info["target"]}_'\
                   f'{obs_info["rootname"]}.jpg'
        plt.savefig(os.path.join(plot_settings['save_dir'], filename), dpi=200)

    if plot_settings['show']:
        plt.show()

    plt.close()


def run_wf3ir(raw_files, verbose, log):
    """
    """
    for raw_file in raw_files:
        ima_file = raw_file.replace('raw','ima')

        wfc3tools.wf3ir(ima_file, verbose=True);

    for raw_file in raw_files:
        ima_file = raw_file.replace('raw','ima')
        if os.path.exists(ima_file):
            display_message(verbose=verbose, log=log, log_type='info',
                            message=f'Found new IMA: {ima_file}')
        else:
            display_message(verbose=verbose, log=log, log_type='info',
                            message=f'No new IMA found: {ima_file}')

#
# def rerun_calwf3(ima_filepaths, verbose, log):
#     """
#     Parameters
#     ----------
#     ima_filepaths : list of str
#         List of strings representing the full filepaths to
#         the helium-corrected IMA files.
#     """
#     for ima_filepath in ima_filepaths:
#         try:
#             asn = glob(f'{ima_filepath[:-12]}*_asn.fits')[0]
#             with fits.open(ima_filepath, mode='update') as ima:
#                 ima[0].header['ASN_TAB'] = asn
#                 display_message(log=log, verbose=verbose, log_type='info',
#                                 message=f'Updated ASN_TAB: {ima[0].header["ASN_TAB"]}')
#                 if ima[0].header['CRCORR'] != 'PERFORM':
#                     ima[0].header['CRCORR'] = 'PERFORM'
#                     display_message(verbose=verbose, log=log, log_type='info',
#                                     message='For some reason CRCORR was not set to'\
#                                             f'PERFORM for {ima_filepath}. Fixed it.')
#
#     #        with CaptureOutput() as outputs:
#         except IndexError:
#             display_message(log=log, verbose=verbose, log_type='error',
#                             message=f'No matching ASN file found for {ima_filepath}')
#
#
#     for ima_filepath in ima_filepaths:
#         wfc3tools.calwf3(ima_filepath, debug=True, verbose=True, save_tmp=True, printtime=True)
#         time.sleep(30)

#        for output in outputs:
#            display_message(verbose=verbose, log=log, log_type='info',
#                            message=f'\t{output}')


#python ir_phot_pipeline.py --trial --local --name 2023-09-06_14-59-16 --log --get_new_data --run_ap_phot --filters F105W --proposals 11451 --targets GD153 --file_type flt --ap_phot_flt --helium_corr


def helium_correction(files, verbose, log):
    print('--------FILEPATHS--------')
    for file in files:
        print(f'\f{file}')

    rootnames = [f.split('/')[-1].split('_')[0] for f in self.filepaths]
    groups = group_rootnames_by_ipppss(rootnames)

    display_message(verbose=self.args.verbose,
                    log=self.args.log,
                    message=f"Number of visits in ObsBatch: {len(groups)}",
                    log_type='info')

    for i, group in enumerate(groups):
        filepaths = [f for f in self.filepaths
                     if f.split('/')[-1].split('_')[0] in groups[group]]

    display_message(verbose=self.args.verbose,
                    log=self.args.log,
                    message=f"Visit group {i+1}/{len(groups)}: {len(groups[group])} "\
                            f"rootnames and {len(filepaths)} files...",
                    log_type='info')

    run_bestrefs(filepaths, self.args.verbose, self.args.log)

    for file_type in ['ima', 'flt']:
        files = [f.replace('_raw', f'_{file_type}') for f in filepaths]

        for file in files:
            new_name = file.replace(f'_{file_type}.', f'_mast_{file_type}.')
            os.rename(file, new_name)

            display_message(verbose=self.args.verbose, log=self.args.log,
                            log_type='info',
                            message=f'Renamed {file} to {new_name}')

    display_message(verbose=self.args.verbose, log=self.args.log,
                    log_type='info',
                    message=f'Running calwf3 on {len(filepaths)} RAWs...')

    run_calwf3_nrf(filepaths, verbose=self.args.verbose, log=self.args.log)

    rename_flts(filepaths, verbose=self.args.verbose, log=self.args.log)
