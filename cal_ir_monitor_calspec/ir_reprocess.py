#pylint: disable=E1101
"""
The calibration pipeline (calwf3) can be very picky when
it comes to the length of the absolute filepaths. Therefore,
when we reprocess files, we move them to a staging directory.
"""
import os
import shutil
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
from tqdm import tqdm

from astropy.io import fits
import wfc3tools

from ir_config import CONFIG


def validate_nlinfile(nlinfile):
    """
    Validates that the linearity file is specified
    and exists if pipeline is set to reprocess with
    linearity correction.
    """
    assert nlinfile is not None, 'no linearity file specified'
    assert os.path.exists(nlinfile), f'no file found at {nlinfile}'


def update_nlinfile(filepath, nlinfile):
    """
    Helper function to update NLINFILE keyword.
    """
    validate_nlinfile(nlinfile)

    hdu = fits.open(filepath, mode='update')
    hdu[0].header['NLINFILE'] = nlinfile
    hdu[0].header['CRCORR'] = 'PERFORM'

    hdu.flush()
    del hdu


def update_crcorr(filepath, perform=False):
    """
    helper function for updating CRCORR
    """
    hdu = fits.open(filepath, mode='update')

    if perform:
        hdu[0].header['CRCORR'] = 'PERFORM'
    else:
        hdu[0].header['CRCORR'] = 'OMIT'

    hdu.flush()
    del hdu


def stage_files(current_dir, dest_dir):
    """
    stage files in a directory
    """
    all_files = glob(os.path.join(current_dir, 'i*'))
    moved_files = []

    for i in tqdm(range(len(all_files))):
        file = all_files[i]
#    for file in all_files:
        moved_file = os.path.join(dest_dir, os.path.basename(file))
        shutil.move(file, moved_file)
        assert os.path.exists(moved_file), f'Failed to move file {file}'
        moved_files.append(moved_file)

    return moved_files


def batch_reprocess(file_dir, lin_bool, he_bool, nlinfile):
    """
    for reprocessing multiple files
    """
    filt = file_dir.split('/')[-1]

    print(f'Staging files in {file_dir}')
    staged_files = stage_files(file_dir, CONFIG['stage_dir'])
    raw_files = [sf for sf in staged_files if 'raw' in sf]

    print(f'Reprocessing {len(raw_files)} RAW files...')
    for raw_file in raw_files:
        raw_filepath = ir_reprocess_file(raw_file, filt, lin_bool, he_bool, nlinfile)

    print(f'Moving files back to {file_dir}')
    processed_files = stage_files(CONFIG['stage_dir'], file_dir)

#    flt_filepaths = [rf.replace('raw', 'flt') for rf in processed_files]
    valid_flt_filepaths = [file for file in processed_files if 'flt' in file]

    return valid_flt_filepaths


def ir_reprocess_file(staged_raw_path, filt, lin_bool, he_bool, nlinfile=None):
    """
    reprocess
    """
    print(f'Reprocessing {staged_raw_path}')
    # Take RAW and run bestrefs to update reference files.
    run_bestrefs(staged_raw_path)

    if lin_bool:
        update_nlinfile(staged_raw_path, nlinfile)

    if he_bool and filt in ['F105W', 'F110W']:
        # set CRCORR to OMIT before running through calwf3
        # the first time
        update_crcorr(staged_raw_path, perform=False)

    # print('.'*40)
    # with fits.open(staged_raw_path) as f:
    #     print(f[0].header['CRCORR'])
    print('.'*40)
    print('First run of wf3ir')
    print('.'*40)
    wfc3tools.wf3ir(staged_raw_path, verbose=True)
    print('-'*40)

    if he_bool and filt in ['F105W', 'F110W']:
        ima_filepath = staged_raw_path.replace('_raw.', '_ima.')

        # Remove FLT so we can re-calibrate
        flt_filepath = staged_raw_path.replace('_raw.', '_flt.')
        os.remove(flt_filepath)


        remove_ima_median_bg(ima_filepath)

        # Turn back on the ramp-fitting for running calwf3.
        update_crcorr(ima_filepath, perform=True)

        print('.'*40)
        print('Second run of wf3ir')
        print('.'*40)

        wfc3tools.wf3ir(ima_filepath, verbose=True)
        # This produces _ima_ima.fits and ima_flt.fits
        cleanup_files(ima_filepath)
        print('-'*40)

    return staged_raw_path
#        cleanup_files(staged_raw_path)
#
#    else:
#        cleanup_files(staged_raw_path)


def cleanup_files(ima_filepath):
    """
    ima_filepath = _ima.fits

    remove unnecessary files and rename files as needed
    """
    print('Cleaning up files...')
    # A "double-IMA" file is produced after running
    # the edited IMA through wf3ir as input.
    ima2 = ima_filepath.replace('_ima.', '_ima_ima.')
    os.remove(ima2)
    assert not os.path.exists(ima2), f'Unable to remove {ima2}'
    print(f'  Removed "double-IMA" file {ima2}')

    imaflt = ima_filepath.replace('_ima.', '_ima_flt.')
    flt2 = ima_filepath.replace('_ima.', '_flt.')
    assert not os.path.exists(flt2), f'_flt.fits file already exists at {flt2}'

    os.rename(imaflt, flt2)
    assert os.path.exists(flt2), f'Unable to rename {imaflt} -> {flt2}'
    print(f'  Renamed {os.path.basename(imaflt)} -> {os.path.basename(flt2)}')

#    shutil.remove(raw_filepath.replace('raw', 'ima'))
#    shutil.remove(raw_filepath.replace('_raw.fits', '.tra'))


def setup_calwf3_environs():
    """
    Sets up calwf3 environment variables.
    """
    print('Checking calwf3 configuration...')

    environment_variables = {'CRDS_SERVER_URL': 'https://hst-crds.stsci.edu',
                             'CRDS_SERVER': 'https://hst-crds.stsci.edu',
                             'CRDS_PATH': './crds_cache',
                             'iref': './crds_cache/references/hst/wfc3/'}

    for env_key, env_value in environment_variables.items():
        os.environ[env_key] = env_value
        print(f"os.environ[{env_key}] has been set to {os.environ[env_key]}")

    os.system('crds list --status')


def run_bestrefs(raw_filepath):
    """
    Parameters
    ----------
    raw_filepath : list of str
        List of strings representing the full filepaths to
        the newly-downloaded RAW files.
    """
    print('Updating RAW file with bestrefs')

    #for raw_filepath in raw_filepaths:
    os.system(f"crds bestrefs --files {raw_filepath} -s 1 --update-bestrefs")


def create_subregion(hdu_shape):
    """
    Helper function
    """

    # Define a subregion for stats, using the entire
    # image (or subarray, if applicable), minus a
    # margin of 5 pixels for the overscan regions.
    stats_region =[[5,hdu_shape[0]-5], [5,hdu_shape[0]]]
    slice_x = slice(stats_region[0][0], stats_region[0][1])
    slice_y = slice(stats_region[1][0], stats_region[1][1])

    return slice_x, slice_y




def make_helium_bg_plot(ext_v_bg, obs_info):
    """
    Parameters
    ----------
    ext_v_bg : np.array
        Array of shape (2, n), where n is the number of
        science extensions. The first row corresponds to
        the numbered science extensions, while the second
        row are the median backgroun values in electrons.
    obs_info : dict
        Dictionary of observation information, including
        exposure rootname, filter name, and target name.
    """
    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(ext_v_bg[0], ext_v_bg[1], c='red', label=obs_info["rootname"])
    ax.set_xlabel('SCI Exposure Number')
    ax.set_ylabel('Median Background Value (e-)')
    ax.legend(loc=2)
    ax.set_title(f'WFC3/IR {obs_info["ir_filt"]}, {obs_info["target"]}')
    fig.tight_layout()


def remove_ima_median_bg(ima_filepath, plot=False, show_plot=False,
                         save_plot=False, save_dir=os.getcwd()):
    """
    Adapted from TVB_flattenramp_notebook.ipynb

    Removes median background and also updates the CRCORR keyword.

    Parameters
    ----------
    filepath : str
        String representation of the full path to the RAW
        file.
    plot : Boolean
        Whether to plot the median background by science
        extension. Defualt is False.
    show_plot : Boolean
        Whether to show the plot. Default is False.
    save_plot : Boolean
        Whether to save the plot. Default is False.
    save_dir : str
        Where to save the plot. Default is the
        current working directory.
    """
    print('^'*50)
    print('Removing median background of IMA file')
    print('v'*50)

    ima = fits.open(ima_filepath, mode='update')

    pri_hdr = ima[0].header

    he_data = subtract_median_bg(ima)

    if plot:
        obs_info={'rootname': pri_hdr['ROOTNAME'],
                  'ir_filt': pri_hdr['FILTER'],
                  'target': pri_hdr['TARGNAME']}

        make_helium_bg_plot(he_data, obs_info)

        if save_plot:
            save_helium_bg_plot(obs_info, save_dir)

        if show_plot:
            plt.show()

        plt.close()

    ima.flush()
    print(f'Removed median background and closed {ima_filepath}')

    del ima


def save_helium_bg_plot(obs_info, save_dir):
    """
    Helper function
    """
    filename = f'he_{obs_info["ir_filt"]}_{obs_info["target"]}_'\
               f'{obs_info["rootname"]}.jpg'
    plt.savefig(os.path.join(save_dir, filename), dpi=200)

    print(f'Saved plot for {obs_info["rootname"]}')


def subtract_median_bg(hdu):
    """
    Subtracts median background
    """
    sub_x, sub_y = create_subregion(hdu[1].data.shape)
    total_countrate = np.median(hdu['SCI',1].data[sub_y, sub_x])

    helium_data = np.zeros((2, hdu[0].header['NSAMP'] - 1))

    for i in range(hdu[0].header['NSAMP'] - 1):
        sci_ext = i + 1
        med = np.median(hdu['SCI', sci_ext].data[sub_y, sub_x])
        hdu['SCI', sci_ext].data += total_countrate - med
        #print(f'{ima_filepath} [SCI,{sci_ext}] median background = '\
        #      f'{med:.3f}')
        #sci_exts.append(sci_ext)
        #medians.append(med)
        helium_data[0, i] = sci_ext
        helium_data[1, i] = med

    return helium_data
