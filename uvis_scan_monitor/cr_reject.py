"""
deprecated by wfc3_phot_tools stabilized version


    This module contains functions for identifying and
    reparing CR hits in spatially scanned data. Optimized
    for WFC3/UVIS scans that are nearly vertical or
    horizontal.

    Adapted from an IDL routine by S. Casertano for STIS.

    Authors
    -------
        Mariarosa Marinelli, 2022
        Clare Shanahan, May 2018

    Use
    ---
    This script is intended to be imported:

            from wfc3_phot_tools.spatial_scan import cr_reject

    Notes
    -----
    Currently, these routines are optimized for data that
    are nearly vertically or horizontally scanned.
    Functionality for arbitrary scan angle will be added
    later. If your scans are at a significant angle on the
    detector, you can rotate the image. Also, keep in mind
    that these routines have been optimized and tested on a
    specific WFC3/UVIS data set, so they may not be
    universally useful and output should be inspected
    carefully.

    The main routine is `mask_and_repair_flagged_pixels`.
    This function takes in a 2D data array and returns the
    corrected data and the mask indicating CR hits. This
    function calls make_cr_mask as a first pass to identify
    CRs in data, and then does a second step to identify
    CRs in the tail of scans.

    The function `make_crcorr_file_scan_wfc3` takes an FLC
    or FLT as input, uses the 'scan_ang' keyword to
    determine if scan was vertical or horizonal, runs
    mask_and_repair_flagged_pixels on the data in the
    specified fits extension, and finally writes out a
    single-extension fits file with the corrected data.

"""

import os
import copy
import glob
import numpy as np
import pandas as pd
from astropy.io import fits
from scipy import interpolate
from scipy.ndimage import generate_binary_structure, label
from scipy.ndimage.measurements import find_objects
from scipy.signal import medfilt

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def unmask_isolated_pixels(data):
    """
    Helper function to remove the CR mask for isolated
    pixels. Here, we have defined isolated pixels as
    touching fewer than 2 other pixels, including
    diagonally.

    Parameters
    ----------
    data : 'numpy.ndarray'
        Array of CR mask data.

    Returns
    -------
    unmasked_array : 'numpy.ndarray'
        Array of CR mask data with isolated pixels removed.

    Notes
    -----
    To account for endian weirdness of input data, this
    function begin with adding a float zero to the data

    See documentation of this issue at:
        https://github.com/astropy/astropy/issues/1156

    """
    array = data + 0.

    s = generate_binary_structure(np.ndim(array), 1)

    labeled_array, num_features = label(array, structure=s)

    feature_labels = []

    for i in range(num_features):
        loc = find_objects(labeled_array)[i]

        feature_shape = labeled_array[loc]
        feature_size = len(feature_shape)

        if feature_size < 3:
            feature_labels.append(feature_shape[0][0])

    mask_indices = []
    for i in feature_labels:
        indices = np.ndarray.nonzero(labeled_array == i)

        row_ind = indices[0]
        col_ind = indices[1]

        for r, c in zip(row_ind, col_ind):
            mask_indices.append((r,c))

    unmasked_array = copy.deepcopy(array)

    for mask_index in mask_indices:
        unmasked_array[mask_index] = 0.

    return unmasked_array


def make_cr_mask(data, nlen=4, ncomp=10, mult=4):
    """
    Create a boolean mask of image where cosmic ray hits
    have been detected.

    Identifies suspected CRs in data as pixels that are
    more than `mult` sigma above the median of ncomp pixels
    +`nlen` ablve and -`nlen` below them. It is expected
    that valid pixels may be above the median of pixels at
    higher Y *or* lower Y because a scan is starting or
    ending, but not both. The filter may mark pixels
    incorrectly in the presence of anomalous low pixels.
    The few tests so far suggest it may be a little too
    agressive. Returns a mask (2D array) of identified
    cosmic rays in data.

    This function tends to miss CRs in scan tails. A
    second pass is done along with this function in
    `mask_and_repair_flagged_pixels` to fully correct
    spatial scans.

    Parameters
    ----------
    data : `numpy.ndarray`
        2D numpy array of floats.
    nlen : int
        Integer number of pixels above and below a given
        pixel that will be used to compute the median.
    ncomp : int
        Number of pixels used to compute surrounding
        median.
    mult : int
        Sigma, above the median, used as a threshold for a
        CR detection.

    Returns
    -------
    mask : `numpy.ndarray`
        2D numpy boolean mask for data marking locations
        of detected CRs. Pixels with value 1 signify clean
        data, and 0 indicate a CR.

    """

    data = copy.copy(data)
    nrange = nlen + ncomp

    upmedian = np.zeros(data.shape)  # median of NCOMP pixels above Y+NLEN
    downmedian = np.zeros(data.shape)  # median of NCOMP pixels below Y-NLEN
    dispimage = np.zeros(data.shape)  # rms of these 2*NCOMP pixels
    updisp = np.zeros(data.shape)
    downdisp = np.zeros(data.shape)

    ilow = 0  # column 0
    ihigh = data.shape[1] - 1  # last column
    jlow = nrange
    jhigh = data.shape[0] - nrange - 1

    for i in range(ilow, ihigh+1):  # rows
        for j in range(jlow, jhigh+1):  # columns
            upvec = data[j+nlen:j+nlen+ncomp-1+1, i]
            downvec = data[j-nlen-ncomp+1:j-nlen+1, i]
            upmedian[j, i] = np.median(upvec)
            downmedian[j, i] = np.median(downvec)
            dispimage[j, i] = np.std(np.concatenate((upvec, downvec)))
            updisp[j, i] = np.std(upvec)
            downdisp[j, i] = np.std(downvec)
            updown = np.concatenate((upvec, downvec))

    mask = (data > ((upmedian + mult * dispimage))) & \
           (data > ((downmedian + mult * dispimage)))

    return mask


def _fill_nan_interp(im_arr):
    """
    Helper function to interpolate data with NaNs, used to
    replace CR hits with surrounding 'good' pixels.

    Parameters
    ----------
    im_arr : `numpy.ndarray`
        2D array of input data containing NaNs.

    Returns
    -------
    trans_im_arr.T : `numpy.ndarray`
        2D array of input data, with interpolated values.

    Notes
    -----
    Currently does not handle top 10/bottom 10 rows as
    intended. Has trouble at the edges of the detector
    (ex. the top of the Amp A 512x512 subarray).

    """
    trans_im_arr = im_arr.T

    for c in range(0, im_arr.shape[1]):
        row = trans_im_arr[c]  # this doesn't do anything really
        if np.isnan(row[0]):
            row[0] = row[~np.isnan(row)][0]
        if np.isnan(row[-1]):
            row[-1] = row[~np.isnan(row)][-1]
        trans_im_arr[c] = pd.Series(row).interpolate()
    return trans_im_arr.T


def _determine_scan_orientation_wfc3(hdr):
    """
    Helper function that uses 'SCAN_ANG' header keyword to
    determine if scan is vertical or horizontal.

    Parameter
    ---------
    hdr :
        Header of the fits file data extension for the file
        being processed

    Returns
    -------
    scan_orient : string
        Scan orientation, either 'H' for horizontal or 'V'
        for vertical.

    Notes
    -----
    Verify that the hdr is always the data/science header.

    """

    scan_ang = hdr['SCAN_ANG']
    if np.abs(scan_ang - 138.5) < 5:
        scan_orient = 'H'
    else:
        scan_orient = 'V'

    return scan_orient


def mask_and_repair_flagged_pixels(data, scan_orient, mult):
    """
    Given a 2D array of data, as well as the orientation of
    the scan (vertical or horizontal), this function will
    identify and correct cosmic rays and return both the
    corrected data and the mask identifying CRs.

    This process consists of two steps. First, make_cr_mask
    is run on the data to identify CR hits in the image.
    To prevent overly-aggressive flagging, any isolated
    pixels (pixels touching less than 2 other flagged
    pixels) are un-masked, as they are unlikely to be
    actual CRs.

    Next, a second pass is done on the data to identify CR
    hits in the tails of the scan, which make_cr_mask tends
    to miss due to the strong gradient in the tails of
    scans.

    Pixels identified as CRs are replaced with a linear
    interpolation of the 2 nearest surrounding 'good' pixels
    above or below, in the direction of the scan.

    Parameters
    ----------
    data : `numpy.ndarray`
        2D array of data
    scan_orient : str
        'V' for vertical scans or 'H' for horizontal.

    Returns
    -------
    (corrected_data, mask) : Tuple of 'numpy.ndarray'
        Tuple of 2D arrays containing the repaired data,
        and mask, respectivley.

    Notes
    -----
    For one particular scan, this function triggered a
    critical failure during the processing of the
    tails of the scan and I'm still unsure why.

    idx5a8s6q is the scan rootname.

    """
    data = copy.copy(data)

    if scan_orient == 'H':
        data = data.T

    ###### start pass 1
    mask = 1.*make_cr_mask(data, mult)

    # remove isolated pixels from mask
    mask = unmask_isolated_pixels(mask)

    # replace masked pix with the linear interp. of the nearest 2 unmasked pix
    k = np.where(mask > 0)
    data[k] = np.nan
    data = _fill_nan_interp(data)

    ##### end pass 1

    # pass #2
    # process the ends of the trails differently than the rest

    rolled_data = np.roll(data, 1, 0)
    gr = data - np.roll(data, 1, 0)

    gr_min = np.nanmin(gr.flatten())
    k_min = np.where(gr == gr_min)  # y,x where min occurs
    if len(k_min[0]) > 1:
        # select only one of these, sometimes it will find two nearby
        k_min = [np.array(item[0]) for item in k_min]

    gr_max = np.nanmax(gr.flatten())
    k_max = np.where(gr == gr_max)  # y,x where max occurs
    #print(k_max)
    if len(k_max[0]) > 1:
        # select only one of these, sometimes it will find two nearby
        k_max = [np.array(item[0]) for item in k_max]

    mult = 5
    dy = 5

    # the following for-loop caused a critical error for
    # one scan, and I'm still not sure why. -MM

    range_low = min([k_min[1], k_max[1]])[0] - 3
    range_high = max([k_min[1], k_max[1]])[0] + 3 + 1

    if range_low < 0:
        range_low = 0

    if range_high > len(data):
        range_high = len(data)

    for j in range(range_low, range_high):

        try:
            v = data[:, j]

            mv = medfilt(v, 5)
            v_mv = v - mv
            sig = np.nanstd(v_mv)

            k = (np.where(v_mv > (mult * sig)))[0]

            if len(k) > 0:
                for i, val in enumerate(k):
                    if ((np.abs(k[i] - k_min[0]) < dy) or
                        (np.abs(k[i] - k_max[0]) < dy)):
                        mask[k, j] = 1
        except IndexError:
            fig, ax = plt.subplots(1,3,figsize=(15,5))
            ax[0].imshow(data, origin='lower', norm=LogNorm())
            ax[1].imshow(rolled_data, origin='lower', norm=LogNorm())
            ax[2].imshow(gr, origin='lower')
            plt.savefig(f'{j}.jpg', dpi=250)
            plt.close()
            print(f'IndexError: saved plots at {j}.jpg')


    k = np.where(mask > 0)

    if len(k) > 0:
        data[k] = np.nan

    corrected_data = _fill_nan_interp(data)

    if scan_orient == 'H':
        corrected_data = corrected_data.T
        mask = mask.T

    return (corrected_data, mask)

def _write_fcr(input_file, output_dir, corrected_data, ext, file_type):
    """
    Writes out CR corrected data (corrected_data) as single
    extension fits file in output_dir. The 0th and 'ext'
    headers are concatenated and written to the output
    file. The name of this file will be the same as the
    input but with file_type 'flt' or 'flc' replaced with
    'fcr'.

    Parameters
    ----------
    input_file : str
        Full path to input fits file.
    output_dir : str
        Directory where corrected files should be output.
        Defaults to input_file directory.
    corrected_data :

    ext :

    file_type : str
        Either 'flt' or 'flc'.

    """

    # concatenate 0th and ['SCI', ext] headers
    hdr_out = fits.open(input_file)[0].header + \
        fits.open(input_file)['SCI', ext].header

    hdu_new = fits.PrimaryHDU(corrected_data, header=hdr_out)
    output_path = output_dir+os.path.basename(input_file).\
        replace('{}.fits'.format(file_type), 'fcr.fits')

    if os.path.isfile(output_path):
        os.remove(output_path)
    print('Writing', output_path)
    hdu_new.writeto(output_path)


def _write_mask(input_file, output_dir, mask, ext, file_type):
    """
    Helper function that writes out CR mask as single-
    extension fits file in output_dir. See _write_fcr().

    Parameters
    ----------
    input_file : str
        Full path to input fits file.
    output_dir : str
        Directory where corrected files should be output.
        Defaults to input_file directory.
    mask : `numpy.ndarray`
        2D numpy boolean mask for data marking locations
        of detected CRs. Pixels with value 1 signify clean
        data, and 0 indicate a CR.
    ext : int
        FITS extension of data.
    file_type :

    """

    # concatenate 0th and ['SCI', ext] headers
    hdr_out = fits.open(input_file)[0].header + \
        fits.open(input_file)['SCI', ext].header

    hdu_new = fits.PrimaryHDU(mask, header=hdr_out)

    output_path = output_dir + os.path.basename(input_file).\
        replace('{}.fits'.format(file_type), 'mask.fits')

    if os.path.isfile(output_path):
        os.remove(output_path)
    print('Writing', output_path)
    hdu_new.writeto(output_path)


def make_crcorr_file_scan_wfc3(input_file, mult=4, output_dir=None, ext=1,
                               write_mask=True):
    """
    Wrapper function that calls routine to identify and
    correct cosmic rays in spatially scanned HST flt.fits
    or flc.fits images and write out a corrected single-
    extension 'fcr.fits' file. Only the data in the input
    file at the specified extension 'ext' is corrected and
    written out; keep this in mind when working with multi-
    extension fits files.

    This function calls the main CR correction routine
    `mask_and_repair_flagged_pixels`, and writes the output
    of this to file. If 'write_mask' is set to True, the
    mask indentifying locations of CR hits in the data will
    be written out as well, to a 'mask.fits' file.

    Currently, this function is optimized for files that
    are nearly vertically or horizontally scanned.
    Functionality for arbitrary scan angle will be added
    later. If your scans are at a significant angle on the
    detector, you can rotate the image, padding it with a
    fill value, save, and pass that as input to this
    function.

    Parameters
    ----------
    input_file : str
        Full path to input fits file.
    mult : int
        Sigma, above the median, used as a threshold for a
        CR detection.
    output_dir : str
        Directory where corrected files should be output.
        If None, defaults to input_file directory.
    ext : int
        FITS extension of data.
    write_mask : bool
        If True, in addition to the CR corrected image the
        mask indicating the location of CR hits will be
        saved as well.

    Outputs
    -------
        A single extension CR-corrected fits file. The file
        name is the same as the input, but with 'flt' or
        'flc' changed to 'fcr'. By default, corrected files
        are output in the same directory as the input
        unless a different 'output_dir' is specified.) If
        write_mask, a 'mask.fits' file will be written out
        as well.
    """

    print('Running CR rejection on {}'.format(input_file))
    # define output directory, same directory as input
    if output_dir is None:
        output_dir = input_file.replace(os.path.basename(input_file), '')
    output_dir = os.path.join(output_dir, '')  # ensure trailing slash.
    # check that input file is either an flt or flc
    if 'flt' in os.path.basename(input_file):
        file_type = 'flt'
    elif 'flc' in os.path.basename(input_file):
        file_type = 'flc'
    else:
        raise ValueError('Input file must be flt.fits or flc.fits')

    # open file and get data, 0th header
    hdu = fits.open(input_file)
    hdr0 = hdu[0].header
    data = hdu[ext].data
    scan_orient = _determine_scan_orientation_wfc3(hdr0)

    # call CR rejection routine
    corrected_data, mask = mask_and_repair_flagged_pixels(data, scan_orient, mult)

    _write_fcr(input_file, output_dir, corrected_data, ext, file_type)

    if write_mask:
        _write_mask(input_file, output_dir, mask, ext, file_type)
