from astropy.convolution import convolve, Gaussian2DKernel
from astropy.io import fits
from astropy.stats import sigma_clipped_stats, gaussian_fwhm_to_sigma
from astropy.table import Table, vstack
from astropy.time import Time
import copy
from glob import glob
import matplotlib
from matplotlib.colors import LogNorm, Normalize
import matplotlib.pyplot as plt
import numpy as np
import os
import photutils
import photutils.segmentation as phot_seg
from scipy.stats import linregress, sigmaclip

def check_rind_parameters(data, scan_x, scan_y, sky_ap_dim, n_pix):
    """
    purpose is to make sure that the entirety of the sky rind is in the subarray

    Parameters
    ----------
    data : array-like
        Input data array.
    scan_x, scan_y : floats
        The center of the detected source, in pixels.
    sky_ap_dim : tuple of int
        The dimensions of the inner boundary of the sky
        background rind, in format (x_pixels, y_pixels).
    n_pix : int
        Width of the sky background rind in pixels.

    Returns
    -------
    rind_fits : Boolean
        If `False`, the sky background rind cannot fit on
        the subarray when centered on the detected source,
        indicating that the scan should be rejected.
    """
    data_dims = data.shape

    ap_y = sky_ap_dim[1]/2.   # 400
    ap_x = sky_ap_dim[0]/2.   # 300

    x_l = scan_x - ap_x - n_pix
    x_r = scan_x + ap_x + n_pix

    y_b = scan_y - ap_y - n_pix
    y_t = scan_y + ap_y + n_pix


    if (x_l > 10) and \
       (y_b > 10) and \
       (x_r < (data_dims[0] - 10)) and \
       (y_t < (data_dims[1] - 10)):
        rind_fits = True

    else:
        rind_fits = False
        print('\tSky background rind will not fit in the data array.')

    return rind_fits

def check_scan_angle(orientation):
    """
    Parameter
    ---------
    orientation : float
        The angle between the x-axis and the major axis of
        a 2D Gaussian function that has the same second-
        order moments as the detected source, in degrees.

    Returns
    -------
    angle_good : Boolean
        If `False`, then the angle of the detected source
        indicates that the scan should be rejected.
    """
    if orientation < 0:              # negative
        offset = 90 + orientation    # positive offset
    else:                            # positive
        offset = orientation - 90    # negative offset

    if np.abs(offset) < 5:
        angle_good = True
    else:
        angle_good = False
        print('\tScan angle is offset from the vertical by '\
              f'{offset:.4f} degrees.')

    return angle_good


def check_source_shape(eccentricity, semimajor_sigma, semiminor_sigma):
    """
    Parameters
    ----------
    eccentricity : float
        The eccentricity of the 2D Gaussian function that
        has the same second-order moments as the source.
    semimajor_sigma : float
        1-sigma standard deviation along the semimajor axis
        of the 2D Gaussian function with the same second-
        order central moments as the source, in pixels.
    semiminor_sigma : float
        1-sigma standard deviation along the semiminor axis
        of the 2D Gaussian function with the same second-
        order central moments as the source, in pixels.

    Returns
    -------
    shape_good : Boolean
        If `False`, then the shape of the detected sources
        indicates that the scan should be rejected.
    """
    if eccentricity < 0.98:
        shape_good = False
    else:
        shape_good = True

    if not shape_good:
        print(f'\tSource eccentricity is {eccentricity:.6f}')

    return shape_good

def assess_scan_quality(data, sky_ap_dim, n_pix, plot=True):
    """
    This function determines whether or not a scan should be
    used for photometry on the basis of four assessments:
        1. If a source is detected. If one is not detected,
           the `detect_sources()` function will be returned
           as a NoneType object.
        2. If the entirety of the sky background rind will
           fit on the subarray when centered on a detected
           source. Evaluated by `check_rind_parameters()`.
        3. If the angle of the detected source is within a
           5 degree offset from the vertical. If not, the
           corresponding scan may have been affected by
           loss of lock during the exposure. Evaluted by
           `check_scan_angle()`.
        4. If the shape of a 2D Gaussian function with the
           same second-order central moments as the
           detected source is not as expected, which may
           be judged by the following:
               - If the eccentricity (the fraction of
                 distance along the semimajor axis at which
                 the focus lies) of said Gaussian is less
                 than 0.98.
               - If the ratio of the 1-sigma standard
                 deviations along the semimajor axis to the
                 semiminor axis is less than 20.
           Evaluated by `check_source_shape()`.

    Parameters
    ----------
    data : array-like
        Input data array.
    sky_ap_dim : tuple of int
        The dimensions of the inner boundary of the sky
        background rind, in format (x_pixels, y_pixels).
    n_pix : int
        Width of the sky background rind in pixels.
    plot : Boolean
        Whether or not to show a source detection plot.

    Returns
    -------
    use_scan : Boolean
        If `False`, indicates that the scan should not be
        used for photometry.
    tbl : `astropy.table.table.QTable` or NoneType
        If a source is detected, this will return a QTable
        of its properties. If no source is detected, will
        return None.
    """
    use_scan = True

    threshold = phot_seg.detect_threshold(data, nsigma=5)
    seg_img = phot_seg.detect_sources(data, threshold=threshold, npixels=250)

    if seg_img is None:
        print('Could not find a source. Will not use this '\
              'observation for photometry.')
        use_scan = False
        tbl = None

    else:
        cat = phot_seg.SourceCatalog(data, seg_img)
        tbl = cat.to_table()

        rind_fits = check_rind_parameters(data,
                                          tbl['xcentroid'][0],
                                          tbl['ycentroid'][0],
                                          sky_ap_dim, n_pix)

        angle_good = check_scan_angle(tbl['orientation'][0].value)
        shape_good = check_source_shape(tbl['eccentricity'][0].value,
                                        tbl['semimajor_sigma'][0].value,
                                        tbl['semiminor_sigma'][0].value)

        if rind_fits and angle_good and shape_good:
            use_scan = True
        else:
            use_scan = False
            print('\tWill not use this scan for photometry.')

        if plot:
            fig, ax = plt.subplots(1,2,figsize=(14,7))
            ax[0].imshow(data, origin='lower', norm=LogNorm())
            ax[1].imshow(seg_img.data, origin='lower')
            plt.show()
            plt.close()

    return use_scan, tbl
