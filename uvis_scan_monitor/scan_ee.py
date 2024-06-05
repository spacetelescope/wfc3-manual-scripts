"""
This script:
1) Takes the PSFs generated from scan_psf.py and uses
aperture photometry to calculate the EE correction.
2) Saves the EE corrections to a table.

For now I'm hard-coding the aperture dimensions but
should add varying functionality in the future.

"""

#!/usr/bin/env python
import sys
import os
import numpy as np

from astropy.table import Table
from astropy.io import fits

import matplotlib as mpl
import matplotlib.pyplot as plt

#sys.path.append('/Users/mmarinelli/work/repos/wfc3-phot-tools/wfc3_phot_tools')

#import WFC3_phot_tools.spatial_scan.phot_tools as pt
#import spatial_scan.phot_tools as pt
import phot_tools as pt

#psf_dir = '/Users/mmarinelli/work/WFC3/uvis_scan_monitor/output/ee/lsf/'
#output_dir = '/Users/mmarinelli/work/WFC3/uvis_scan_monitor/output/ee/'

monitor_dir = '/grp/hst/wfc3v/wfc3photom/data/uvis_scan_monitor'
output_dir = os.path.join(monitor_dir, 'output')


def get_psf_ee(psf_data, ap_dim, sky_ap_dim, n_pix=30):
    """
    Calculates EE correction with aperture photometry.
    We don't call detect_sources_scan() like we normally
    do for the scan aperture photometry routine because
    these PSFs were generated to have a center point of
    (256, 256) and don't have an angle.

    MM: does this need to change for blended EE calc?

    Parameters
    ----------
    psf_data : array
        Array of data read in from PSF file.
    ap_dim : tuple of int
        Photometric aperture, in pixels.
    sky_ap_dim : tuple of int
        Inner dimensions of sky background rind.
    n_pix : int
        Number of pixels representing the thickness of the
        sky background rind.

    Returns
    -------
    ee_phot : float
        Correction to enrectangled energy at specified
        apertures. Equal to the total "flux" in the
        photometric aperture minus the "flux" in the
        background aperture, divided by the total "flux"
        in the PSF.
    ee_bg : float
        "Flux" of the background aperture divided by the
        total "flux" of the PSF
    """
    syn_apphot = pt.aperture_photometry_scan(psf_data, 256, 256,
                                             ap_dim[0], ap_dim[1], theta=0.0,
                                             show=False, plt_title=None)
    syn_sum = syn_apphot['aperture_sum'][0]

    mean_bg = pt.calc_sky(psf_data, 256, 256,
                           sky_ap_dim[1], sky_ap_dim[0],
                           n_pix=n_pix, method='mean')


    # to get the total background, multiply mean pixel level
    # by total number of pixels in the photometric aperture
    syn_bg = mean_bg[0] * (ap_dim[0] * ap_dim[1])

    # to get the total background in the sky background rind,
    # subtract the number of pixels in the inner boundary
    # from the number of pixels in the outer boundary
    # [(x + n) * (y + n)] - (x * y)
    rind_px = n_pix * (sky_ap_dim[0] + sky_ap_dim[1] + n_pix)
    rind_bg = mean_bg[0] * rind_px

    # sky-subtracted photometric total
    phot_syn = syn_sum - syn_bg

    ee_phot = phot_syn/np.sum(psf_data)
    ee_bg = rind_bg/np.sum(psf_data)   # EE of background rind itself

    return ee_phot, ee_bg

def create_ee_table(filters, uvis_names,
                    ap_dim, sky_ap_dim,
                    ee_dir, ssf_dir, psf_type):
    """
    Creates a table with the EE values.
    """
    rows = []

    for uvis_name in uvis_names:
        for filt in filters:
            fname = f'{uvis_name}_{filt}_convolvedpsf.csv'

            if psf_type == 'blended':
                fname = f'jay_{uvis_name}_{filt}_convolvedpsf.csv'

            psf = np.loadtxt(f'{ssf_dir}/{fname}', delimiter=',')

            ee_phot, ee_bg = get_psf_ee(psf, ap_dim, sky_ap_dim)

            row = [uvis_name, filt, ee_phot, ee_bg, str(ap_dim), str(sky_ap_dim), psf_type]
            rows.append(row)

    ee = Table(rows=rows,
               names=('uvis', 'filter', 'ee_phot', 'ee_bg',
                      'ap_dim', 'sky_ap_dim', 'PSF type'))

    ee.write(f'{ee_dir}/{psf_type}_{ap_dim[0]}_{ap_dim[1]}.csv',
             format='csv', overwrite=True)
    print(f'Table saved for {psf_type}')

def parse_args():
    parser = ArgumentParser(prog='uvis_ssf_to_ee',
                            description='creates a "line spread function" from '\
                                        'published EE values & empirical PSF models',
                            epilog = 'Authors: Mariarosa Marinelli, Varun Bajaj')

    parser.add_argument("-t", "--type",
                        choices=["simple", "blended"],
                        help="type of convolved PSF to make, `simple` or `blended`",
                        required=True)

#    parser.add_argument("-i", "--input_dir",
#                        help="name of input directory in uvis_scan_monitor",
#                        required=True)
    parser.add_argument("-f", "--filters",
                        nargs="+",
                        help="filter or list of filters (default is `all`)",
                        default=["all"])
    parser.add_argument("-u", "--uvis",
                        help="either 1, 2, or both",
                        choices=['1', '2', 'both'],
                        default="both")

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    test_dir = '/grp/hst/wfc3v/wfc3photom/data/uvis_scan_monitor/synphot/2023_03_14_test1'
    ssf_dir = os.path.join(test_dir, 'lsf')
    ee_dir = os.path.join(test_dir, 'ee')

    if not os.path.exists(ee_dir):
        os.mkdir(ee_dir)
        print(f'Made new directory at {ee_dir}')

    filters = ['F218W', 'F225W', 'F275W', 'F336W', 'F438W', 'F606W', 'F814W']
    uvis_names = ['uvis1', 'uvis2']

    jfilters = filters[2:]

    sky_ap_dim = (300, 400)
    ap_dim = (44, 268)

    create_ee_table(filters, uvis_names, ap_dim, sky_ap_dim, ee_dir, ssf_dir, psf_type='simple')

    create_ee_table(jfilters, ['uvis2'], ap_dim, sky_ap_dim, ee_dir, ssf_dir, psf_type='blended')
