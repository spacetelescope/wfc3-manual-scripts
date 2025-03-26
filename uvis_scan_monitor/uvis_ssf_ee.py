#!/usr/bin/env python
"""
Note - still in progress
uvis_ssf_ee.py replaces scan_ee.py

This script:
1) Takes the PSFs generated from uvis_convolve_psf.py and uses
aperture photometry to calculate the EE correction.
2) Saves the EE corrections to a table.

"""
from argparse import ArgumentParser
import numpy as np
import os
import sys

from astropy.io import fits
from astropy.table import Table
import matplotlib.pyplot as plt

import wfc3_phot_tools.spatial_scan.phot_tools as pt

from toolbox import check_subdirectory, display_message, make_timestamp
from uvis_convolve_psf import reformat_syn_args


def get_ssf_ee(ssf_data, ap_dim, sky_ap_dim, center=(256, 256), n_pix=30,
               verbose=True, log=False):
    """
    Calculates EE correction with aperture photometry.
    We don't call detect_sources_scan() like we normally
    do for the scan aperture photometry routine because
    these PSFs were generated to have a center point of
    (256, 256) and don't have an angle.

    MM: does this need to change for blended EE calc?

    Parameters
    ----------
    ssf_data : array
        Array of data read in from PSF file.
    ap_dim : tuple of int
        Photometric aperture, in pixels.
    sky_ap_dim : tuple of int
        Inner dimensions of sky background rind.
    sky_thickness : int
        Width of the sky background rind in pixels.
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
    syn_apphot = pt.aperture_photometry_scan(ssf_data, center[0], center[1],
                                             ap_dim[0], ap_dim[1], theta=0.0,
                                             show=False, plt_title=None)
    syn_sum = syn_apphot['aperture_sum'][0]

    mean_bg = pt.calc_sky(ssf_data, center[0], center[1],
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

    ee_phot = phot_syn / np.sum(ssf_data)
    ee_bg = rind_bg / np.sum(ssf_data)   # EE of background rind itself

    return ee_phot, ee_bg


def write_ee_tbl(ee_tbl, parent_dir, csv_name, verbose, log):
    """Write out encircled energy table

    Parameters
    ----------
    ee_tbl :
    parent_dir :
    csv_name :
    verbose :
    log :
    """
    ee_file = os.path.join(parent_dir, 'ee', csv_name)

    if os.path.exists(ee_file):
        alt_file = ee_file.replace('.csv', f'_{make_timestamp()}.csv')
        decision = input(f'File already exists at {ee_file}. '\
                         'Please choose from the following options:'\
                         '\n\t1. Overwrite existing file'\
                         f'\n\t2. Save as new file {alt_file.split("/")[-1]}'\
                         '\n\t3. Do not write out file'\
                         '\n\t\t')

        if decision == '1':
            ee_tbl.write(ee_file, format='csv', overwrite=True)
            display_message(verbose, log, log_type='info',
                            message='Option 1 selected. Overwrote existing '\
                                    f'EE table at {ee_file}')

        elif decision == '2':
            ee_tbl.write(alt_file, format='csv')
            display_message(verbose, log, log_type='info',
                            message='Option 1 selected. Wrote EE table to '\
                                    f'{alt_file}')
        else:
            display_message(verbose, log, log_type='info', message='Option 3 '\
                            'selected. Did not write out EE table.')

    else:
        ee_tbl.write(ee_file, format='csv')
        display_message(verbose, log, log_type='info',
                        message=f'Wrote EE table to {ee_file}')

    return ee_tbl


def parse_args():
    """
    Parses command line arguments.

    Returns
    -------
    args : `argparse.Namespace`
        Object where the attributes correspond to the
        arguments given at the command line (and the
        default values for optional arguments, if
        applicable).

    """
    parser = ArgumentParser(prog='uvis_ssf_ee',
                            description='calculate encircled energy corrections'\
                                        ' for "scan spread functions" (SSF)',
                            epilog = 'Authors: Mariarosa Marinelli & Varun Bajaj')

    parser.add_argument("-v", "--verbose",
                        help="when set, prints statements to command line",
                        action="store_true")
    parser.add_argument("-l", "--log",
                        help="when set, logs statements to log file",
                        action="store_true")
    parser.add_argument("-n", "--name",
                        help="name of pipeline run where PSF files are located",
                        required=True)

    parser.add_argument("-t", "--type",
                        choices=["simple", "blended"],
                        help="type of convolved PSF to make, `simple` or `blended`",
                        required=True)

    parser.add_argument("-f", "--filters",
                        nargs="+",
                        help="filter or list of filters (default is `all`)",
                        default=["all"])

    parser.add_argument("-u", "--uvis",
                        help="list integer number(s) for UVIS CCD(s)",
                        nargs="+",
                        type=int,
                        default=[1, 2])

    parser.add_argument("--ap_dim",
                        help="photometric aperture dimensions: x_px y_px",
                        nargs=2,
                        type=int,
                        default=[44, 268])
    parser.add_argument("--sky_ap_dim",
                        help="sky rind inner dimensions: x_px y_px "\
                                "(default is 300px wide and 400px tall)",
                        nargs=2,
                        type=int,
                        default=[300,400])
    parser.add_argument("--center",
                        help="center point coordinates (x, y)",
                        nargs=2,
                        type=int,
                        default=[256, 256])
    parser.add_argument("--sky_thickness",
                        help="sky background rind thickness: px (default is 30)",
                        default=30,
                        type=int)

    args = reformat_syn_args(parser.parse_args())

    return args


def uvis_ssf_ee(psf_type, uvis, filters, parent_dir,
                ap_dim, sky_ap_dim, sky_thickness, center,
                verbose=True, log=False):
    """
    Main function.

    Parameters
    ----------
    psf_type :
    uvis :
    filters :
    parent_dir :
    ap_dim :
    sky_ap_dim :
    ee_dir :
    """
    ee_dir = check_subdirectory(parent_dir, 'ee', verbose, log)

    rows = []

    for u in uvis:
        for filt in filters:
            ssf_dir = os.path.join(parent_dir, 'ssf')   # i think this should be the SSF, not PSF
            ssf_file = os.path.join(ssf_dir, f'{psf_type}_{u}_{filt}_ssf.csv')

            try:
                ssf_data = np.loadtxt(ssf_file, delimiter=',')
                phot, bg = get_ssf_ee(ssf_data, ap_dim, sky_ap_dim, center)
                rows.append([psf_type, u, filt, str(ap_dim), str(sky_ap_dim),
                             str(sky_thickness), str(center), phot, bg])

            except FileNotFoundError:
                display_message(verbose, log, log_type='critical',
                                message=f"Unable to locate SSF file {ssf_file}")

    ee_tbl = Table(rows=rows,
               names=('type', 'uvis', 'filter',
                      'ap_dim', 'sky_ap_dim', 'sky_thickness',
                      'center', 'ee_phot', 'ee_bg'))

    # Construct name of CSV file:
    ee_csv = f'{psf_type}_ee_'\
             f'APER-{ap_dim[0]}-{ap_dim[1]}_'\
             f'SKY-{sky_ap_dim[0]}-{sky_ap_dim[1]}_'\
             f'RIND-{sky_thickness}_'\
             f'CENTER-{center[0]}-{center[1]}.csv'

    ee_tbl = write_ee_tbl(ee_tbl, parent_dir, ee_csv, verbose, log)


if __name__ == '__main__':
    args = parse_args()

    uvis_ssf_ee(args.type, args.uvis, args.filters, args.dir,
                args.ap_dim, args.sky_ap_dim, args.sky_thickness,
                args.center, args.verbose, args.log)
