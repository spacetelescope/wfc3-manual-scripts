"""
Functions for compiling catalogs of WFC3/UVIS spatial scan photometry.

Author
------
    Mariarosa Marinelli

Use
---
    This module can be run from the command line, i.e.

        python compile_catalog.py --csv_dir DIRECTORY
        --norm_method METHOD --write_loc LOCATION --filename FILENAME
"""
from argparse import ArgumentParser
from astropy.table import Table, vstack
from glob import glob
import numpy as np
import os


def merge_csvs(csv_dir):
    """
    In a given directory, merges CSV files that match the
    wildcard file name: '?????_*_F???W.csv'.

    Arguments
    ---------
    csv_dir : str
        String representation of directory path.

    Returns
    -------
    csv_tbl : astropy.table.Table
        Astropy table of stacked CSVs.
    """
    csv_tbl = Table()

    # Get all CSVs in the directory. Ignore any existing
    # catalogs that aren't directly produced by the scan
    # monitor pipeline.
    csv_paths = glob(os.path.join(csv_dir, '?????_*_F???W.csv'))
    print(f'Found files:\n\t{csv_dir}\n\t{len(csv_paths)} CSV files')

    if len(csv_paths) > 0:
        for csv_path in csv_paths:
            tbl = Table.read(csv_path, format='csv')
            if len(tbl) > 0:
                tbl['linenum'] = [str(linenum) for linenum in tbl['linenum']]
                csv_tbl = vstack([csv_tbl, tbl])

            del tbl

    print(f'Merged catalog has {len(csv_tbl)} rows.')

    return csv_tbl


def normalize_phot(subset, norm_method='first', file_type='fcr'):
    """
    Function to normalize photometry by a specified method.
    Also calculates the subset standard deviation and
    subset mean to provide a 'norm_dist_std' value.

    Parameters
    ----------
    subset : `astropy.table.table`
    norm_method : str
        Method by which to normalize the data. Supported
        options: 'first' (default), 'mean', and 'median'.
    file_type : str
        Refers to the files used for the photometry. 99.99%
        of the time this should be 'fcr' (flat-fielded and
        calibrated with custom spatial scan cosmic ray
        routine applied). If importing this function, then
        you can also try it with 'flt'.

    Returns
    -------
    subset : `astropy.table.table`
        Input table with additional two columns: one
        indicating the normalized count rate (method of
        normalization being determined by the `norm_method`
        parameter) and the difference between the
        normalized count rate and the mean normalized count
        rate, in standard deviations.
    """
    if norm_method == 'first':
        first_program = subset[subset['proposid'] == np.min(subset['proposid'])]
        comp_cr = np.mean(first_program[f'{file_type}_phot'])

    elif norm_method == 'mean':
        comp_cr = np.mean(subset[f'{file_type}_phot'])

    else:
        comp_cr = np.median(subset[f'{file_type}_phot'])

    norm_colname = f'{file_type}_phot_norm_{norm_method}'

    norm_phot = [cr / comp_cr for cr in subset['fcr_phot']]
    norm_std = np.std(norm_phot)
    norm_mean = np.mean(norm_phot)
    norm_dist = [(norm_mean - norm_cr) / norm_std for norm_cr in norm_phot]

    subset[norm_colname] = norm_phot
    subset[f'norm_dist_std'] = norm_dist

    return subset


def parse_args():
    """
    Parses command line arguments, if you're running this
    as a standalone module instead of importing functions.

    Returns
    -------
    args : `argparse.Namespace`
        Namespace class object that has as attributes the
        2 configurable arguments.
    """
    parser = ArgumentParser(prog='compile_catalog',
                            description='merge UVIS scan photometry catalogs, '\
                                        'perform basic analysis',
                            epilog = 'Author: Mariarosa Marinelli')

    # Settings:
    #parser.add_argument("-a", "--analyze",
    #                    help="when set, prints statements to command line",
    #                    action="store_true")
    parser.add_argument("-d", "--csv_dir",
                        help="directory where the CSVs are located",
                        type=str)
    parser.add_argument("-n", "--norm_method",
                        choices=['mean', 'median', 'first'],
                        default='first',
                        help="how to normalize data: mean, median, or first")
    parser.add_argument("-w", "--write_loc",
                        help='where to write the compiled catalog',
                        default=os.getcwd())
    parser.add_argument("-f", "--filename",
                        help='name of compiled catalog',
                        default='all_catalog.csv')

    args = parser.parse_args()

    return args


def compile_catalog(csv_dir, norm_method):
    """
    Function to compile a catalog by merging CSVs in a
    directory and normalizing subset data (matching target/
    filter/chip).

    Parameter
    ---------
    csv_dir : str
        Where the CSVs are located. Even if other compiled
        or reduced catalogs are present, it will only grab
        those following the filename convention:
            <program>_<target>_<filter>.csv
    norm_method : str
        How to normalize photometric count rate. Valid
        options are 'first' (normalize to the mean of the
        first program's data, for each target/filter/chip
        data subset), 'mean' (normalize to the overall mean
        count rate of the target/filter/chip subset), and
        'median' (you can probably guess).

    Returns
    -------
    compiled : `astropy.table.table`
        Compiled catalog with additional columns.
    """
    print(f'Compiling catalogs....')
    csv_tbl = merge_csvs(csv_dir)

    catalog = Table()
    for targ in sorted(list(set(csv_tbl['targname']))):
        print(f'{" "*2}TARGNAME: {targ}')
        targ_tbl = csv_tbl[csv_tbl['targname'] == targ]

        for filt in sorted(list(set(targ_tbl['filter']))):
            print(f'{" "*4}FILTER: {filt}')
            filt_tbl = targ_tbl[targ_tbl['filter'] == filt]

            for chip in sorted(list(set(filt_tbl['ccdamp']))):
                print(f'{" "*6}CCDAMP: {chip}')
                chip_tbl = filt_tbl[filt_tbl['ccdamp'] == chip]

                norm_tbl = normalize_phot(chip_tbl, norm_method)
                catalog = vstack([catalog, norm_tbl])

                del norm_tbl

    return catalog


def write_catalog(catalog, write_loc, filename):
    """
    Function to write out catalog. Prints an error message
    if a file with that particular filename already exists
    (so it does not overwrite anything). Also checks to
    make sure that the file was written out.

    Parameters
    ----------
    compiled : `astropy.table.table`
        Compiled catalog with additional columns.
    write_loc : str
        String representation of directory path where
        catalog should be saved.
    filename : str
        Name of file.
    """
    catalog_filepath = os.path.join(write_loc, filename)
    print(f'Writing catalog....')

    if os.path.exists(catalog_filepath):
        print(f'ERROR: catalog already exists!\n\t{catalog_filepath}'\
              '\n\tPlease specify a different filename and/or directory.')

    else:
        try:
            catalog.write(catalog_filepath, format='csv')

            if os.path.exists(catalog_filepath):
                print(f'Wrote catalog to disk:\n\t{catalog_filepath}')
            else:
                print(f"ERROR: can't find catalog.\n\t{catalog_filepath}")

        except Exception as e:
            print(f'ERROR: unknown exception occurred:\n\t{e}')


if __name__ == '__main__':
    args = parse_args()

    catalog = compile_catalog(args.csv_dir, args.norm_method)

    write_catalog(catalog, args.write_loc, args.filename)
