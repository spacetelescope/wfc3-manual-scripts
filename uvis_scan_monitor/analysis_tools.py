"""
TK work on!
"""

from glob import glob
import os
from astropy.table import Table, vstack
from scipy.stats import linregress
from astropy.stats import sigma_clip
import numpy as np
import argparse
from argparse import ArgumentParser

def flag_outliers(subset_data, colname='fcr_phot'):
    subset_outliers, subset_distances = [], []
    subset_data.sort('expstart_decimalyear')

    # first pass LSR to flag outliers
    # not the LSR used to compute the slope of the best-fit line!
    lsr = linregress(subset_data['expstart_decimalyear'],
                     subset_data[colname])

    y_expected = [(lsr[0]*x)+lsr[1] for x in subset_data['expstart_decimalyear']]
    residuals = [y_exp-y_act for y_exp, y_act in zip(y_expected, subset_data[colname])]

    residual_std = np.std(residuals)
    residual_mean = np.mean(residuals)

    residuals_sc = sigma_clip(residuals, cenfunc='mean')

    for i, x in enumerate(residuals_sc):
        outlier_bool = residuals_sc.mask[i]
        distance = (residual_mean-residuals_sc.base[i])/residual_std

        subset_outliers.append(f'{outlier_bool}')
        subset_distances.append(distance)

    subset_data['outlier'] = subset_outliers
    subset_data['distances'] = subset_distances

    return subset_data
#    is_outlier.extend(subset_outliers)

def process_data(data, colname='fcr_phot', write=True,
                 output_dir=None, filename='all_data.csv'):
    analyzed_data = Table()

    grouped_data = data.group_by(['filter', 'ccdamp', 'targname'])
    indices = grouped_data.groups.indices

    for i, ind in enumerate(indices):
        if i+1 < len(indices):
            endpoint = indices[i+1]
            subset_data = grouped_data[ind:endpoint]
            subset_data = flag_outliers(subset_data, colname)
            subset_data = normalize_subset(subset_data, colname)

            best_fit_line = fit_subset_data(subset_data, colname='fcr_phot')

            analyzed_data = vstack([analyzed_data, subset_data])
    if write:
        if output_dir == None:
            output_dir = os.getcwd()
        analyzed_data.write(os.path.join(output_dir, filename),
                            format='csv', overwrite=True)
        print(f'Table with all analyzed data saved to {output_dir}')

    return analyzed_data

def calc_mean_and_err(values):
    """
    Helper function to quickly calculate the mean and error
    of the mean from a set of values.

    Parameter
    ---------
    values : list-like of floats
        A set of iterable values.

    Returns
    -------
    mean : float
        Mean of input values.
    mean_err : float
        Propagated error of the mean.
    """
    mean = np.mean(values)
    sum_sqr_diff = [(val - mean)**2 for val in values]
    _n = len(values)

    mean_err = np.sqrt(np.sum(sum_sqr_diff)/(_n - 1))

    return mean, mean_err

def calc_norm_phot_err(norm_phot, phot, phot_err, mean, mean_err):
    """
    Helper function to propagate error for normalized
    photometry:
        Z = X / Y
        s_Z = Z * np.sqrt((s_X / X) ** 2 + (s_Y / Y) ** 2)

    Parameters
    ----------
    norm_phot : float
        Sky-subtracted photometric count-rate (e-/s),
        normalized to the mean value of the data set.
    phot : float
        Sky-subtracted photometric count-rate (e-/s).
    phot_err : float
        Error in the sky-subtracted photometric count-rate
        (e-/s), as calculated in the same manner as IRAF/
        DAOphot by the `wfc3_phot_tools.daophot_err`
        function `compute_phot_err_daophot()`.
    mean : float
        Mean of the sky-subtracted photometric count-rates
        (e-/s) in a data set.
    mean_err : float
        Error in the mean of the sky-subtracted photometric
        count-rates (e-/s) in a data set, as calculated by
        `calc_mean_and_err()`.
    """
    phot_err_term = (phot_err / phot)**2
    mean_err_term = (mean_err / mean)**2
    norm_phot_err = norm_phot * np.sqrt(phot_err_term + mean_err_term)

    return norm_phot_err

def normalize_subset(subset_data, colname='fcr_phot'):
    """
    subset_data : Astropy table
        Must have `outlier` column. Should have been sigma-
        clipped already.
    """
    subset_inliers = subset_data[subset_data['outlier'] == 'False']
    subset_mean, subset_mean_err = calc_mean_and_err(subset_inliers[colname])

    subset_norm_phot = [row[colname]/subset_mean
                        if row['outlier'] == 'False'
                        else -9999 for row in subset_data]

    subset_norm_phot_err = [calc_norm_phot_err(norm_phot, phot, phot_err,
                                               subset_mean, subset_mean_err)
                            if norm_phot != -9999 else -9999
                            for norm_phot, phot, phot_err
                            in zip(subset_norm_phot, subset_data[colname],
                                   subset_data[f'{colname}_rms'])]

    subset_data[f'norm_{colname}'] = subset_norm_phot
    subset_data[f'norm_{colname}_err'] = subset_norm_phot_err

    return subset_data

def fit_subset_data(subset_data, colname='fcr_phot'):
    """
    should have norm_fcr_phot and norm_fcr_phot_err columns

    Returns
    -------
    best_fit_line : tuple of floats
        The values for plotting a line of best fit to the
        subset data, in format (x_values, y_values).
    """
    subset_inliers = subset_data[subset_data['outlier'] == 'False']

    lsr = linregress(subset_inliers['expstart_decimalyear'],
                     subset_inliers[f'norm_{"fcr_phot"}'])

    xs = np.arange(min(subset_data['expstart_decimalyear']),
                   max(subset_data['expstart_decimalyear']),
                   0.1)
    ys = [lsr[0]*x + lsr[1] for x in xs]

    best_fit_line = (xs, ys)

    return best_fit_line

def filter_available_csvs(available_csvs, args, filter_on_arg):
    if len(available_csvs) > 0:
        available_csvs = [csv for csv in available_csvs
                          if len(os.path.basename(csv).split('_')) == 3]

        arg_dict = {'proposals': [[str(p) for p in args.proposals], 0],
                    'targets': [args.targets, 1],
                    'filters': [args.filters, 2]}

        available_csvs = [csv for csv in available_csvs
                          if os.path.basename(csv).split('_')[arg_dict[filter_on_arg][1]]
                          in arg_dict[filter_on_arg][0]]

    return available_csvs

def automated_analysis(args):
    """
    If both `--name` and `--directory` were specified
    on the command line, `--name` is used.
    """
    if args.name != None:
        output_dir = os.path.join('/grp/hst/wfc3v/wfc3photom/data/uvis_scan_monitor', args.name, "output")
    else:
        if args.directory == None:
            output_dir = os.getcwd()
        else:
            output_dir = args.directory

    if os.path.exists(output_dir):
        print(f'Output directory is set to {output_dir}')
        available_csvs = glob(os.path.join(output_dir, '*.csv'))

        if len(available_csvs) > 0:
            available_csvs = [csv for csv in available_csvs
                              if len(os.path.basename(csv).split('_')) == 3]
              # remove "all_data.csv, etc."

            if args.proposals != None:
                available_csvs = filter_available_csvs(available_csvs, args,
                                                       filter_on_arg='proposals')
            if args.targets != None:
                available_csvs = filter_available_csvs(available_csvs, args,
                                                       filter_on_arg='targets')

            if args.filters != None:
                available_csvs = filter_available_csvs(available_csvs, args,
                                                       filter_on_arg='filters')

            if len(available_csvs) == 0:
                print('No CSVs remain.')
            else:
                all_data_table = Table()
                for available_csv in available_csvs:
                    data_table = Table.read(available_csv, format='csv')
                    data_table['linenum'] = [str(x) for x in data_table['linenum']]
                    all_data_table = vstack([all_data_table, data_table])

                analyzed_data = process_data(all_data_table,
                                             colname='fcr_phot',
                                             write=True,
                                             output_dir=output_dir,
                                             filename='all_data.csv')
        else:
            print(f'No CSVs found in output directory {output_dir}')
    else:
        print(f'Output directory {output_dir} does not exist.')
# now can move on to actually running analysis scripts


def parse_cl_args():
    parser = ArgumentParser(prog='uvis_scan_monitor_auto_analysis',
                            description='WFC3/UVIS calibration scan '\
                                        'photometry monitor pipeline: '\
                                        'automated analysis',
                            epilog = 'Author: Mariarosa Marinelli')

    parser.add_argument("-n", "--name",
                        help="/wfc3v/wfc3photom/data/uvis_scan_monitor/<name>")
    parser.add_argument("-d", "--directory",
                        help="full path to directory with scan monitor tables")

    parser.add_argument("-p", "--proposals",
                        nargs="+",
                        type=int,
                        help="proposals to include (default is all available)")
    parser.add_argument("-f", "--filters",
                        nargs="+",
                        type=str,
                        help="filters to include (default is all available)")
    parser.add_argument("-t", "--targets",
                        nargs="+",
                        type=str,
                        help="targets to include (default is all available)")

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_cl_args()
    automated_analysis(args)
#    if args.name == None:
#        if verify_path(os.path.cwd()):
#            automated_analysis(os.path.cwd(), args)


    #if len(sys.argv) == 1:
    #    cwd = os.getcwd()
    #    if verify_path(cwd):
    #        automated_analysis(output_dir=cwd)
    #    else:
    #        print('No pipeline run name specified, and no CSV '\
    #              f'files in current working directory: {cwd}\n'\
    #              'Unable to run automated analysis script.')


    #automated_analysis(trial_dir_name)
