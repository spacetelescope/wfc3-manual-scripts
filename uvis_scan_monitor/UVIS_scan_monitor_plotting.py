#!/usr/bin/env python

"""
Functions and utilities for handling and plotting data
generated from the UVIS spatial scan monitor pipeline.

Authors
-------
    Mariarosa Marinelli, 2021

Use
---
    This module is intended to be imported and used as part
    of the UVIS spatial scan monitor pipeline.

        import UVIS_scan_monitor_plotting as scan_plot

"""
import os

from scipy.stats import linregress

from astropy.io import fits
from astropy.table import Table, vstack
from astropy.time import Time
from astropy.stats import sigma_clip

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.patches import Patch, Rectangle
from matplotlib.lines import Line2D

import numpy as np

from UVIS_scan_monitor_pipeline import LOG_DIR

def verify_input(data_table, filter_list, target_list):
    """Checks to make sure all filters and targets exist.

    Parameters
    ----------
    data_table : `astropy.table.table.Table`
        Astropy table of all photometry data.
    filter_list : list
        List of string names of filters.
    target_list : list
        List of string names of targets.

    Returns
    -------
    data_table : `astropy.table.table.Table`
        Astropy table of all photometry data.
    filter_list : list
        List of string names of filters, minus any filters
        specified by the user that do not exist in the
        data table.
    target_list : list
        List of string names of targets, minus any targets
        specified by the user that do not exist in the
        data table.

    """
    for f in filter_list:
        try:
            data_table[data_table['filter'] == f][0]
        except IndexError:
            print(f'{f} is not a listed filter in table.')
            filter_list.remove(f)

    for t in target_list:
        try:
            data_table[data_table['targname'] == t][0]
        except IndexError:
            print(f'{t} is not a listed target in table.')
            target_list.remove(t)

    return data_table, filter_list, target_list

def extract_data(data_table, f, t):
    """
    Extracts data for both chips corresponding to
    specified filter and target, and does some light
    formatting.

    Parameters
    ----------
    data_table : `astropy.table.table.Table`
        Astropy table of all photometry data.
    f : str
        String representation of filter.
    t : str
        String representation of target.

    Returns
    -------
    a : `astropy.table.table.Table`
        Astropy table of UVIS1/CCDAMP 'A' data
        corresponding to the filter and target
        specified by `f` and `t`,
    c : `astropy.table.table.Table`
        Same as `a`, but for UVIS2/CCDAMP 'C'.
    """
    subset = data_table[data_table['filter'] == f]
    subset = subset[subset['targname'] == t]
    a = subset[subset['ccdamp'] == 'A']
    c = subset[subset['ccdamp'] == 'C']

    a.sort(['expstart'])
    c.sort(['expstart'])

    return a, c

def make_xy_lists(data_table):
    """Make lists for x and y axes data.

    Parameters
    ----------
    data_table : `astropy.table.table.Table`
        Astropy table of data for a specific filter/target/
        chip configuration.

    Returns
    -------
    xaxis : list
        List of time data, converted from MJD to decimal
        year format.
    yaxis : list
        List of countrate data.
    """
    xaxis = [Time(x, format='mjd', scale='utc') for x in data_table['expstart']]
    xaxis = [x.to_value('decimalyear', subfmt='float') for x in xaxis]
    for each in data_table.columns:
        if each[0:9] == 'countrate':
            cr_colname = each
    yaxis = data_table[cr_colname]

    return xaxis, yaxis

def make_bf_line(x_data, y_data):
    """Wrapper for SciPy linregress.

    Calculate linear regression of data and also get
    y-values for line of best fit.

    Parameters
    ----------
    x_data : list or `astropy.table.column.MaskedColumn`
        X-axis values, either in a list or as a slice of
        an Astropy table.
    y_data : list or `astropy.table.column.MaskedColumn`
        Same as x_data, but for Y-axis values.

    Returns
    -------
    lsr : `scipy.stats._stats_mstats_common.LinregressResult`
        Indexable linear regression result.
    bf_y : list
        List of y-values for the best-fit line.
    """

    lsr = linregress(x_data, y_data)
    bf_y = [(lsr[1] + lsr[0]*x) for x in x_data]

    return lsr, bf_y

def compute_residuals(chip_tup, f, t,
                      show_residuals,
                      save_residuals):
    """Compute and plot residuals.

    Parameters
    ----------
    chip_tup : tuple
        Tuple of 2 Astropy tables, corresponding to the
        data from each specific filter/target configuration
        for each chip.
    f : str
        String name of filter.
    t : str
        String name of target.
    show_residuals : bool
        Whether to show residual plots with outliers
        highlighted.
    save_residuals : bool
        Whether to save residual plots.

    Returns
    -------
    outliers : list
        List of outliers to remove from the data table.

    """
    outliers = []
    if len(chip_tup[0]) > 5:
        fig, ax = plt.subplots(1, 2, figsize=(8, 4), sharey=True)
        for c, chip in enumerate(chip_tup):
            uvis_names = ['UVIS1', 'UVIS2']
            x, y = make_xy_lists(chip)
            ax[c].axhline(0, color='gray', linestyle='dashed')
            try:
                lsr, bf_y = make_bf_line(x, y)

                residuals = []
                for xi, yi in zip(x, y):
                    y_exp = lsr[0]*xi + lsr[1]
                    residuals.append(yi - y_exp)

                ax[c].scatter(x, residuals, label='Residuals')
                ax[c].set_title(uvis_names[c], fontsize=10)
                ax[c].set_xlabel('Observation Date', fontsize=8)
                ax[c].set_ylabel('Residuals', fontsize=8)
                ax[c].set_xlim((2016.5, 2021.5))
                ax[c].set_xticklabels(ax[c].get_xticks(),
                                      rotation=45, fontsize=8)
                ax[c].set_yticklabels(ax[c].get_yticks(), fontsize=8)

                sc = sigma_clip(residuals, sigma=2.5, maxiters=5,
                                        cenfunc='median', masked=False)

                clipped = [s for s in sc]

                rej_x, rej_y = [], []

                for ind, res in enumerate(residuals):
                    if res not in clipped:
                        rej_x.append(x[ind])
                        rej_y.append(res)
                        outliers.append((uvis_names[c], x[ind], y[ind]))

                if len(rej_x) > 0:
                    ax[c].scatter(rej_x, rej_y, color='red', label='Outliers')

                ax[c].legend(loc='best', fontsize=8,
                             facecolor='w', edgecolor='k')
            except ValueError:
                pass

        if show_residuals == True:
            plt.suptitle(f'{f}/{t} Residuals', fontsize=12)
            plt.show()
        if save_residuals == True:
            plot_dir = os.path.join(LOG_DIR, 'plots/')
            if not os.path.isdir(plot_dir):
                os.mkdir(plot_dir)
            fig.savefig(f'{plot_dir}residuals_{f}_{t}.png')

        plt.close()

    return outliers

def clip_chip_data(chip_tup, outliers):
    """Removes outliers from data.

    Stacks a two-item list of Astropy tables together,
    representing the two UVIS chips. Then, compares the
    outliers from the outlier list (generated by the
    compute_residuals() function) to the table and removes
    rows containing the outliers.

    Parameters
    ----------
    chip_tup : tuple
        Tuple of Astropy tables, one table per UVIS chip.
    outliers : list
        List of outliers constructed by calling
        the compute_residuals() function.

    Returns
    -------
    chip_table : `astropy.table.table.Table`
        Astropy table of data for unique filter/
        object/chip configuration, clipped for
        outliers.
    """

    chip_table = vstack([chip_tup[0], chip_tup[1]])

    for cc in chip_table.columns:
        if cc[0:9] == 'countrate':
            cr_colname = cc

    chip_table.add_index(cr_colname)

    o_rows = []
    if len(outliers) > 0:
        for o in outliers:
            if o[2] in chip_table[cr_colname]:
                loc = int(chip_table.loc_indices[o[2]])
                o_rows.append(loc)
        chip_table.remove_rows(o_rows)

    return chip_table

def norm_by_earliest_proposal(chip_table):
    """ Normalize data by earliest proposal.

    Normalize photometry countrate over time by the mean of
    the earliest proposal's data, which roughly corresponds
    to around 6 months of data.

    Parameters
    ----------
    chip_table : `astropy.table.table.Table`
        Astropy table of data for unique filter/
        object/chip configuration, clipped for
        outliers.

    Returns
    -------
    x_axis_list : list
        List of time values corresponding with y_norm_list.
    y_norm_list : list
        List of normalized values (percent change in
        countrate over time).
    mean_list : list
        List of mean values used for normalization.
    """
    x_axis_list = []
    y_norm_list = []
    mean_list = []

    chips = ['A', 'C']


    for c in chips:
        subset = chip_table[chip_table['ccdamp'] == c]

        prop_list = list(set(subset['proposid']))
        prop_list.sort()
        e_propid = prop_list[0]

        # Find mean:
        e_data = subset[subset['proposid'] == e_propid]

        for col in e_data.columns:
            if col[0:9] == 'countrate':
                    cr_colname = col

        e_mean = np.mean(e_data[cr_colname])
        for i in range(len(subset)):
            mean_list.append(e_mean)

        xaxis, yaxis = make_xy_lists(subset)

        y_norm = [(y/e_mean -1)*100 for y in yaxis]

        for x, y in zip(xaxis, y_norm):
            x_axis_list.append(x)
            y_norm_list.append(y)

    return x_axis_list, y_norm_list, mean_list

def clean_data(data_table, filters, targets,
               show_residuals=True, save_residuals=False):
    """Remove outliers and normalize data.

    Parameters
    ----------
    data_table : `astropy.table.table.Table`
        Input table of data from photometry catalogs.
    filters : list
        List of filter names to plot.
    targets : list
        Targets to plot.
    show_residuals : bool
        Whether to show residual plots with outliers
        highlighted. Default is True.
    save_residuals : bool
        Whether to save residual plots. Default is False.

    Returns
    -------
    clean_table : `astropy.table.table.Table`
        Data table that has been cleaned (outliers
        have been removed) and normalized.
    outlier_table : `astropy.table.table.Table`
        Data table for outliers removed from final plots
        and calculations.
    """
    clean_table = Table()
    dtype_list = [data_table.dtype[x] for x in data_table.colnames]
    outlier_table = Table(names=data_table.colnames, dtype=dtype_list)

    data_table, filters, targets = verify_input(data_table, filters, targets)

    for f in filters:
        for t in targets:
            chip_tup = extract_data(data_table, f, t)
            outliers = compute_residuals(chip_tup, f, t,
                                         show_residuals=show_residuals,
                                         save_residuals=save_residuals)

            for ind, t in enumerate(data_table['expstart']):
                tt = Time(t, format='mjd', scale='utc')
                tt = tt.to_value('decimalyear', subfmt='float')
                for o in outliers:
                    if tt == o[1]:
                        row = data_table[ind]
                        outlier_table.add_row(row)

            chip_table = clip_chip_data(chip_tup, outliers)

            x, y, m = norm_by_earliest_proposal(chip_table)

            chip_table['expstart_decimalyear'] = x
            chip_table['norm_cr_fpm'] = y
            chip_table['first_prop_mean'] = m

            clean_table = vstack([clean_table, chip_table])

    clean_table.write(f'{LOG_DIR}/clean_table.csv',
                      format='csv')
    outlier_table.write(f'{LOG_DIR}/outlier_table.csv',
                        format='csv', overwrite=True)

    return clean_table, outlier_table

def plot_cr_by_filter(clean_table, filter_list,
                      show_plots, save_plots):
    """Plot chip sensitivity by filter.

    Plots clipped and normalized photometry data, with each
    UVIS chip on a separate subplot.

    Parameters
    ----------
    clean_table : `astropy.table.table.Table`
        Data table that has been cleaned (outliers
        have been removed) and normalized.
    filter_list : list
        List of filters to plot.
    show_plots : bool
        Whether to show plots of percent change over time.
    save_plots : bool
        Whether to save plots of percent change over time.
    """

    targ_list = list(set(clean_table['targname']))
    targ_list.sort()
    targ_markers = ['o', '<', '*']
    targ_style = ['dashed', 'dotted', 'dashdot']

    chip_list = ['A', 'C']
    chip_col = ['#FE6100', '#648FFF']

    for f in filter_list:
        fig, ax = plt.subplots(2, figsize=(12,10), sharey=True)

        filter_data = clean_table[clean_table['filter'] == f]

        for c, chip in enumerate(chip_list):
            chip_data = filter_data[filter_data['ccdamp'] == chip]

            if chip == 'A':
                uvis_name = 'UVIS1'
            else:
                uvis_name = 'UVIS2'

            for t, targ in enumerate(targ_list):
                targ_data = chip_data[chip_data['targname'] == targ]

                x_data = targ_data['expstart_decimalyear']
                y_data = targ_data['norm_cr_fpm']

                lsr, bf_y = make_bf_line(x_data, y_data)

                ax[c].plot(x_data, bf_y, color=chip_col[c],
                           linewidth=1, linestyle=targ_style[t],
                           marker=targ_markers[t], markersize=0.25,
                           label=f'{uvis_name} ({targ}): ' \
                                 'm = {:.3f} '.format(lsr[0])+ \
                                 u'\u00b1'+' {:.3f}% / year'.format(lsr[4]))

                ax[c].scatter(x_data, y_data, s=15, marker=targ_markers[t],
                            color=chip_col[c], alpha=0.75)

            chip_x = chip_data['expstart_decimalyear']
            chip_y = chip_data['norm_cr_fpm']

            chip_lsr, chip_bf_y = make_bf_line(chip_x, chip_y)

            ax[c].plot(chip_x, chip_bf_y, linestyle='solid',
                      linewidth=2, alpha=1, color=chip_col[c],
                      label=f'{uvis_name}: '+ \
                            'm = {:.3f} '.format(chip_lsr[0])+ \
                            u'\u00b1'+' {:.3f}% / year'.format(chip_lsr[4]))

            ax[c].legend(loc='best', fontsize=10, numpoints=1,
                         markerfirst=True, markerscale=15,
                         facecolor='w', edgecolor='k')
            ax[c].set_xlabel('Observation Date', fontsize=10)
            ax[c].set_ylabel('% change in countrate', fontsize=10)
            ax[c].set_title(uvis_name, fontsize=12)


        if show_plots == True:
            plt.suptitle(f'WFC3/UVIS/{f} Spatial Scan\n' \
                         'Temporal Photometric Variability', fontsize=12)
            if save_plots == True:
                plot_dir = os.path.join(LOG_DIR, 'plots/')
                if not os.path.isdir(plot_dir):
                    os.mkdir(plot_dir)
                plt.savefig(f'{plot_dir}crchange_{f}.png')
            plt.show()
        else:
            plt.close()


def plot_cr_all_filters(clean_table, filter_list,
                        show_plots, save_plots):
    """Plot chip sensitivity.

    Plots clipped and normalized photometry data, with each
    UVIS chip on a separate subplot.

    Parameters
    ----------
    clean_table : `astropy.table.table.Table`
        Data table that has been cleaned (outliers
        have been removed) and normalized.
    filter_list : list
        List of filters to plot.
    show_plots : bool
        Whether to show plots of percent change over time.
    save_plots : bool
        Whether to save plots of percent change over time.
    """

    targ_list = list(set(clean_table['targname']))
    targ_list.sort()
    targ_markers = ['o', '<', '*']
    targ_style = ['dashed', 'dotted', 'dashdot']

    chip_list = ['A', 'C']

    f_cols = ['#332288', '#117733', '#44AA99', '#88CCEE',
              '#DDCC77', '#CC6677', '#AA4499', '#882255']

    fig, ax = plt.subplots(2, figsize=(12,10), sharey=True)

    for c, chip in enumerate(chip_list):
        chip_data = clean_table[clean_table['ccdamp'] == chip]
        if chip == 'A':
            uvis_name = 'UVIS1'
        else:
            uvis_name = 'UVIS2'

        filt_legend = []

        for f, filt in enumerate(filter_list):
            filter_data = chip_data[chip_data['filter'] == filt]

            for t, targ in enumerate(targ_list):
                targ_data = filter_data[filter_data['targname'] == targ]

                x_data = targ_data['expstart_decimalyear']
                y_data = targ_data['norm_cr_fpm']

                ax[c].scatter(targ_data['expstart_decimalyear'],
                              targ_data['norm_cr_fpm'],
                              s=15, marker=targ_markers[t],
                              alpha=0.75, c=f_cols[f])

            ax[c].set_xlabel('Observation Date', fontsize=10)
            ax[c].set_ylabel('% change', fontsize=10)
            ax[c].set_title(uvis_name, fontsize=12)

            filt_leg_elem = Line2D([0], [0], color=f_cols[f], label=filt)
            filt_legend.append(filt_leg_elem)

        targ_legend = []
        for t, targ in enumerate(targ_list):
            targ_legend.append(Line2D([0], [0], marker=targ_markers[t],
                                      color='w', label=targ, markersize=10,
                                      markerfacecolor='k'))

        leg1 = ax[c].legend(handles=filt_legend, loc=3, ncol=2, fontsize=8,
                            facecolor='w', edgecolor='k')
        ax[c].add_artist(leg1)
        leg2 = ax[c].legend(handles=targ_legend, loc=1, fontsize=8,
                            facecolor='w', edgecolor='k')
        ax[c].add_artist(leg2)

    plt.suptitle('WFC3/UVIS Spatial Scan\nTemporal Photometric Variability',
                 fontsize=14)
    fig.tight_layout()

    if show_plots == True:
        if save_plots == True:
            plot_dir = os.path.join(LOG_DIR, 'plots/')
            if not os.path.isdir(plot_dir):
                os.mkdir(plot_dir)
            plt.savefig(f'{plot_dir}crchange_allfilters.png')
        plt.show()
    else:
        plt.close()

def make_slope_table(clean_table):
    """Makes an Astropy table of slope and error data.

    Parameters
    ----------
    clean_table : `astropy.table.table.Table`
        Data table that has been cleaned (outliers
        have been removed) and normalized.

    Returns
    -------
    slope_table : `astropy.table.table.Table`
        Data table with a row corresponding to each filter,
        target, and chip configuration, and containing the
        slope and slope error of the linear regression of
        that data subset.
    year_span : str
        String representation of the years that these
        observations spanned.
    """
    slope_table = Table(names=('filter', 'targname', 'ccdamp',
                               'slope', 'slope_err'),
                        dtype=('str', 'str', 'str', 'float', 'float'))

    filters = list(set(clean_table['filter']))
    filters.sort()
    targets = list(set(clean_table['targname']))
    targets.sort()
    chips = ['A', 'C']

    for filt in filters:
        filt_set = clean_table[clean_table['filter'] == filt]
        for targ in targets:
            targ_set = filt_set[filt_set['targname'] == targ]
            for chip in chips:
                chip_set = targ_set[targ_set['ccdamp'] == chip]
                try:
                    lsr = linregress(chip_set['expstart_decimalyear'],
                                     chip_set['norm_cr_fpm'])
                    slope_table.add_row([filt, targ, chip, lsr[0], lsr[4]])
                except ValueError:
                    pass

    slope_table.write(f'{LOG_DIR}/slope_table.csv',
                      format='csv', overwrite=True)

    start_yr = int(min(clean_table['expstart_decimalyear']))
    stop_yr = int(max(clean_table['expstart_decimalyear']))

    year_span = str(start_yr)+'-'+str(stop_yr)

    return slope_table, year_span

def plot_slope_by_filter(slope_table, year_span, save_plot=True):
    """Plot slope as a function of filter.

    Parameters
    ----------
    slope_table : `astropy.table.table.Table`
        Data table with a row corresponding to each filter,
        target, and chip configuration, and containing the
        slope and slope error of the linear regression of
        that data subset.
    year_span : str
        String representation of the years that these
        observations spanned.
    save_plot : bool
        Whether to save the resulting plot. Default is True.
    """
    targets = list(set(slope_table['targname']))
    targets.sort()
    targ_markers = ['o', '<', '*']

    filters = list(set(slope_table['filter']))
    filters.sort()

    chips = [('A', 'UVIS1'), ('C', 'UVIS2')]

    fig, ax = plt.subplots(2, 1, figsize=(8, 8))

    plt.rc('xtick', labelsize='small')
    plt.rc('ytick', labelsize='small')

    for t, targ in enumerate(targets):
        targ_subset = slope_table[slope_table['targname'] == targ]
        for c, chip in enumerate(chips):
            chip_subset = targ_subset[targ_subset['ccdamp'] == chip[0]]
            chip_subset.sort(['filter'])

            ax[c].scatter(chip_subset['filter'], chip_subset['slope'],
                          s=10, alpha=0.5, label=targ, marker=targ_markers[t])
            ax[c].errorbar(chip_subset['filter'], chip_subset['slope'],
                           chip_subset['slope_err'], fmt='o', alpha=0.5,
                           capsize=3)
            ax[c].set_title(chip[1], fontsize=10)

            ax[c].set_xlabel('Filter', fontsize=8)
            ax[c].set_ylabel(f'% change/year\n({year_span})', fontsize=8)
            ax[c].legend(loc=4, fontsize=8, facecolor='w', edgecolor='k')

    fig.tight_layout()
    plt.suptitle('WFC3/UVIS Spatial Scan\n'+ \
                 'Chip Sensitivity by Filter', fontsize=12)
    plt.subplots_adjust(top=0.9)

    if save_plot == True:
        plot_dir = os.path.join(LOG_DIR, 'plots/')
        if not os.path.isdir(plot_dir):
            os.mkdir(plot_dir)
        plt.savefig(f'{plot_dir}filter_slope.png', Transparent=False, dpi=300)

    plt.show()

def plot_slope_by_pivotwl(slope_table, year_span, save_plot=True):
    """Plot slope as a function of filter pivot wavelength.

    Parameters
    ----------
    slope_table : `astropy.table.table.Table`
        Data table with a row corresponding to each filter,
        target, and chip configuration, and containing the
        slope and slope error of the linear regression of
        that data subset.
    year_span : str
        String representation of the years that these
        observations spanned.
    save_plot : bool
        Whether to save the resulting plot. Default is True.
    """
    pivot = {'F218W': 222.4, 'F225W': 235.9, 'F275W': 270.4, 'F336W': 335.5,
             'F390W': 392.1, 'F438W': 432.5, 'F475W': 477.3, 'F555W': 530.8,
             'F606W': 588.7, 'F625W': 624.2, 'F775W': 764.7, 'F814W': 802.4}

    targets = list(set(slope_table['targname']))
    targets.sort
    filters = list(set(slope_table['filter']))
    filters.sort()
    chips = [('A', 'UVIS1'), ('C', 'UVIS2')]

    fig, ax = plt.subplots(2, 1, figsize=(8, 8))

    plt.rc('xtick', labelsize='small')
    plt.rc('ytick', labelsize='small')

    for targ in targets:
        targ_subset = slope_table[slope_table['targname'] == targ]
        for c, chip in enumerate(chips):
            chip_subset = targ_subset[targ_subset['ccdamp'] == chip[0]]
            chip_subset.sort(['filter'])
            pivot_x = [pivot[f] for f in chip_subset['filter']]
            ax[c].scatter(pivot_x, chip_subset['slope'],
                          s=10, alpha=0.5, label=targ)
            ax[c].errorbar(pivot_x, chip_subset['slope'],
                           chip_subset['slope_err'], fmt='o', alpha=0.5,
                           capsize=3)

            ax[c].set_title(chip[1], fontsize=10)

            ax[c].set_xlabel('Wavelength (nm)', fontsize=8)
            ax[c].set_ylabel(f'% change/year\n({year_span})', fontsize=8)
            ax[c].set_ylim(-0.4, 0.05)
            ax[c].legend(loc=4, fontsize=8, facecolor='w', edgecolor='k')

    fig.tight_layout()
    plt.suptitle('WFC3/UVIS Spatial Scan\n'+ \
                 'Chip Sensitivity by Filter Pivot Wavelength',
                 fontsize=10)
    plt.subplots_adjust(top=0.9)
    if save_plot == True:
        plot_dir = os.path.join(LOG_DIR, 'plots/')
        if not os.path.isdir(plot_dir):
            os.mkdir(plot_dir)
        plt.savefig(f'{plot_dir}pivot_slope.png', Transparent=False, dpi=300)

    plt.show()
