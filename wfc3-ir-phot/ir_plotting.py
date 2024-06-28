"""
Functions to create plots for the IR standard star staring mode photometry monitor.

Author
------
    Mariarosa Marinelli, 2023

Functions
---------
    set_plt_rcparams()
        Helper function to update Matplotlib settings.
    plot_flt_sources()
        Make and save source detection plots, plotting the PAM-corrected
        data of an observation (log-normalized) and the locations of all
        detected sources, color-coded by if they are rejected based on
        proximity to the detector edge, rejected based on the photometry
        compared to the simulated count rate of the observation source, or
        if it is deemed to be the source. 
"""
import os
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from ir_logging import display_message


def set_plt_rcparams():
    """Helper function to update Matplotlib settings.
    """
    plt.rcParams.update({'axes.facecolor': 'white',
                         'figure.facecolor': 'white',
                         'axes.edgecolor': 'black'})
    plt.rcParams.update({"font.family": "serif",
                         "font.serif": "cmr10",
                         "font.size": 14 })

    matplotlib.rcParams['axes.linewidth'] = 1


def plot_flt_sources(obs_batch, props_tbl, use_source, cr_pd, verbose, log, plot_dir):
    """Make and save source detection plots.

    This function plots the PAM-corrected data of an
    observation (log-normalized) and the locations of
    all detected sources, color-coding as follows.
        red:
            Source(s) too close to the edge of
            the detector.
        yellow:
            Source(s) with count rate more than
            `cr_pd` percent different
            compared to synthetic count rate.
        cyan:
            Source used for photometry.

    The plot titles (`title`) and file names
    (`plot_name`) are derived from the `exposure_file`
    attribute.
        `exposure_file`:
            /data/<proposal>/<targname>/<filter>/<rootname>_flt.fits
        `title`:
            /<proposal>/<targname>/<filter>/<rootname>_flt.fits
        `plot_name` :
            <proposal>_<targname>_<filter>_<rootname>_flt.jpg

    Parameters
    ----------
    obs_batch : `ObsBatch`
        Staring mode observation object.
    props_tbl : `astropy.table.table.Table`
        Table with identified sources' properties.
    use_source : `astropy.table.row.Row`
        The row corresponding to the identified source
        that will be used for photometry.
    cr_pd : float or int
        Threshold for percent difference between source
        count rate and synthetic count rate.
    verbose : Boolean
        Whether to print the message.
    log : Boolean
        Whether to log the message.
    plot_dir : str or path-like

    Notes
    ----
      - Even if no source is validated for use in
        photometry (if `use_source` is `None`), a
        source detection plot will still be created.
        This is the intended behavior, since the plot
        can then be used to diagnose any issues with
        source detection and selection.
      - TK: Figure out a way to improve the legend
        location assignation.
    """
    set_plt_rcparams()

    # Make title, directory names, file names, etc.
    title = obs_batch.exposure_file.split('/data')[-1]
    plot_subdir = os.path.join(plot_dir, 'source_detection')

    if not os.path.exists(plot_subdir):
        os.mkdir(plot_subdir)
        display_message(verbose=verbose,
                        log=log,
                        log_type='info',
                        message=f'Made new directory at {plot_subdir}')

    plot_name = f"{title.split('.fits')[0][1:].replace('/', '-')}.jpg"
    plot_path = os.path.join(plot_subdir, plot_name)

    # Set up plot itself.
    fig, ax = plt.subplots(figsize=(7,7))
    ax.set_title(title)
    # Plot image data (corrected for distortion).
    ax.imshow(obs_batch.data_corr, norm=LogNorm(), origin='lower',
              cmap='Greys_r', zorder=0)

    xticks = ax.get_xticks()
    yticks = ax.get_yticks()
    xticklabels = ax.get_xticklabels()
    yticklabels = ax.get_yticklabels()

    if len(props_tbl) > 1:
        nope_edge = props_tbl[props_tbl['within_edge'] == 'y']
        if len(nope_edge) > 0:
            ax.scatter(nope_edge['xcentroid'], nope_edge['ycentroid'],
                       marker='o', facecolors='none', edgecolors='#FBD036',
                       lw=3, s=300, label='rejected: proximity to edge')

        if use_source is not None:
            on_frame = props_tbl[props_tbl['within_edge'] == 'n']
            nope_cr = on_frame[on_frame['xcentroid'] != use_source['xcentroid']]
        else:
            nope_cr = on_frame[on_frame['within_edge'] == 'n']

        if len(nope_cr) > 0:
            ax.scatter(nope_cr['xcentroid'], nope_cr['ycentroid'],
               marker='o', facecolors='none', edgecolors='#ED553B',
               lw=3, s=300,label=f'rejected: source flux')

    if use_source is not None:
        ax.scatter(use_source['xcentroid'], use_source['ycentroid'],
                   marker='o', facecolors='none', edgecolors='#20639B', lw=3,
                   zorder=10, s=300,
                   label=f'identified: {obs_batch.targname}')

    ax.set_xticks(xticks[1:-1])
    ax.set_yticks(yticks[1:-1])
    ax.set_xticklabels(xticklabels[1:-1])
    ax.set_yticklabels(yticklabels[1:-1])
    ax.legend(loc='best', fontsize=10, markerscale=0.5)
    fig.tight_layout()

    plt.savefig(plot_path, format='jpg', dpi=200)
    plt.close()

    messages = ['  Source detection/selection plot saved to:',
                f'{" "*4} {plot_path}']

    for message in messages:
        display_message(verbose=verbose,
                        log=log,
                        log_type='info',
                        message=message)
