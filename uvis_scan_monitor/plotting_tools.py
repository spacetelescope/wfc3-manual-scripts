"""
Authors: Mariarosa Marinelli, Anne O'Connor
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import datetime as dt
import plotly.express as px
from astropy.table import Table, vstack
from astropy.time import Time
from glob import glob
from scipy.stats import linregress

def plot_normcr(data, filt, uvis='both', save=False, save_dir=None):
    if uvis == 'both':
        amps = ['A', 'C']
        uvis_numbers = [str(1), str(2)]
        plot_titles = ['WFC3/UVIS 1', 'WFC3/UVIS 2']

        fig, ax = plt.subplots(2,1, figsize=(12,12), sharex=True)
    else:
        plot_titles = ['WFC3/UVIS {}'.format(uvis)]

        if uvis==1:
            amps = ['A']
            uvis_numbers = [str(1)]
        else:
            amps = ['C']
            uvis_numbers = [str(2)]

        fig, ax = plt.subplots(1,1, figsize=(12,6))

    for i, amp in enumerate(amps):
        non_outliers = data[data['outlier'] == 'False']
        amp_data = non_outliers[non_outliers['ccdamp'] == amp]

        targs = list(set(amp_data['targname']))
        targs.sort()
        targ_labels = ['GD153', 'GRW70']

        for t, targ in enumerate(targs):
            targ_data = amp_data[amp_data['targname'] == targ]

            row = get_slope_row(filt, amp, targ)

            ax[i].scatter(targ_data['expstart_decimalyear'], targ_data['norm_44_268'],
                          label=r'{} ($m = {{{:.3f}}} \pm {{{:.3f}}}$ \%)'.format(targ_labels[t],
                                                                                  row['slope']*100,
                                                                                  row['sterr']*100),
                          marker=targ_markers[t], s=(t+1)*30,
                          c=colors[t], alpha=0.7)
            ax[i].plot(targ_data['expstart_decimalyear'], targ_data['expected norm_44_268'],
                       c=colors[t], alpha=0.5)

        row = get_slope_row(filt, amp, targ='both')
        subscript = '{},U{}'.format(filt, uvis_numbers[i])
        ax[i].text(2021, 0.9875,r'$m_{{{}}} = {{{:.3}}} \pm {{{:.3f}}}$ \%'.format(subscript,
                                                                                row['slope']*100,
                                                                                row['sterr']*100),
                   ha='center', va='center', fontsize=20)

        ax[i].set_title('{} {} - Scan Mode'.format(plot_titles[i], filt), fontsize=24)
        ax[i].set_ylabel('Normalized Flux', fontsize=20)
        ax[i].set_ylim(0.985,1.01)
        ax[i].set_xlim(2016.95,2022.25)
        ax[i].legend(loc=3, edgecolor='k', facecolor='w')
        ax[i].grid(zorder=0, c='gray', alpha=0.5)

    ax[-1].set_xlabel('Date', fontsize=20)
    if save != False:
        if save_dir != None:
            if not save_dir.endswith('/'):
                save_dir = save_dir+'/'
            filename = '{}{}_uvis{}_normcr.jpg'.format(save_dir, filt, uvis)
            plt.savefig(filename, dpi=250)

    plt.show()

# def plot_scan_phot_over_time(filters, output_dirname):
# 
#     fig_list=[]
#     fig_name_list=[]
#     for filter in filters:
#         data_csvs = sorted(glob(f'{output_dirname}/*_G*_{filter}.csv'))
#         data_tbl = pd.DataFrame()
#         for data_csv in data_csvs:
#             print(data_csv)
#             try:
#                 tbl = pd.read_csv(data_csv)
#             #print(tbl)
#             except:
#                 print(f'something happened with {data_csv}')
#                 tbl=pd.DataFrame()
#             #tbl
#             if len(tbl) > 0:
#                 data_tbl = pd.concat([data_tbl, tbl])
#             #print(data_csv.split('/')[-1].split('_')[-1].split('.')[0], len# (tbl))
#             #print(data_tbl.colnames)
#         for amp in ['A', 'C']:
#             filt_data_only = data_tbl.query(f"ccdamp == '{amp}'")
# 
#             times=Time(filt_data_only['expstart'], format='mjd').to_datetime()
#             
#             filt_data_only['date'] = [dt.date() for dt in times]
#             fig=plt.figure()
#             for targ in filt_data_only['targname'].unique():
#                 print(targ)
#                 targ_data_only=filt_data_only.query(f"targname == '{targ}'")
#                 plt.scatter(targ_data_only['date'], 
#                             targ_data_only['fcr_phot']/np.mean(targ_data_only# ['fcr_phot']), label=targ)
#                 plt.legend()
#             #plt.axhline(1, ls='dashed')
#             fig_name=f'{filter}_scan_phot_amp_{amp}.png'
#             plt.title(f'{filter} Scan Photometry - Amp {amp}')
#             plt.xlabel('Date')
#             plt.ylabel('Normalized Flux')
#             plt.ylim(0.99,1.01)
#             fig_list.append(fig)
#             fig_name_list.append(fig_name)
#     return fig_list, fig_name_list


def plot_scan_phot_over_time_plotly(args, dirs):
    '''
     Create plots of UVIS scan over time per filter and write a copy to Quicklook manual outputs. When the write arg is set to True, these files will overwrite the exisiting plots in the QL manual outputs directory. 

     Parameters
    ----------
    args : `argparse.Namespace` or `InteractiveArgs`
        Arguments.
    dirs : dict
        Dictionary of directories.
    '''


    filters=args.filters
    output_dirname=dirs["output_dir"]
    fig_list = []
    fig_name_list = []

    for filter in filters:
        data_csvs = sorted(glob(f'{output_dirname}/*_{filter}.csv'))
        data_tbl = pd.DataFrame()

        for data_csv in data_csvs:
            print(data_csv)
            try:
                tbl = pd.read_csv(data_csv)
            except Exception as e:
                print(f'Something happened with {data_csv}: {e}')
                tbl = pd.DataFrame()
            if len(tbl) > 0:
                data_tbl = pd.concat([data_tbl, tbl], ignore_index=True)

        # Exclude P330E for filters starting with F2
        if filter.startswith('F2'):
            before = len(data_tbl)
            data_tbl = data_tbl.query("targname != 'P330E'")
            after = len(data_tbl)
            print(f"Filtered out P330E rows for {filter}: {before - after} rows removed.")

        for amp in ['A', 'C']:
            filt_data_only = data_tbl.query(f"ccdamp == '{amp}'")
            if filt_data_only.empty:
                continue

            times = Time(filt_data_only['expstart'], format='mjd').to_datetime()
            filt_data_only['date'] = [t.date() for t in times]

            fig = go.Figure()

            colors = px.colors.qualitative.Plotly

            for i, targ in enumerate(filt_data_only['targname'].unique()):
                targ_data_only = filt_data_only.query(f"targname == '{targ}'")
                norm_flux = targ_data_only['fcr_phot'] / np.mean(targ_data_only['fcr_phot'])
                x_dates = np.array(targ_data_only['date'])
                x_days = np.array([t.toordinal() for t in x_dates])
                y = np.array(norm_flux)
                color = colors[i % len(colors)]

                slope_label = ""
                slope_per_year = np.nan
                std_err_per_year = np.nan
                if len(x_days) > 1:
                    slope, intercept, r_value, p_value, std_err = linregress(x_days, y)
                    slope_per_year = slope * 365.25 * 100  # % per year
                    std_err_per_year = std_err * 365.25 * 100
                    slope_label = f"{slope_per_year:+.2f} ± {std_err_per_year:.2f} %/yr"
                    fit_y = intercept + slope * x_days
                else:
                    fit_y = np.full_like(y, np.mean(y))

                # Scatter points
                fig.add_trace(go.Scatter(
                    x=x_dates,
                    y=y,
                    mode='markers',
                    name=f"{targ} ({slope_label})" if slope_label else targ,
                    legendgroup=targ,
                    marker=dict(size=6, color=color),
                    showlegend=True,
                ))

                # Regression line (same color, grouped)
                fig.add_trace(go.Scatter(
                    x=x_dates,
                    y=fit_y,
                    mode='lines',
                    name=f"{targ} fit",
                    legendgroup=targ,
                    showlegend=False,  # hide duplicate entry
                    line=dict(color=color, dash='dot', width=2)
                ))

            # Reference line at 1
            fig.add_hline(y=1, line_dash='dash', line_color='gray')

            fig.update_layout(
                title=f'{filter} Scan Photometry - Amp {amp}',
                xaxis_title='Date',
                yaxis_title='Normalized Flux',
                yaxis=dict(range=[0.99, 1.01]),
                template='plotly_white',
                legend=dict(
                    title='Target (Slope)',
                    bgcolor='rgba(255,255,255,0.8)',
                    bordercolor='lightgray',
                    borderwidth=1,
                    font=dict(size=10)
                )
            )

            fig_name = f'{filter}_scan_phot_amp_{amp}.html'
            fig_list.append(fig)
            fig_name_list.append(fig_name)
            path_to_ql=os.path.join('/grp/hst/wfc3a/manual_outputs/uvis_scan_photometry/display_outputs/', fig_name)
            fig.write_html(path_to_ql, full_html=False)

    return fig_list, fig_name_list


