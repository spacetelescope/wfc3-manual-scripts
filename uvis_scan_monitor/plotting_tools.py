"""
TK work on!
"""

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
