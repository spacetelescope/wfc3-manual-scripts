#!/usr/bin/env python

from glob import glob
import numpy as np
import os

from astropy import units as u
from astropy.table import Table
import stsynphot as stsyn
import synphot as syn


monitor_dir = '/grp/hst/wfc3v/wfc3photom/data/uvis_scan_monitor'

print('Only using uvis1 in synthetic bandpass.')


def make_syn_spec(target, use_latest=True):
    """
    Creates a synthetic spectrum for the target of interest.

    Parameters
    ----------
    target : str
        String representation of target name.

    Returns
    -------
    syn_spec : 'synphot.spectrum.SourceSpectrum'
        Synthetic spectrum generated from CALSPEC model of target.
    """
    files = {'GD153': 'gd153_stiswfcnic_*.fits',
             'GRW70': 'grw_70d5824_stiswfcnic_*.fits'}

    calspec_dir = os.path.join(os.environ['PYSYN_CDBS'], 'calspec')

    if use_latest:
        matching_files = sorted(glob(os.path.join(calspec_dir, files[target])))
        file = matching_files[-1]

    else:
        file = os.path.join(calspec_dir, files[target].replace('*', '003'))

    syn_spec = syn.SourceSpectrum.from_file(os.path.join(calspec_dir, file))

    return syn_spec, file.split('/')[]


def lookup_ee(uvis_name, filt, psf_type, ee_dir, ap_dim):
    """
    Looks up previously calculated enrectangled energy
    aperture correction value.

    Parameters
    ----------
    uvis_name : str
    filt : str
    psf_type : str

    Returns
    -------
    ee_corr : 'astropy.table.column.Column'
        Calculated enrectangled energy aperture correction.
    """

    ee = Table.read(f'{ee_dir}/{psf_type}_{ap_dim[0]}_{ap_dim[1]}.csv')

    slice = ee[ee['filter'] == filt]

    if psf_type == 'simple':
        slice = slice[slice['uvis'] == uvis_name]

#    uvis_slice = ee[ee['uvis'] == uvis_name]
#    filt_slice = uvis_slice[uvis_slice['filter'] == filt]

    if len(slice) != 1:
        print("Uh oh, that's enough slices!!")

    ee_corr = slice['ee_phot'][0]

    return ee_corr

def make_syn_obs(syn_spec, obs, uvis_name, psf_type, ee_dir, ap_dim, timecorr=True):
    """
    Creates a synthetic observation using the synthetic
    spectrum and the aperture correction calculated with
    scan_ee.py.

    Parameters
    ----------
    syn_spec : 'synphot.spectrum.SourceSpectrum'
        Synthetic spectrum generated from CALSPEC model of target.
    obs : 'astropy.table.row.Row'
        Row corresponding to a single observation.
    psf_type : str
        The type of PSF used to calculate the aperture
        corrections to the enrectangled energy. Either
        simple or blended
    timecorr : bool
        Whether to calculate with time corrections or not.

    Returns
    -------
    syn_obs : 'synphot.observation.Observation'

    """

    filt = obs['filter']

    aperture_corr = lookup_ee(uvis_name, filt, psf_type, ee_dir, ap_dim)

    # changed to uvis1 to resolve 0.25% offset between methods
    if timecorr:
        mjd = obs['expstart']
        bandpass = stsyn.band(f'wfc3,uvis1,{filt},mjd#{mjd}')
    else:
        bandpass = stsyn.band(f'wfc3,uvis1,{filt}')

    syn_obs = syn.Observation(syn_spec*aperture_corr, bandpass,
                              binset=bandpass.binset)

    return syn_obs

def get_phtratio(uvis_name, obs):
    """
    Parameters
    ----------
    uvis_name : str

    obs : 'astropy.table.row.Row'
        Row corresponding to a single observation.
    """
    if uvis_name == 'uvis1':
        phtratio = 1.0
    else:
        try:
            phtratio = obs['phtratio']
        except KeyError as key_err:
            if str(key_err) == 'phtratio':
                phtratio = obs['PHTRATIO']

    return phtratio

def calculate_syncr(syn_obs, phtratio):
    """
    Parameters
    ----------
    syn_obs : 'synphot.observation.Observation'

    phtratio : float
        Value of PHTRATIO; 1.0 for UVIS 1 to prevent scaling.

    Returns
    -------
    syn_cr : 'astropy.units.quantity.Quantity'
    """
    syn_cr = syn_obs.countrate(stsyn.conf.area)*phtratio

    return syn_cr




# def set_paths(trial_dir_name):
#     data_dir = os.path.join(monitor_dir, trial_dir_name)
#     if not os.path.exists(data_dir):
#         print(f'{data_dir} does not exist. Will not be able to'\
#               'calculate observed to synthetic ratios...')
#         data_dir = None
#
#     ee_dir = os.path.join(monitor_dir, 'ee')
#     synphot_dir = os.path.join(monitor_dir, 'synphot')
#
#     for directory in [ee_dir, synphot_dir]:
#         if not os.path.exists(directory):
#             os.mkdir(directory)
#             print(f'Made new directory at {directory}')
#
#     return data_dir, ee_dir, synphot_dir

if __name__ == '__main__':
    filters = ['F218W', 'F225W', 'F275W', 'F336W', 'F438W', 'F606W', 'F814W']
    jfilters = ['F275W', 'F336W', 'F438W', 'F606W', 'F814W']

    targets = ['GD153', 'GRW70']
    uvis_names = ['uvis1', 'uvis2']

#    psf_type = 'ToyPSF'
    #psf_type = 'JayPSF'

    ap_dim = (44, 268)

    synphot_dir = '/grp/hst/wfc3v/wfc3photom/data/uvis_scan_monitor/synphot'
    ee_dir = os.path.join(synphot_dir, '2023_03_14_test1/ee')
    data_dir = '/grp/hst/wfc3v/wfc3photom/data/uvis_scan_monitor/2023_03_06_test1'

#    if len(sys.argv) == 1:
#        print('No directory specified. Please specify data directory.')

#    else:
#        data_dir, ee_dir, synphot_dir = set_paths(sys.argv[1])

    # generalize this later:
    #syn_column_name = f'{psf_type}_countrate_{ap_dim[0]}_{ap_dim[1]}'
#    for psf_type in ["simple", "blended"]:
    gd153_spec, gd153_file = make_syn_spec('GD153', use_latest=True)
    grw70_spec, grw70_file = make_syn_spec('GRW70', use_latest=True)

    for psf_type in ["blended"]:
        if psf_type == "blended":
            filters = jfilters
        print(psf_type)
        syn_column_name = f'{psf_type}_fcr_phot'
        timecorr = True

    #    ee_dir = '/Users/mmarinelli/work/WFC3/uvis_scan_monitor/output/ee/'
        all_data = Table.read(os.path.join(data_dir, "output/all_data.csv"), format='csv')

        for filt in filters:
            syn_column_vals = []
            filt_tbl = all_data[all_data['filter'] == filt]
    #        filt_tbl = Table.read(f'{output_dir}corrected/{psf_type}_{filt}_obssyn.csv',
    #                              format='csv')
    #        filt_csvs = glob(os.path.join(data_dir, "output", f"*{filt}.csv"))
    #        filt_tbl = Table()
    #        for filt_csv in filt_csvs:
    #            tbl = Table.read(filt_csv, format='csv')
    #            filt_tbl = vstack([filt_tbl, tbl])

            # clean up table so we can do group operations
            filt_tbl['targname'][np.where(filt_tbl['targname'] != 'GD153')] = 'GRW70'
            filt_tbl.sort('targname')

            grouped_tbl = filt_tbl.group_by('targname')

            for i in range(0, 2):
                print(filt, 'target', i+1)
                targ_group = grouped_tbl.groups.keys[i]
                targ = targ_group['targname']

                for obs in grouped_tbl.groups[i]:
                    if obs['outlier'] == 'False':
                        if (obs['ccdamp'] == 'A') and (psf_type == 'simple'):
                            uvis_name = 'uvis1'
                        else:
                            uvis_name = 'uvis2'

                        if obs['targname'] == 'GD153':
                            syn_spec = gd153_spec
                        else:
                            syn_spec = grw70_spec

                        #phtratio = get_phtratio(uvis_name, obs)

                        syn_obs = make_syn_obs(syn_spec, obs, uvis_name, psf_type,
                                               ee_dir, ap_dim, timecorr=timecorr)

                        syn_cr = calculate_syncr(syn_obs, 1.)

                        syn_column_vals.append(syn_cr)  #/(1.0*(u.ct/u.s)))
                    else:
                        syn_column_vals.append(-9999)

            grouped_tbl[syn_column_name] = syn_column_vals #* (u.ct / u.s)

            grouped_tbl.write(f'{synphot_dir}/{psf_type}_{filt}_obssyn.csv',
                                  format='csv', overwrite=True)

            print(f'Table for {filt} saved.')
