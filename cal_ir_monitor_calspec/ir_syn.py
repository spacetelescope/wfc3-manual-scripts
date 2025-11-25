"""
Functions and a class to enable time-independent synthetic photometry for
the IR staring mode standard star pipeline.

Functions
---------
make_synthetic_spectrum(targname)
    Make synthetic spectrum for specified target. Compares target name
    to catalog of names and CALSPEC files. If the target name is not found
    or resolved, displays an error message and returns `None`.
make_syn_targets(filepaths_batches)
    Makes dictionary of synthetic targets & photometry.

    Methods
    -------
        make_bandpass()
            Simulate bandpass by filter.
        make_observation()
            Simulate observation by filter.
        get_phot_table()
            Simulates photometry for specified filters. Calls
            class methods `make_bandpass()` and `make_observation()`.

Classes
-------
SynTarget()
    Simulated IR staring mode standards class.
"""
import os
from astropy.table import Table
import numpy as np
import stsynphot
import synphot

from ir_config import CONFIG
from ir_toolbox import resolve_targnames


class SynTarget:
    """Simulated IR staring mode standards class.

    A class to represent simulated targets for IR staring
    mode observations. Requires two attributes to
    initialize, and has three methods to enable reducing,
    analyzing, and compiling data.

    Attributes
    ----------
    targname : str
        Target name. Will attempt to resolve name is non-
        standard version is supplied.

    Methods
    -------
    make_bandpass(filt, aper_arcsec, time_dep=False)
        Simulate bandpass by filter.
    make_observation(filt)
        Simulate observation by filter.
    get_phot_table(filters, aper_arcsec)
        Simulates photometry for specified filters. Calls
        `make_bandpass()` and `make_observation()`.
    """
    def __init__(self, targname):
        self.targname = resolve_targnames(targname, simplify=True)
        self.bandpasses = {}
        self.observations = {}
        self.spectrum = self.make_synthetic_spectrum()
        if self.spectrum is None:
            print(f'  Unable to initialize SynTarget for {self.targname}.')
        else:
            print(f'  Initialized SynTarget for {self.targname}')

        self.phot_table = None


    def make_synthetic_spectrum(self):
        """Make synthetic spectrum for specified target.

        Compares target name to catalog of names and CALSPEC
        files. If the target name is not found or resolved,
        displays an error message and returns `None`.

        Returns
        -------
        spectrum : `synphot.SourceSpectrum` or None
            If the specified target isn't in the `star_catalog`
            dictionary, will return None.
        """
        calspec_filename = CONFIG['calspec_catalog'][self.targname]

        try:
            spectrum_path = os.path.join(CONFIG['calspec_dir'],
                                         calspec_filename)
            spectrum = synphot.SourceSpectrum.from_file(spectrum_path)

        except KeyError:
            spectrum = None
            print(f'Did not recognize target name: {self.targname}')

        return spectrum



    def make_bandpass(self, filt, aper_arcsec):
        """Simulate bandpass by filter.

        Makes bandpass for observation based on detector,
        filter, and aperture size. `SynTarget` can handle
        multiple filter bandpasses for a single target.

        Parameters
        ----------
        self : `SynTarget`
            Simulated target object.
        filt : str
            Which WFC3 filter to use.
        aper_arcsec : float or str
            Default is '.4'.
        time_dep : Boolean
            If the observation should account for time-
            dependent zeropoints. Default for IR is False.
            Provided here to hopefully generalize to UVIS
            at some point.
        """
        print(f'  Creating IR bandpass for {filt} and a {aper_arcsec} '\
              'arcsec aperture')

        bandpass = stsynphot.band(f'wfc3,ir,{filt.lower()},'\
                                  f'aper#{aper_arcsec}')

        if filt in self.bandpasses:
            print(f'    Overwriting bandpass for {filt}')
        self.bandpasses[filt] = bandpass


    def make_observation(self, filt):
        """Simulate observation by filter.

        Generates synthetic observation using bandpass
        created with `make_bandpass()` method. If filter
        does not have a generated bandpass (stored as the
        value for `self.bandpasses[filt]`), will print/log
        an error message.

        Parameters
        ----------
        filt : str
            Which WFC3 filter to use.
        """
        if filt in self.bandpasses.keys():
            print('    Generating synthetic observation for '\
                  f'{self.targname}, {filt}')
            obs = synphot.Observation(self.spectrum,
                                      self.bandpasses[filt],
                                      binset=self.bandpasses[filt].binset)
            if filt in self.observations:
                print('      Overwriting synthetic observation for '\
                      f'{self.targname}, {filt}')
            self.observations[filt] = obs

        else:
            print('    Unable to generate synthetic observation for '\
                  f'{self.targname} because the {filt} bandpass was not '\
                  'created.\n    Please use `make_bandpass()`.')


    def get_phot_table(self, filters, aper_arcsec):
        """Simulates photometry for specified filters.

        Creates table of synthetic photometry for the given
        target in specified filters on the WFC3/IR
        detector.

        Parameters
        ----------
        self : `SynTarget`
            Object representing a synthetic staring mode
            observation.
        filters : str or list of str
            Which WFC3 filters to use.
        aper_arcsec : str
        """
        if isinstance(filters, str):        # if only one filter, put it into a list.
            filters = [filters]

        print(f'  Making photometry table for {self.targname}')

        rows = []
        for filt in filters:
            SynTarget.make_bandpass(self, filt, aper_arcsec)
            SynTarget.make_observation(self, filt)

            syn_obs = self.observations[filt]
            syn_mag = np.log10(syn_obs.countrate(stsynphot.conf.area).value) * -2.5
            syn_cr = syn_obs.countrate(stsynphot.conf.area)

            row = [self.targname, filt, aper_arcsec, syn_mag, syn_cr]
            rows.append(row)

            print(f'      Photometry calculated for {filt}')

        self.phot_table = Table(rows=rows,
                                names=('targname', 'filter',
                                       'aperture (arcsec)',
                                       'syn_mag', 'syn_cr'))


def split_batches(batches, category='targets'):
    """Helper function.
    """
    i_dict = {'targets': 1, 'filters': 2}
    items = sorted(list(set([b.split('/')[i_dict[category]] for b in batches])))

    return items


def make_syn_targets(filepaths_batches, radius):
    """Makes dictionary of synthetic targets & photometry

    Parameters
    ----------
    filepaths_batches : list
    radius : int

    Returns
    -------
    syn_targets : dict
    """
    print(f'{"-"*52}\nConstructing dictionary of synthetic observations...')

    targets = split_batches(filepaths_batches, category='targets')
    filters = split_batches(filepaths_batches, category='filters')

    syn_targets = {}
    for target in targets:
        syn_target = SynTarget(target)

        if syn_target.spectrum is not None:
            aper_arcsec = str(0.13 * radius)
            syn_target.get_phot_table(filters, aper_arcsec=aper_arcsec,)#aper_arcsec='.4',
            syn_targets[target] = syn_target  # add to dictionary
            del syn_target                    # then delete

        else:
            print(f'      Unable to create synthetic spectrum for {target}.')

    return syn_targets
