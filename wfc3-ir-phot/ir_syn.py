"""
Functions and a class to enable synthetic photometry for the IR staring mode
standard star pipeline.


Functions
---------
make_synthetic_spectrum(targname)
    Make synthetic spectrum for specified target. Compares target name
    to catalog of names and CALSPEC files. If the target name is not found
    or resolved, displays an error message and returns `None`.
make_syn_targets(filepaths_batches, verbose, log)
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

from ir_logging import display_message
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
    verbose : Boolean
        Whether to print the message.
    log : Boolean
        Whether to log the message.

    Methods
    -------
    make_bandpass(filt, aper_arcsec, verbose, log, time_dep=False)
        Simulate bandpass by filter.
    make_observation(filt, verbose, log)
        Simulate observation by filter.
    get_phot_table(filters, aper_arcsec, verbose, log)
        Simulates photometry for specified filters. Calls
        `make_bandpass()` and `make_observation()`.
    """
    def __init__(self, targname, verbose, log):
        self.targname = resolve_targnames(targname, simplify=True)
        self.bandpasses = {}
        self.observations = {}
        self.spectrum = make_synthetic_spectrum(self.targname, verbose, log)
        if self.spectrum is None:
            display_message(verbose=verbose, log=log, log_type='critical',
                            message='  Unable to initialize SynTarget for '\
                                    f'{self.targname}.')
        else:
            display_message(verbose=verbose, log=log, log_type='info',
                            message='  Initialized SynTarget for '\
                                    f'{self.targname}')

        self.phot_table = None


    def make_bandpass(self, filt, aper_arcsec, verbose, log, time_dep=False):
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
        verbose: Boolean
            Whether to print output.
        log : Boolean
            Whether to log output.
        time_dep : Boolean
            If the observation should account for time-
            dependent zeropoints. Default for IR is False.
            Provided here to hopefully generalize to UVIS
            at some point.
        """
        display_message(verbose=verbose,
                        log=log,
                        log_type='info',
                        message=f'  Creating IR bandpass for '\
                                f'{filt} and a {aper_arcsec} arcsec aperture')
        if not time_dep:
            bandpass = stsynphot.band(f'wfc3,ir,{filt.lower()},aper#{aper_arcsec}')

        if filt in self.bandpasses:
            display_message(verbose=verbose,
                            log=log,
                            log_type='warning',
                            message=f'{" "*4} Overwriting bandpass for {filt}')
        self.bandpasses[filt] = bandpass


    def make_observation(self, filt, verbose, log):
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
        verbose: Boolean
            Whether to print output.
        log : Boolean
            Whether to log output.
        """
        if filt in self.bandpasses.keys():
            display_message(verbose=verbose,
                            log=log,
                            log_type='info',
                            message=f'{" "*4} Generating synthetic observation'\
                                    f'for {self.targname}, {filt}')
            obs = synphot.Observation(self.spectrum,
                                      self.bandpasses[filt],
                                      binset=self.bandpasses[filt].binset)
            if filt in self.observations:
                display_message(verbose=verbose,
                                log=log,
                                log_type='warning',
                                message=f'{" "*6} Overwriting synthetic '\
                                        f'observation for {self.targname}, '\
                                        f'{filt}')
            self.observations[filt] = obs

        else:
            error_messages = [f'{" "*4} Unable to generate synthetic '\
                              f'observation for {self.targname} because the '\
                              f'{filt} bandpass has not been created.',
                              f'{" "*4} Please use `make_bandpass()`.']
            for error_message in error_messages:
                display_message(verbose=verbose,
                                log=log,
                                log_type='error',
                                message=error_message)


    def get_phot_table(self, filters, aper_arcsec, verbose, log):
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
        verbose: Boolean
            Whether to print output.
        log : Boolean
            Whether to log output.
        """
        if isinstance(filters, str):        # if only one filter, put it into a list.
            filters = [filters]

        display_message(verbose=verbose,
                        log=log,
                        log_type='info',
                        message=f'  Making photometry table for {self.targname}')
        rows = []
        for filt in filters:
            SynTarget.make_bandpass(self, filt, aper_arcsec, verbose, log)
            SynTarget.make_observation(self, filt, verbose, log)

            syn_obs = self.observations[filt]
            syn_mag = np.log10(syn_obs.countrate(stsynphot.conf.area).value) * -2.5
            syn_cr = syn_obs.countrate(stsynphot.conf.area)

            row = [self.targname, filt, aper_arcsec, syn_mag, syn_cr]
            rows.append(row)

            display_message(verbose=verbose,
                            log=log,
                            log_type='info',
                            message=f'{" "*6} Photometry calculated for {filt}')

        self.phot_table = Table(rows=rows,
                                names=('targname', 'filter',
                                       'aperture (arcsec)',
                                       'syn_mag', 'syn_cr'))


def make_synthetic_spectrum(targname, verbose, log):
    """Make synthetic spectrum for specified target.

    Compares target name to catalog of names and CALSPEC
    files. If the target name is not found or resolved,
    displays an error message and returns `None`.

    Parameter
    ---------
    targname : str
        Target name.

    Returns
    -------
    spectrum : `synphot.SourceSpectrum` or None
        If the specified target isn't in the `star_catalog`
        dictionary, will return None.
    """
    calspec_dir = '/grp/hst/cdbs/calspec'
    star_catalog = {'GD-153': 'gd153_stiswfcnic_003.fits',
                    'GD153': 'gd153_stiswfcnic_003.fits',
                    'GD-71': 'gd71_stiswfcnic_003.fits',
                    'GD71': 'gd71_stiswfcnic_003.fits',
                    'GSC-02581-02323': 'p330e_stiswfcnic_003.fits',
                    'GSC0258102323': 'p330e_stiswfcnic_003',
                    'P330E': 'p330e_stiswfcnic_003.fits',
                    'GRW+70D5824':  'grw_70d5824_stiswfcnic_003.fits',
                    'GRW70':  'grw_70d5824_stiswfcnic_003.fits',
                    'G191B2B': 'g191b2b_stiswfcnic_003.fits'}

    try:
        spectrum_path = os.path.join(calspec_dir, star_catalog[targname])
        spectrum = synphot.SourceSpectrum.from_file(spectrum_path)

    except KeyError:
        spectrum = None
        display_message(verbose=verbose,
                        log=log,
                        log_type='critical',
                        message=f'Did not recognize target name: {targname}')

    return spectrum


def make_syn_targets(filepaths_batches, verbose, log):
    """Makes dictionary of synthetic targets & photometry

    Parameters
    ----------
    filepaths_batches : list

    Returns
    -------
    syn_targets : dict
    """
    function_desc = 'Constructing dictionary of synthetic observations'
    dashes = '-'*len(function_desc)
    for message in [dashes, function_desc]:
        display_message(verbose=verbose,
                        log=log,
                        log_type='info',
                        message=message)

    # this is kind of ugly but basically I need to know what
    # targets and filters I will have before I do photometry.
    targets = list(set([batch_key.split('/')[1]
                        for batch_key in filepaths_batches]))
    filters = list(set([batch_key.split('/')[2]
                        for batch_key in filepaths_batches]))

    syn_targets = {}
    for target in targets:
        syn_target = SynTarget(target, verbose=verbose, log=log)

        if syn_target.spectrum is not None:
            syn_target.get_phot_table(filters, aper_arcsec='.4',
                                      verbose=verbose, log=log)
            syn_targets[target] = syn_target  # add to dictionary
            del syn_target                    # then delete

        else:
            display_message(verbose=verbose,
                            log=log,
                            log_type='error',
                            message=f'{" "*4} Unable to create synthetic '\
                                    f'spectrum for {target}.')

    return syn_targets
