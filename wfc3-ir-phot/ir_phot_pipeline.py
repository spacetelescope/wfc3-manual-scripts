# pylint: disable=E1101
"""
Pipeline for IR staring mode standard star photometry monitor.

Notes
-----
- Right now, just works for FLTs
- No analysis set up here, all in notebooks
- Set up function to physically (re)move bad data (GS fail, no source)

Usage
-----

python ir_phot_pipeline.py --trial --verbose --log --get_new_data --run_ap_phot --ap_phot_flt --helium_corr

    This monitor is primarily designed to run from the
    command line, with a total of 22 possible configurable
    arguments: 4 pipeline settings, 5 pipeline execution
    flags, and 13 pipeline parameters.

        > python ir_phot_pipeline.py [-n NAME] [--trial]
              [--verbose] [--log] [--get_new_data] [--redownload]
              [--drizzle] [--storm] [--run_ap_phot]
              [--proposals PROPOSALS [PROPOSALS ...]]
              [--targets TARGETS [TARGETS ...]]
              [--filters FILTERS [FILTERS ...]]
              [--file_type {flt,drz}] [--radius RADIUS]
              [--annulus ANNULUS] [--dannulus DANNULUS]
              [--back_method {mean,median,mode}]
              [--ap_phot_drz] [--ap_phot_flt]
              [-w WRITE_DIR] [--plot_sources]
              [--helium_corr]

    All arguments have defaults set, so the monitor can be
    run without any arguments at all:

        > python ir_phot_pipeline.py

    The 22 arguments are explained in greater detail in
    `ir_phot_toolbox.py`, and can also be viewed by using
    the `--help` flag.

        > python ir_phot_pipeline.py --help

Author
------
    Mariarosa Marinelli, 2023
"""

import os
import warnings
from glob import glob

from astropy.io import fits
from astropy.table import Table
import numpy as np
from photutils.aperture import CircularAnnulus, CircularAperture, aperture_photometry
from photutils.detection import DAOStarFinder
from photutils.segmentation import detect_sources, detect_threshold, SourceCatalog

from pyql.database.ql_database_interface import session, Master, Anomalies
from wfc3_phot_tools.staring_mode.background import make_aperture_stats_tbl
from wfc3_phot_tools.staring_mode.aperture_phot import iraf_style_photometry
from wfc3_phot_tools.staring_mode.rad_prof import RadialProfile

from ir_download import get_new_data_wrapper
from ir_file_io import initialize_directories, locate_data, move_bad_files, set_tbl_path
from ir_fits import get_ext_data, get_hdr_info
from only_helium import only_helium
from ir_logging import command_line_logging, display_message
from ir_plotting import plot_flt_sources
from ir_syn import make_syn_targets
from ir_toolbox import display_args, make_phot_cols, parse_args, PAM

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

class ObsBatch():
    """WFC3/IR standard star staring mode observations.

    A class to represent a batch of WFC3/IR staring mode
    observations for a specific proposal, standard star,
    and filter. Requires five attributes to initialize, and
    has 6 methods for reducing/analyzing/compiling data.

    Attributes
    ----------
    args : `argparse.Namespace` or `InteractiveArgs`
        Parsed command line arguments.
    proposal : int
        Program ID.
    targname : str
        Name of target. Should be in simplest form.
    filter : str
        Name of WFC3/IR target.
    filepaths : list of str
        List of filepaths to files for this batch's
        proposal/target/filter combination.

    Methods
    -------
    flt_dq_cutout(buffer)
        Count number of flagged pixels in DQ cutout.
    flt_apply_pam(pam_filepath)
        Applies pixel area map to FLT data.
    flt_find_sources(nsigma, npixels, edge_pixels, plot, plot_dir, cr_pd)
        Identifies sources in FLT data.
    find_sources_drz(fwhm=1.2, threshold=10.)
        TK in progress.
    check_for_anomalies()
        Checks for Quicklook-flagged anomalies in an
        observation.
    flt_photometry(self, syn_target, plot_dir)
        Does photometry on batch of files.

    Notes
    -----
      - Should either initialize with list of FLT files (then
        drizzle and do photometry and/or just do photometry),
        or with a specific drizzled file (and .coo ?)
    """
    def __init__(self, proposal, targname, filt, filepaths, args):
        """
        Parameters
        ----------
        self : `ObsBatch`
            Staring mode observation object.
        proposal : int
            Program ID.
        targname : str
            Name of target. TK: resolvable.
        filter : str
            Name of WFC3/IR target.
        filepaths : list of str
            List of filepaths to files for this batch's
            proposal/target/filter combination.
        args : `argparse.Namespace` or `InteractiveArgs`
            Arguments.
        """
        self.args = args
        self.proposal = proposal
        self.targname = targname
        self.filt = filt
        self.file_type = args.file_type#.lower()
        self.filepaths = filepaths                  # raw files if helium corr on
        self.ltv1 = None
        self.ltv2 = None
        self.source_tbl = None
        self.coords = None
        self.phot_tbl = None
        self.ql_root = None
        self.ql_flags = None
        self.exposure_file = None
        self.rootname = None
        self.hdr = None
        self.data_arr = None
        self.err_arr = None
        self.dq_arr = None
        self.data_corr = None
        self.syn_phot_row = None
        self.syn_phot_mag = None
        self.syn_phot_cr = None
        self.source_row = None
        self.xcentroid = None
        self.ycentroid = None
        self.dq_buffer = None
        self.dq_cutout = None
        self.dq_count = None
        self.phot_ap = None
        self.sky_ap = None
        self.recentered_x = None
        self.recentered_y = None
        self.detx = None
        self.dety = None

        if (self.filt in ['F105W', 'F110W']) and self.args.helium_corr:
            display_message(verbose=self.args.verbose,
                            log=self.args.log,
                            message="Commencing helium correction....",
                            log_type='info')

            dirname = os.path.dirname(self.filepaths[0])
            staging = '/grp/hst/wfc3v/wfc3photom/data/ir_staring_monitor/stage'
            only_helium(dirname, staging)


    def flt_dq_cutout(self, buffer):
        """Count number of flagged pixels in DQ cutout.

        Parameters
        ----------
        self : `ObsBatch`
            Staring mode observation object.
        buffer : int
            Width in pixels around the detected source to
            include in the DQ cutout. A 3-pixel buffer will
            produce a 7x7 pixel cutout, since the detected
            source centroid is the center pixel.

        Returns
        -------
        dq_cutout : array-like
            Square cutout around the detected source, with
            width equal to 2n + 1, where n = `buffer`.
        dq_count : int
            Number of pixels in `dq_cutout` that have at
            least one DQ flag.
        """
        index_x = int(self.xcentroid)
        index_y = int(self.ycentroid)

        dq_cutout = self.dq_arr[index_y-buffer:index_y+buffer,
                                index_x-buffer:index_x+buffer]
        dq_count = np.sum(dq_cutout > 0.)

        return dq_cutout, dq_count


    def flt_apply_pam(self, pam_filepath):
        """Applies pixel area map to FLT data.

        To correct for geometric distortion, FLT data
        is multiplied by the pixel area map.

        Parameters
        ----------
        self : `ObsBatch`
            Staring mode observation object.
        pam_filepath : str
            String representation of the path to the
            pixel area map files.

        Notes
        -----
            - TK: make generalizable for both detectors.
        """
        _dy, _dx = self.data_arr.shape

        # from Varun's code: not sure why we do this
        self.ltv1 = int(-1*self.hdr['ltv1'])
        self.ltv2 = int(-1*self.hdr['ltv2'])

        y_bounds = (self.ltv2, self.ltv2 + _dy)
        x_bounds = (self.ltv1, self.ltv1 + _dx)

        with fits.open(pam_filepath) as pam_fits:
            pam = pam_fits[1].data

        pam_sec = pam[y_bounds[0]:y_bounds[1], x_bounds[0]:x_bounds[1]]
        corr_data = self.data_arr * pam_sec

        return corr_data


    def compare_detected_sources_phot(self, props_tbl, edge_pixels, cr_pd):
        """Compare sources to synthetic target.

        First, checks detected source(s) against the shape
        of the data array to make sure the observation(s)
        are within a certain pixel margin of the edges.
        This margin right now is set to:
            `args.annulus` + `args.dannulus`

        Any sources that are too close to the edge by this
        metric are removed from `props_tbl`. If no sources
        remain, then this function returns `None`. If any
        sources do remain, a quick round of photometry is
        performed.

        The source with the smallest percent difference
        between the measured and synthetic count rates
        should ideally be the actual source; one last check
        is performed to ensure that the measured count rate
        is within a 25% percent difference threshold. If it
        is, then this function returns the corresponding
        `props_tbl` row. Otherwise, it returns `None`.

        Parameters
        ----------
        self : `ObsBatch`
            Staring mode observation object.
        props_tbl : `astropy.table.table.Table`
            Table with identified sources' properties.
        edge_pixels : int
            Limit for source location relative to the edges
            of the detector (sub)array.
        cr_pd : float
            Threshold for percent difference between source
            count rate and synthetic count rate.

        Returns
        -------
        use_source : `astropy.table.row.Row` or NoneType
            Row corresponding to detected source to use for
            photometry. If no detected source is inside the
            edge margins and within 25% of the synthetic
            count rate, then `use_source` is `None`.
        props_tbl : `astropy.table.table.Table`
            Table with identified sources' properties, and
            added columns:
                'within_edge' : str
                    Values are either 'y' or 'n'.
                'sources_pd' : float
                    Values are either 9999., indicating
                    that the source was too close to the
                    edge, or the percent difference between
                    the source measured count rate and the
                    synthetic count rate.
        messages : list of str
            String(s) to be logged/printed, communicating
            the outcome of the detected sources.
        """
        xmin = ymin = edge_pixels
        xmax = self.data_corr.shape[0] - edge_pixels
        ymax = self.data_corr.shape[1] - edge_pixels

        within_edge = []
        for row in props_tbl:
            xcentroid = row['xcentroid']
            ycentroid = row['ycentroid']

            if (xcentroid > xmin) and (xcentroid < xmax) \
                and (ycentroid > ymin) and (ycentroid < ymax):
                within_edge.append('n')
            else:
                within_edge.append('y')

        props_tbl['within_edge'] = within_edge

        not_within_edge_tbl = props_tbl[props_tbl['within_edge'] == 'n']

        if len(not_within_edge_tbl) == 0:
            use_source = None

            if len(props_tbl) == 1:
                message = '  The only detected source is'
            else:
                message = f'  All {len(props_tbl)} detected sources are'

            messages = [f'{message} within {edge_pixels} pixels '\
                        'of the detector/subarray edge.',
                        f'{" "*4} Observation will not be used for photometry.']

        else:
            sources_cr = []
            for row in props_tbl:
                # To make sure length of list equals length of OG table:
                if row['within_edge'] == 'y':
                    sources_cr.append(9999.)

                else:
                    xcentroid = row['xcentroid']
                    ycentroid = row['ycentroid']

                    phot_ap = CircularAperture([(xcentroid, ycentroid)], r=3)
                    sky_ap = CircularAnnulus([(xcentroid, ycentroid)],
                                             r_in=14, r_out=19)

                    row_phot = iraf_style_photometry(phot_ap, sky_ap,
                                                     self.data_corr,
                                                     error_array=self.err_arr,
                                                     bg_method='median',
                                                     epadu=1.)[0]
                    sources_cr.append(row_phot['flux'])

            sources_pd = [100 * np.abs(source_cr - self.syn_phot_cr)/self.syn_phot_cr
                          if source_cr != 9999. else 9999.
                          for source_cr in sources_cr]

            props_tbl['sources_pd'] = sources_pd

            # If the smallest percent difference between the
            # source count rate and the synthetic count rate
            # is less than 25%, then we'll use it
            if np.min(sources_pd) < cr_pd:
                use_source = props_tbl[sources_pd.index(np.min(sources_pd))]
                if len(props_tbl) == 1:
                    message = '  1 source found.'
                else:
                    message = f'  {len(props_tbl)} total sources found.'
                messages = [message,
                            f'{" "*4} Using source at '\
                            f'x={use_source["xcentroid"]}, '\
                            f'y={use_source["ycentroid"]}']

            else:
                use_source = None

                if len(not_within_edge_tbl) == 1:
                    message = '  Measured count rate of detected source is'
                else:
                    message = '  Measured count rate of all '\
                              f'{len(not_within_edge_tbl)} detected sources are'

                messages = [f'{message} greater than {cr_pd}% '\
                            'different from synthetic target count rate.',
                            f"{' '*4} Exposure won't be used for photometry."]

        return use_source, props_tbl, messages


    def flt_find_sources(self, nsigma, npixels, edge_pixels, plot_dir, cr_pd=25.):
        """Identifies sources in FLT data.

        Parameters
        ----------
        self : `ObsBatch`
            Staring mode observation object.
        nsigma : float or int
            The number of standard deviations per pixel
            above the background for which to consider a
            pixel as possibly being part of a source.
            Passed to the `detect_threshold()` function
            from `photutils.segmentation`.
        npixels : int
            The minimum number of connected pixels, each
            greater than threshold, that an object must
            have to be detected. Used in `detect_sources()`
            function from `photutils.segmentation`.
        plot_dir : str
            Directory to which plot should be saved.
        cr_pd : float
            Threshold for percent difference between source
            count rate and synthetic count rate. Default is
            25%.

        Returns
        -------
        use_source : `astropy.table.row.Row` or NoneType
            If at least one source is found, should return
            the row of the properties row corresponding to
            the matching source. If no viable sources is
            found, returns `None`.
        """
        # Create a threshold image from the PAM-corrected data.
        threshold = detect_threshold(self.data_corr, nsigma=nsigma)
        # Now use the threshold image to make a segmentation map.
        segm = detect_sources(self.data_corr, threshold, npixels=npixels)

        # If no sources are detected, the segmentation map will be `None`.
        if segm is None:
            use_source = None
            display_message(verbose=self.args.verbose, log=self.args.log,
                            log_type='warning', message='No sources found.')

        else:
            props = SourceCatalog(self.data_corr, segm)
            props_tbl = props.to_table()

            use_source, props_tbl, \
            messages = self.compare_detected_sources_phot(props_tbl, edge_pixels,
                                                          cr_pd)
            for message in messages:
                display_message(verbose=self.args.verbose, log=self.args.log,
                                log_type='info', message=message)

            if self.args.plot_sources:
                with warnings.catch_warnings():
                    # Not even sure this is doing anything to be honest.
                    warnings.filterwarnings("ignore",
                                            message="findfont: Generic family"\
                                                    " 'serif' not found because"\
                                                    " none of the following "\
                                                    "families were found: "\
                                                    "Computer Modern Roman")

                    plot_flt_sources(self, props_tbl, use_source, cr_pd,
                                     verbose=self.args.verbose, log=self.args.log,
                                     plot_dir=plot_dir)

        return use_source


    def find_sources_drz(self, data, fwhm=1.2, threshold=10.):
        """Identifies sources in DRZ data.

        Parameters
        ----------
        self : `ObsBatch`
            Staring mode observation object.
        data : something
        fwhm : float or int
            Full width at half-maximum.
        threshold : float or int
            Threshold over background.

        Notes
        -----
            - Currently unused.
            - Parameters not optimized yet
        """
        dsf = DAOStarFinder(fwhm=fwhm, threshold=threshold)
        self.source_tbl = dsf.find_stars(self, data)
        self.coords = list(zip(self.source_tbl['xcentroid'],
                               self.source_tbl['ycentroid']))


    def check_for_anomalies(self):
        """Checks for flagged anomalies in an observation.

        This function uses `pyql` to check the `Anomalies`
        table of the Quicklook database, pulling Boolean
        flags indicating whether an observation was
        affected by guidestar failures and/or satellite
        trails. If an observation was not found to have
        either anomaly AND the daily Quicklooker DID NOT
        submit a blank report, the observation will not
        appear in the `Anomalies` table. If an observation
        was not found to have either anomaly AND the daily
        Quicklooker DID submit a blank report, the flags in
        the table will both be 'False', so we can feed the
        values to the dictionary `self.ql_flags`.

        Flags are returned as strings instead of Booleans
        because there's a weird bug causing masked values
        to appear in the photometry table downstream, and I
        was too lazy to properly track down what was
        triggering the unexpected behavior.

        Parameter
        ---------
        self : `ObsBatch`
            Staring mode observation object.

        Returns
        -------
        use_obs_for_phot : str
            Flag indicating whether observation should be
            used for photometry.
        satellite_trail : str
            Whether the observation has been flagged in
            Quicklook as containing a satellite trail.
        """
        # Use ql_root to avoid mixups between transmission characters.
        self.ql_root = self.hdr['rootname'][:-1]
        self.ql_flags = {}

        results = session.query(Master.ql_root, Anomalies.ql_root,
                                Anomalies.satellite_trail,
                                Anomalies.guidestar_failure).\
                          join(Master, Master.ql_root == Anomalies.ql_root).\
                          filter(Master.ql_root == self.ql_root).\
                          all()

        # If no results are returned, then the observation has not been added
        # to the Anomalies table. Why this works: prior to early 2022, daily
        # Quicklookers did not use the "Submit" button if there were no noted
        # anomalies. Observations are only added to this table when the "Submit"
        # button is pressed. Assuming that all images have been Quicklooked
        # (i.e. that you are not running this between ingest and when the daily
        # Quicklooker reaches this image), this serves as a nifty shortcut. Not
        # in the Anomalies table? Must not be any anomalies.
        if len(results) == 0:
            self.ql_flags['satellite_trail'] = False
            self.ql_flags['guidestar_failure'] = False

        # Just because it's in the Anomalies table doesn't mean it must have a
        # satellite trail or guidestar failure. An image with only the diamond
        # feature would still get added to the table and would be returned by
        # our query. So we have to check the actual values of the the satellite
        # trail/guidestar failure columns.
        # TO DO : this is populating masked items for some reason. Fix.
        else:
            self.ql_flags['satellite_trail'] = results[0].satellite_trail
            self.ql_flags['guidestar_failure'] = results[0].guidestar_failure

        # Don't use images with GS fails for calibration photometry. Duh.
        if self.ql_flags['guidestar_failure']:
            use_obs_for_phot = 'False'

            if self.ql_flags['satellite_trail']:
                satellite_trail = 'True'
            else:
                satellite_trail = 'False'

            display_message(verbose=self.args.verbose, log=self.args.log,
                            log_type='error',
                            message='  Affected by guidestar failure, '\
                                    'cannot use for photometry.')

        # Images with satellite trails should be inspected more closely later.
        else:
            use_obs_for_phot = 'True'

            if self.ql_flags['satellite_trail']:
                satellite_trail = 'True'
                display_message(verbose=self.args.verbose, log=self.args.log,
                                log_type='warning',
                                message='  Affected by satellite trail. Use '\
                                        'with caution.')
            else:
                satellite_trail = 'False'
                display_message(verbose=self.args.verbose, log=self.args.log,
                                log_type='info',
                                message='  Unaffected by guidestar failure '\
                                        'or satellite trails.')

        return use_obs_for_phot, satellite_trail


    def flt_photometry(self, syn_target, plot_dir):
        """Does photometry on batch of files.

        Parameters
        ----------
        syn_targ : `synTarget`
            Synthetic target with multiple bandpasses
            and observations corresponding to filters
            set by `args.filters`.
        plot_dir : str

        Returns
        -------
        phot_tbl : `astropy.table.Table`
        """
        # Begin creating the columns and rows of our photometry table.
        phot_rows = []

        bad_files = []
        # Refactor to use subclass?
        # Iterate over filepaths in observation batch.
        # Reassign values each time - we won't need them again.
        for i, filepath in enumerate(self.filepaths):
            self.exposure_file = filepath
            self.rootname = fits.getval(self.exposure_file, 'ROOTNAME', 0)

            condition1 = fits.getval(self.exposure_file, 'SCAN_TYP', 0) == 'N'

            if not condition1:
                messages = ['.', f'{self.rootname}:', 'Skipped:'
                            f'{" "*4}- Spatial scan observation.']
                for message in messages:
                    display_message(verbose=self.args.verbose, log=self.args.log,
                                    log_type='info', message=message)

                bad_files.append(self.exposure_file)

            else:
                self.hdr = get_hdr_info(self.exposure_file, self.args.verbose,
                                        self.args.log)

                messages = ['.', f'{self.hdr["rootname"]}:']
                for message in messages:
                    display_message(verbose=self.args.verbose, log=self.args.log,
                                    log_type='info', message=message)

                # Don't use GS fails, warn for any satellite trails.
                use_obs_for_phot, satellite_trail = self.check_for_anomalies()
                # Second condition is that there are no QL flags that
                # indicate this observation can't be used for photometry.
                condition2 = use_obs_for_phot == 'True'

                # Get data arrays from FITS file. This is the basis
                # of the third condition.
                self.data_arr, self.err_arr, self.dq_arr = get_ext_data(self.exposure_file)
                condition3 = not isinstance(self.dq_arr, type(None))

                # Apply pixel area map to correct geometric distortion.
                self.data_corr = self.flt_apply_pam(PAM)

                # Extract appropriate row in synthetic target photometry table.
                self.syn_phot_row = syn_target.phot_table[\
                                    syn_target.phot_table['filter'] == \
                                    self.hdr['filter']][0]
                # For ease, assign these values their own attributes.
                # Yeah this is hacky and I should have split this up more.
                self.syn_phot_mag = self.syn_phot_row['syn_mag']
                self.syn_phot_cr = self.syn_phot_row['syn_cr']


                #edge_pixels = self.args.annulus + self.args.dannulus  # 19 by default
                edge_pixels = 6  # Lowered threshold. Will this crash?

                # Find sources in corrected data.
                # TK May need to build in iterative re-scaling of `npixels` for N bands.
                self.source_row = self.flt_find_sources(nsigma=3.0, npixels=15,
                                                        edge_pixels=edge_pixels,
                                                        plot_dir=plot_dir) # TKTK

                # Fourth condition is that at least one matching source
                # is identified. `self.source_row` will be `None` if no
                # viable sources can be found.
                if not isinstance(self.source_row, type(None)):
                    condition4 = True
                else:
                    condition4 = False
#                condition4 = isinstance(self.source_row, type(None))

                # Entonces, only do photometry if all conditions are met.
                if condition2 and condition4 and condition3:
                    self.xcentroid = self.source_row['xcentroid']
                    self.ycentroid = self.source_row['ycentroid']

                    # Check the data quality flags.
                    self.dq_buffer = 3
                    dq_cutout, dq_count = self.flt_dq_cutout(buffer=self.dq_buffer)
                    self.dq_cutout, self.dq_count = dq_cutout, dq_count

                    # Create aperture/annulus objects.
                    self.phot_ap = CircularAperture([(self.xcentroid, self.ycentroid)],
                                                    r=self.args.radius)
                    self.sky_ap = CircularAnnulus([(self.xcentroid, self.ycentroid)],
                                                  r_in=self.args.annulus,
                                                  r_out=self.args.annulus + \
                                                        self.args.dannulus)

                    # Make first pass fit with RadialProfile,
                    # recenter source and re-fit.
                    prof = RadialProfile(self.xcentroid, self.ycentroid, self.data_corr,
                                         recenter=True, fit=False, r=1)
                    self.recentered_x, self.recentered_y = prof.x, prof.y
                    prof = RadialProfile(self.recentered_x, self.recentered_y,
                                         self.data_corr, recenter=False,
                                         fit=True, r=2)
                    # Set dummy values for invalid FWHM or chi-squared values.
                    if np.isnan(prof.fwhm):
                        prof.fwhm = -9999.
                    if np.isnan(prof.chisquared):
                        prof.chisquared = -9999.

                    # Leftover from testing background subtraction methods.
                    # Can probably remove at some point.
                    method_fluxes, method_flux_errs = [], []
                    back_methods = ['mean', 'median', 'mode']

                    for method in back_methods:
                        # TO DO: use wrapper to display output.
                        phot_row = iraf_style_photometry(self.phot_ap,
                                                         self.sky_ap,
                                                         self.data_corr,
                                                         error_array=self.err_arr,
                                                         bg_method=method,
                                                         epadu=1.)[0]
                        method_fluxes.append(phot_row['flux'])
                        method_flux_errs.append(phot_row['flux_error'])

                    # TO DO: use wrapper to display output
                    photutils_sum = aperture_photometry(apertures=self.phot_ap,
                                                        data=self.data_corr)\
                                                             ['aperture_sum'][0]

                    # Get statistics/measurements for background annulus.
                    # TO DO: rethink parameter naming for this one?
                    bg_stats = make_aperture_stats_tbl(self.data_corr, self.sky_ap)[0]

                    # Leftover from testing background subtraction methods.
                    obs_syn_crs = []
                    for i, method in enumerate(back_methods):
                        obs_syn_crs.append(method_fluxes[i]/self.syn_phot_cr)

                    obs_syn_photutils = photutils_sum/self.syn_phot_cr

                    # Have to revert LTV_ to original form to calc detx and dety
                    self.detx = self.xcentroid - (self.ltv1 / -1)
                    self.dety = self.ycentroid - (self.ltv2 / -1)

                    file_row = [self.exposure_file, self.file_type,
                                self.args.radius, self.args.annulus,
                                self.args.dannulus, self.args.back_method,
                                satellite_trail]

                    file_row.extend([value for key, value in self.hdr.items()])

                    # If nothing has been added to row list,
                    # i.e. earlier filepaths didn't have detections:
                    if len(phot_rows) == 0:
                        phot_cols = make_phot_cols(self.hdr, self.dq_buffer)

                    file_row.extend([self.xcentroid, self.ycentroid,
                                     self.recentered_x, self.recentered_y,
                                     prof.fwhm, prof.chisquared,
                                     self.ltv1, self.ltv2, self.detx, self.dety,
                                     bg_stats['aperture_median'],
                                     bg_stats['aperture_mode'],
                                     bg_stats['aperture_mean'],
                                     bg_stats['aperture_std'],
                                     bg_stats['aperture_nonnan_area'],
                                     phot_row['phot_ap_area'],
                                     method_fluxes[0], method_flux_errs[0],
                                     method_fluxes[1], method_flux_errs[1],
                                     method_fluxes[2], method_flux_errs[2],
                                     self.syn_phot_mag, self.syn_phot_cr,
                                     obs_syn_crs[0],  obs_syn_crs[1],
                                     obs_syn_crs[2], photutils_sum,
                                     obs_syn_photutils, np.std(self.data_corr),
                                     self.dq_count])

                    phot_rows.append(file_row)

                else:
                    messages = ['  Skipped:']

                    if not condition2:
                        messages.append(f'{" "*4}- Observation affected by '
                                        'guidestar failure.')
                        bad_files.append(self.exposure_file)

                    if not condition4:
                        messages.append(f'{" "*4}- No viable sources detected.')
                        # TKTK
                    if not condition3:
                        messages.append(f'{" "*4}- DQ array is empty.')
                        bad_files.append(self.exposure_file)

                    for message in messages:
                        display_message(verbose=self.args.verbose, log=self.args.log,
                                        log_type='info', message=message)


        move_bad_files(bad_files, verbose=self.args.verbose, log=self.args.log)


        if len(phot_rows) == 0:
            phot_tbl = None
        else:
            phot_tbl = Table(rows=phot_rows, names=phot_cols)

        return phot_tbl


def run_process(args, dirs, write, overwrite):
    """Run drizzling and/or photometry for pipeline.

    Parameters
    ----------
    args : `argparse.Namespace` or `InteractiveArgs`
        Arguments.
    dirs : dict
        Dictionary of directories.
    write : Boolean
        Determines if the photometry table should be saved.
        Default is True.
    overwrite : Boolean
        Determines if an existing photometry table should
        be overwritten. Default is False.
    """
    # TO DO: update this assignation once drizzling is implemented.
    process_name = 'photometry'

    if process_name is None:
        error_messages = ['Drizzling & aperture photometry flags are both set'\
                          'to `False`.', 'Will not take any further action.']
        for error_message in error_messages:
            display_message(verbose=args.verbose, log=args.log,
                            log_type='error', message=error_message)

    else:
        # Dictionary keys are batch keys in the format `'proposal/target/filter'`,
        # while the values are lists of file paths in that directory.
        filepaths_batches = locate_data(args, dirs['data_dir'])

        if args.run_ap_phot:
            # Set up synthetic targets first, to use for each batch.
            syn_targets = make_syn_targets(filepaths_batches,
                                           verbose=args.verbose, log=args.log)

            # Iterate through batches.
            for batch_key, filepaths in filepaths_batches.items():
                proposal, target, filt = batch_key.split('/')
                filename = f'phot_{proposal}_{target}_{filt}.csv'

                # Essentially bookmark where the batch starts.
                status = f'Initializing batch of files for {process_name}:'
                dashes = '-'*len(status)
                messages = [dashes, status,
                            f'PROGRAM: {proposal}',
                            f'TARGET:  {target}',
                            f'FILTER:  {filt}',
                            f'FILES:   {len(filepaths)}']
                for message in messages:
                    display_message(verbose=args.verbose, log=args.log,
                                    log_type='info', message=message)

#            if args.drizzle:
#                pass

            #if args.run_ap_phot:
                if write:
                    # Aborts if overwrite is False but table exists.
                    tbl_path = set_tbl_path(filename, write_dir=args.write_dir,
                                            overwrite=overwrite,
                                            verbose=args.verbose, log=args.log)

                # Caution: target should be in simplest form.
                batch = ObsBatch(proposal, target, filt, filepaths, args)
                batch.filepaths = [f.replace('_raw', '_flt') for f in batch.filepaths]

#                if resolved_calwf3_issues:
                    # Run photometry for this batch of files.
                phot_table = batch.flt_photometry(syn_targets[target],
                                                  dirs['plots_dir'])

                if phot_table is not None:
                    phot_table.write(tbl_path, overwrite=overwrite, format='csv')
                    messages = ['Wrote table to:', f'    {tbl_path}', dashes]
                    for messages in message:
                        display_message(verbose=args.verbose, log=args.log,
                                        log_type='info', message=message)

                # Clean up after yourself.
                del batch

        status = f'Finished {process_name} for batches:'
        messages = [f'{"-"*len(status)}', status]
        messages.extend([f'    {batch_key}' for batch_key in filepaths_batches])
        messages.append('*'*80)

        for message in messages:
            display_message(verbose=args.verbose, log=args.log,
                            log_type='info', message=message)


def run_pipeline(args, dirs):
    """Run the whole shebang.

    Runs the IR photometry pipeline using the parsed
    arguments and the dictionary of directories.

    Parameters
    ----------
    args :
    dirs :
    """
    if args.get_new_data:
        get_new_data_wrapper(args, dirs)

    #if args.helium_corr:
    #    setup_calwf3_environs(args.verbose, args.log)

    run_process(args, dirs, write=True, overwrite=True)


if __name__ == '__main__':
    # Parse command line arguments.
    parsed_args = parse_args()

    # Set up logging if necessary.
    command_line_logging(parsed_args)

    # Display command line arguments.
    display_args(parsed_args)

    # Set up needed directories.
    run_dirs = initialize_directories(parsed_args)

    # Showtime.
    run_pipeline(parsed_args, run_dirs)
