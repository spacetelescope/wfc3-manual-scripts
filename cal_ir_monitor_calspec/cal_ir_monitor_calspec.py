# pylint: disable=E1101
"""
Pipeline for IR staring mode standard star photometry monitor.

Usage
-----

python cal_ir_monitor_calspec.py --trial
    --get_new_data --run_ap_phot --ap_phot_flt --helium_corr

    This monitor is primarily designed to run from the
    command line, with a total of 22 possible configurable
    arguments: 4 pipeline settings, 5 pipeline execution
    flags, and 13 pipeline parameters.

        > python ir_phot_pipeline.py [-n NAME] [--trial]
              [--get_new_data] [--redownload] [--run_ap_phot]
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

        > python cal_ir_monitor_calspec.py

    The 22 arguments are explained in greater detail in
    `ir_phot_toolbox.py`, and can also be viewed by using
    the `--help` flag.

        > python cal_ir_monitor_calspec.py --help

    Addionally, the `config.yaml` file contains configuration
    settings.

Author
------
    Mariarosa Marinelli, 2023-2025
"""

from dataclasses import dataclass, field
import warnings

from astropy.io import fits
from astropy.table import Table
import numpy as np
from photutils.aperture import CircularAnnulus, CircularAperture, aperture_photometry
from photutils.segmentation import detect_sources, detect_threshold, SourceCatalog

from pyql.database.ql_database_interface import session, Main, Anomalies
from wfc3_phot_tools.staring_mode.background import make_aperture_stats_tbl
from wfc3_phot_tools.staring_mode.aperture_phot import iraf_style_photometry
from wfc3_phot_tools.staring_mode.rad_prof import RadialProfile

from ir_download import get_new_data_wrapper
from ir_file_io import initialize_directories, locate_data, move_bad_files, set_tbl_path
from ir_fits import get_ext_data, get_hdr_info
from ir_plotting import plot_flt_sources
from ir_syn import make_syn_targets
from ir_td import correct_tds
from ir_toolbox import CR_PD, display_args, make_phot_cols, parse_args, PAM


warnings.filterwarnings("ignore", category=RuntimeWarning)


with fits.open(PAM) as pam_fits:
    PAM_ARR = pam_fits[1].data




@dataclass
class BatchMetadata:
    """Stores general metadata for a batch of observations.

    TKTK """
    paths: []
    proposal: int = None
    targname: str = None
    filt: str = None


@dataclass
class SyntheticPhotometry:
    """
    TKTK
    """
    #model: np.ndarray = None   # The model array for synthetic photometry TK
    #flux: float = None   # The flux value for synthetic photometry TK
    syn_phot_row: any = None
    syn_phot_cr: float = None


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
    flt_find_sources(plot_dir)
        Identifies sources in FLT data.
    find_sources_drz(fwhm=1.2, threshold=10.)
        TK in progress.
    check_for_anomalies()
        Checks for Quicklook-flagged anomalies in an
        observation.
    flt_photometry(self, syn_target, plot_dir)
        Does photometry on batch of files.

    """
    def __init__(self, args, param_dict,
                 metadata = BatchMetadata,
                 synthetic = SyntheticPhotometry):
    #proposal, targname, filt, filepaths, args):
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
        self.phot_tbl = None
        self.meta = metadata
        self.syn = synthetic
        self.image = None

        self.setup(param_dict)
        #self._get_metadata()

    def setup(self, param_dict):
        """
        setup
        """
        self._get_metadata(param_dict['proposal'],
                           param_dict['target'],
                           param_dict['filt'],
                           param_dict['filepaths'])

    def _get_metadata(self, proposal, targname, filt, paths):
        """
        get metadata
        """
        self.meta.proposal = proposal
        self.meta.targname = targname
        self.meta.filt = filt
        self.meta.paths = paths


    def get_synthetic_phot(self, syn_target):
        """
        Extract appropriate row in synthetic target photometry table.
        """
        # TKTKTK
        self.syn.syn_phot_row = syn_target.phot_table[\
                            syn_target.phot_table['filter'] == \
                            self.meta.filt][0]
        # For ease, assign these values their own attributes.
        # Yeah this is hacky and I should have split this up more.
#        self.syn.syn_phot_mag = self.syn.syn_phot_row['syn_mag']
        self.syn.syn_phot_cr = self.syn.syn_phot_row['syn_cr']


    def batch_photometry(self, syn_target, plot_dir):
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
        # Begin creating the rows of our photometry table.
        phot_rows = []
        bad_files = []

        self.get_synthetic_phot(syn_target)

        # Iterate over filepaths in observation batch.
        # Reassign values each time - we won't need them again.
        for path in self.meta.paths:
            # Reassign values each time.
            self.image = ObsImage(path, self.args)

            if self.image.check_quality(self.syn.syn_phot_cr, plot_dir):
                phot_rows.append(self.image.make_phot_row())

            else:
                bad_files.append(self.image.path)

        move_bad_files(bad_files)

        # Use the last image in the batch to generate columns.
        phot_cols = make_phot_cols(self.image.data.hdr,
                                   self.image.data.dq_buffer)
        if len(phot_rows) > 0:
            phot_tbl = Table(rows=phot_rows, names=phot_cols)
        else:
            phot_tbl = None

        return phot_tbl


def run_process(args, dirs, write, overwrite):
    """
    Run photometry for all data (with optional reprocessing
    controlled by the `--linearity` and `--helium` command
    line arguments).

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
    # Dictionary keys are batch keys in the format `'proposal/target/filter'`,
    # while the values are lists of file paths in that directory.
    filepaths_batches = locate_data(args, dirs['data_dir'])

    if args.run_ap_phot:
        # Set up synthetic targets first, to use for each batch.
        syn_targets = make_syn_targets(filepaths_batches, args.radius)

        # Iterate through batches.
        for batch_key, filepaths in filepaths_batches.items():
            proposal, target, filt = batch_key.split('/')

            # Essentially bookmark where the batch starts.
            print(f'{"-"*28}\nInitializing batch of files:\n'\
                  f'PROGRAM: {proposal}\nTARGET:  {target}\n'\
                  f'FILTER:  {filt}\nFILES:   {len(filepaths)}')

            batch = ObsBatch(args,
                             {'proposal': proposal, 'target': target,
                             'filt': filt, 'filepaths': filepaths})
            # TKTK after reprocessing, update filepaths
#            batch.meta.paths = [f.replace('_raw', '_flt') for f in batch.filepaths]

#                if resolved_calwf3_issues:
            # Run photometry for this batch of files.
            phot_table = batch.batch_photometry(syn_targets[target],
                                                dirs['plots_dir'])

            if phot_table is not None and write:
                # Aborts if overwrite is False but table exists.
                tbl_path = set_tbl_path(f'phot_{proposal}_{target}_{filt}.csv',
                                        args.write_dir, overwrite)
                phot_table.write(tbl_path, overwrite=overwrite, format='csv')
                print(f'Wrote table to:\n    {tbl_path}')

            # Clean up after yourself.
            del batch

        print(f'{"-"*32}\nFinished photometry for batches:')
        for key in filepaths_batches:
            print(f'    {key}')
        print('*'*80)


@dataclass
class AnomalyFlags:
    """Stores flags related to anomalies in the observation."""
    ql_root: str = None
    satellite_trail: bool = None
    guidestar_failure: bool = None
    use_obs_for_phot: bool = None

@dataclass
class ImagePhotometry:
    """Stores data related to detected sources and their photometry."""
    centroid: [None, None] #list = field(default_factory=list)
    det_coords: [None, None] #list = field(default_factory=list) #float = None
    recentered: [None, None] #list = field(default_factory=list) #float = None
    #ycentroid: float = None
    phot_ap: None
    sky_ap: None
    source_row: Table() = None
    dq_count: int = None
    #recentered_y: float = None
    #dety: float = None

@dataclass
class ImageData:
    """Stores the main observation data arrays and corrections."""
    hdr: {} = None
    #ltv: list = field(default_factory=list)
    ltv1: float = None
    ltv2: float = None
    data_arr: np.ndarray = None
    err_arr: np.ndarray = None
    dq_arr: np.ndarray = None
    dq_cutout: np.ndarray = None
    data_corr: np.ndarray = None


class ObsImage():
    """image class"""
    def __init__(self, filepath, args,
                 data = ImageData,
                 photometry = ImagePhotometry,
                 anomalies = AnomalyFlags):
        """
        initialize ObsImage
        """
        print(f'Created ObsImage: {filepath}')
        self.path = filepath
        self.args = args
        self.data = data
        self.phot = photometry
        self.anom = anomalies
        self.syn_cr = None

        self.get_data()


    def get_data(self):
        """
        Wrapper.
        """
        self.data.hdr = get_hdr_info(self.path)
        (self.data.data_arr, self.data.err_arr,
            self.data.dq_arr) = get_ext_data(self.path)


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
        self.anom.ql_root = self.data.hdr['rootname'][:-1]

        results = session.query(Main.ql_root, Anomalies.ql_root,
                                Anomalies.satellite_trail,
                                Anomalies.guidestar_failure).\
                          join(Main, Main.ql_root == Anomalies.ql_root).\
                          filter(Main.ql_root == self.anom.ql_root).\
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
            self.anom.satellite_trail = False
            self.anom.guidestar_failure = False

        # Just because it's in the Anomalies table doesn't mean it must have a
        # satellite trail or guidestar failure. An image with only the diamond
        # feature would still get added to the table and would be returned by
        # our query. So we have to check the actual values of the the satellite
        # trail/guidestar failure columns.
        # TO DO : this is populating masked items for some reason. Fix.
        else:
            self.anom.satellite_trail = results[0].satellite_trail
            self.anom.guidestar_failure = results[0].guidestar_failure

        # Don't use images with GS fails for calibration photometry. Duh.
        if self.anom.guidestar_failure:
            self.anom.use_obs_for_phot = False

            print('  Affected by guidestar failure, cannot use for photometry.')

        # Images with satellite trails should be inspected more closely later.
        else:
            self.anom.use_obs_for_phot = True

            if self.anom.satellite_trail:
                print('  WARNING: Affected by satellite trail. Use with caution.')
            else:
                print('  Unaffected by guidestar failure or satellite trails.')


    def check_margin(self, det_tbl):
        """check if source center is in the margin"""
        margin_px = 6

        xmin = ymin = margin_px
        xmax = self.data.data_corr.shape[0] - margin_px
        ymax = self.data.data_corr.shape[1] - margin_px

        in_margin = []
        for row in det_tbl:
            row_xcentroid = row['xcentroid']
            row_ycentroid = row['ycentroid']

            if xmin < row_xcentroid < xmax and ymin < row_ycentroid < ymax:
#            if (row_xcentroid > xmin) and (row_xcentroid < xmax) \
#                and (row_ycentroid > ymin) and (row_ycentroid < ymax):
                in_margin.append(False)
            else:
                in_margin.append(True)

        return in_margin, margin_px


    def correct_syn_cr(self, sources_cr):
        """not used right now"""
        print('Not in use', sources_cr)


    def compare_detected_sources_phot(self, det_tbl):
        """Compare sources to synthetic target.

        First, checks detected source(s) against the shape
        of the data array to make sure the observation(s)
        are within a certain pixel margin of the margins.
        This margin right now is set to:
            `args.annulus` + `args.dannulus`

        Any sources that are too close to the margin by
        this metric are removed from `det_tbl`. If no
        sources remain, then this function returns `None`.
        If any sources do remain, a quick round of
        photometry is performed.

        The source with the smallest percent difference
        between the measured and synthetic count rates
        should ideally be the actual source; one last check
        is performed to ensure that the measured count rate
        is within a 25% percent difference threshold. If it
        is, then this function returns the corresponding
        `det_tbl` row. Otherwise, it returns `None`.

        Currently hardcoded:
            margin_px : int
                Limit for source location relative to the
                margins of the detector (sub)array.
            cr_pd : float
                Threshold for percent difference between source
                count rate and synthetic count rate.

        Parameters
        ----------
        self : `ObsBatch`
            Staring mode observation object.
        det_tbl : `astropy.table.table.Table`
            Table with identified sources' properties.

        Returns
        -------
        source_row : `astropy.table.row.Row` or NoneType
            Row corresponding to detected source to use for
            photometry. If no detected source is inside the
            margins and within 25% of the synthetic
            count rate, then `source_row` is `None`.
        det_tbl : `astropy.table.table.Table`
            Table with identified sources' properties, and
            added columns:
                'in_margin' : str
                    Values are either True or False.
                'sources_pd' : float
                    Values are either 9999., indicating
                    that the source was too close to the
                    margin, or the percent difference
                    between the source measured count rate
                    and the synthetic count rate.
        """
        # defaults:
        det_tbl['cr_pd'] = CR_PD

        det_tbl['in_margin'], margin_px = self.check_margin(det_tbl)

        not_in_margin_tbl = det_tbl[~det_tbl['in_margin']]

        if len(not_in_margin_tbl) == 0:
            source_row = None

            print(f'  All ({len(det_tbl)}) detected source(s) are within '\
                  f'{margin_px} pixels of detector/subarray margin.\n'\
                  'Observation will not be used for photometry.')

        else:
            sources_cr = []

            for row in det_tbl:
                # To make sure length of list equals length of OG table:
                if row['in_margin']:
                    sources_cr.append(9999.)

                else:
                    #xcen = row['xcentroid']
                    #ycen = row['ycentroid']

                    det_phot_ap = CircularAperture([
                                    (row['xcentroid'], row['ycentroid'])], r=3)
                    det_sky_ap = CircularAnnulus([
                                    (row['xcentroid'], row['ycentroid'])],
                                    r_in=14, r_out=19)

                    row_phot = iraf_style_photometry(det_phot_ap, det_sky_ap,
                                    self.data.data_corr,
                                    error_array=self.data.err_arr,
                                    bg_method='median', epadu=1.)[0]
                    sources_cr.append(row_phot['flux'])

            sources_pd = [100 * np.abs(source_cr - self.syn_cr) / \
                          self.syn_cr
                          if source_cr != 9999. else 9999.
                          for source_cr in sources_cr]

            det_tbl['sources_pd'] = sources_pd

            # If the smallest percent difference between the
            # source count rate and the synthetic count rate
            # is less than 25%, then we'll use it
            if np.min(sources_pd) < CR_PD:
                source_row = det_tbl[sources_pd.index(np.min(sources_pd))]
                self.phot.centroid = (source_row['xcentroid'],
                                      source_row['ycentroid'])
                #self.phot.ycentroid = source_row['ycentroid']

                if len(det_tbl) == 1:
                    print('  1 source found.')
                else:
                    print(f'  {len(det_tbl)} total sources found.')
                print(f'    Using source at x={self.phot.centroid[0]}, '\
                      f'y={self.phot.centroid[1]}')

            else:
                source_row = None

                if len(not_in_margin_tbl) == 1:
                    message = 'Measured count rate of detected source is'
                else:
                    message = 'Measured count rate of all '\
                              f'{len(not_in_margin_tbl)} detected sources are'

                print(f"  {message} greater than {CR_PD}% different from "\
                      "synthetic target count rate.\n    "\
                      "Exposure won't be used for photometry.")

        return source_row, det_tbl


    def flt_dq_cutout(self, buffer=3):
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
        self.data.dq_buffer = buffer
        i_x = int(self.phot.centroid[0])
        i_y = int(self.phot.centroid[1])

        self.data.dq_cutout = self.data.dq_arr[(i_y - buffer):(i_y + buffer),
                                               (i_x - buffer):(i_x + buffer)]
        self.data.dq_count = np.sum(self.data.dq_cutout > 0.)


    def flt_apply_pam(self):
        """Applies pixel area map to FLT data.

        To correct for geometric distortion, FLT data
        is multiplied by the pixel area map.

        Parameters
        ----------
        self : `ObsBatch`
            Staring mode observation object.
        """
        _dy, _dx = self.data.data_arr.shape

        # from Varun's code: not sure why we do this
        self.data.ltv1 = int(-1*self.data.hdr['ltv1'])
        self.data.ltv2 = int(-1*self.data.hdr['ltv2'])

        y_bounds = (self.data.ltv2, self.data.ltv2 + _dy)
        x_bounds = (self.data.ltv1, self.data.ltv1 + _dx)

        pam_sec = PAM_ARR[y_bounds[0]:y_bounds[1], x_bounds[0]:x_bounds[1]]
        self.data.data_corr = self.data.data_arr * pam_sec


    def flt_find_sources(self, plot_dir):
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
        cr_pd : float
            Threshold for percent difference between source
            count rate and synthetic count rate. Default is
            25%.

        Returns
        -------
        source_row : `astropy.table.row.Row` or NoneType
            If at least one source is found, should return
            the row of the properties row corresponding to
            the matching source. If no viable sources is
            found, returns `None`.
        """
        # Defaults:
        nsigma = 3.0
        npixels = 15

        # Create a threshold image from the PAM-corrected data.
        threshold = detect_threshold(self.data.data_corr, nsigma=nsigma)
        # Now use the threshold image to make a segmentation map.
        segm = detect_sources(self.data.data_corr, threshold, npixels=npixels)

        # If no sources are detected, the segmentation map will be `None`.
        if segm is None:
            source_row = None
            print('No sources found.')

        else:
            det_tbl = SourceCatalog(self.data.data_corr, segm).to_table()

            source_row, det_tbl = self.compare_detected_sources_phot(det_tbl)

            if self.args.plot_sources:
                with warnings.catch_warnings():
                    # Not even sure this is doing anything to be honest.
                    warnings.filterwarnings("ignore",
                                            message="findfont: Generic family"\
                                                    " 'serif' not found because"\
                                                    " none of the following "\
                                                    "families were found: "\
                                                    "Computer Modern Roman")

                    plot_flt_sources(self, det_tbl, source_row, plot_dir)

        return source_row


    def check_quality(self, syn_cr, plot_dir):
        """
        Checks if image meets all four criteria.
        """
        self.syn_cr = syn_cr
        condition1 = self.data.hdr['scan_typ'] == 'N'

        if condition1:
            # Don't use GS fails, warn for any satellite trails.
            self.check_for_anomalies()

            # Second condition is that there are no QL flags that
            # indicate this observation can't be used for photometry.
            # Technically this is redundant.
            condition2 = self.anom.use_obs_for_phot is True

            # Check DQ array from FITS file. This is the basis
            # of the third condition.
            condition3 = not isinstance(self.data.dq_arr, type(None))

            # Apply pixel area map to correct geometric distortion.
            self.flt_apply_pam()

            # Find sources in corrected data.
            # TK May need to build in iterative re-scaling of `npixels` for N bands.
            self.phot.source_row = self.flt_find_sources(plot_dir) # TKTK

            # Fourth condition is that at least one matching source
            # is identified. `self.source_row` will be `None` if no
            # viable sources can be found.
            #if not isinstance(obs_img.phot.source_row, type(None)):
            #    condition4 = True
            #else:
            #    condition4 = False
            condition4 = not isinstance(self.phot.source_row, type(None))

            if not condition2:
                print(f'{" "*4}- Observation affected by guidestar failure.')

            if not condition4:
                print(f'{" "*4}- No viable sources detected.')

            if not condition3:
                print(f'{" "*4}- DQ array is empty.')

        else:
            print(f'{" "*4}- Spatial scan')

        return condition1 and condition2 and condition3 and condition4


    def get_obs_syn_crs(self, fluxes, photutils_flux):
        """
        get observed-to-synthetic count rate ratios
        """
        obs_syn_crs = []
        for flux in fluxes:
            obs_syn_crs.append(flux / self.syn_cr)

        obs_syn_crs.append(photutils_flux / self.syn_cr)

        return obs_syn_crs


    def make_phot_row(self):
        """
        do photometry on one image
        """
        self.phot.centroid = [self.phot.source_row['xcentroid'],
                              self.phot.source_row['ycentroid']]

        # Check the data quality flags.
        self.flt_dq_cutout()

        # Create aperture/annulus objects.
        self.phot.phot_ap = CircularAperture(
            [(self.phot.centroid[0], self.phot.centroid[1])],
            r=self.args.radius)

        self.phot.sky_ap = CircularAnnulus(
            [(self.phot.centroid[0], self.phot.centroid[1])],
            r_in=self.args.annulus,
            r_out=self.args.annulus + self.args.dannulus)

        # Make first pass fit with RadialProfile,
        # recenter source and re-fit.
        prof = RadialProfile(self.phot.centroid[0],
                    self.phot.centroid[1],
                    self.data.data_corr,
                    recenter=True, fit=False, r=1)

        self.phot.recentered = [prof.x, prof.y]
        prof = RadialProfile(self.phot.recentered[0],
                    self.phot.recentered[1],
                    self.data.data_corr,
                    recenter=False, fit=True, r=2)


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
            iraf_row = iraf_style_photometry(self.phot.phot_ap,
                            self.phot.sky_ap,
                            self.data.data_corr,
                            error_array=self.data.err_arr,
                            bg_method=method, epadu=1.)[0]
            method_fluxes.append(iraf_row['flux'])
            method_flux_errs.append(iraf_row['flux_error'])

        # TO DO: use wrapper to display output
        photutils_sum = aperture_photometry(
                            apertures=self.phot.phot_ap,
                            data=self.data.data_corr)\
                            ['aperture_sum'][0]

        # Get statistics/measurements for background annulus.
        # TO DO: rethink parameter naming for this one?
        bg_stats = make_aperture_stats_tbl(self.data.data_corr, self.phot.sky_ap)[0]

        obs_syn_crs = self.get_obs_syn_crs(method_fluxes, photutils_sum)

        # Have to revert LTV_ to original form to calc detx and dety
        self.phot.det_coords = [self.phot.centroid[0] \
                                - (self.data.ltv1 / -1),
                                self.phot.centroid[1] \
                                - (self.data.ltv2 / -1)]

        phot_row = [self.path, self.args.radius, self.args.annulus,
                    self.args.dannulus, self.args.back_method,
                    self.anom.satellite_trail]

        phot_row.extend([val for key, val in self.data.hdr.items()])

        phot_row.extend([self.phot.centroid[0], self.phot.centroid[1],
                         self.phot.recentered[0], self.phot.recentered[1],
                         prof.fwhm, prof.chisquared,
                         self.data.ltv1, self.data.ltv2,
                         self.phot.det_coords[0], self.phot.det_coords[1],
                         bg_stats['aperture_median'],
                         bg_stats['aperture_mode'],
                         bg_stats['aperture_mean'],
                         bg_stats['aperture_std'],
                         bg_stats['aperture_nonnan_area'],
                         iraf_row['phot_ap_area'],
                         method_fluxes[0], method_flux_errs[0],
                         method_fluxes[1], method_flux_errs[1],
                         method_fluxes[2], method_flux_errs[2],
                         self.syn_cr,
                         obs_syn_crs[0],  obs_syn_crs[1],
                         obs_syn_crs[2], photutils_sum,
                         obs_syn_crs[3], np.std(self.data.data_corr),
                         self.phot.dq_count])

        phot_row.extend([correct_tds(flux, self.data.hdr['expstart'],
                                     self.data.hdr['filter'])
                         for flux in method_fluxes])

        return phot_row



def cal_ir_monitor_calspec(args, dirs):
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

    if args.run_ap_phot or args.helium or args.linearity:
        run_process(args, dirs, write=True, overwrite=True)


if __name__ == '__main__':
    # Parse command line arguments.
    parsed_args = parse_args()

    # Display command line arguments.
    display_args(parsed_args)

    # Set up needed directories.
    run_dirs = initialize_directories(parsed_args)

    # Showtime.
    cal_ir_monitor_calspec(parsed_args, run_dirs)
