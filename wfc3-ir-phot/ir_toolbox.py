"""
Assorted tools for the IR staring mode standard star pipeline.

Usage
-----
    This module is designed to be imported into the
    `ir_phot_pipeline.py` script. The individual functions
    and global variables can also be imported separately.

        > import ir_phot_toolbox
        > from ir_phot_toolbox import make_timestamp
        > from ir_phot_toolbox import MONITOR_DIR

Classes
-------
InteractiveArgs(name, verbose, log, trial,
                get_new_data, redownload, drizzle, storm,
                run_ap_phot, proposals, targets, filters,
                file_type, radius, annulus, dannulus, back_method,
                ap_phot_drz, ap_phot_flt, write_dir)
    Class for handling pipeline settings in an interactive
    mode, like in a Jupyter notebook.


Functions
---------
    display_args(args)
        Prints and/or logs `args` values.
    get_decimalyear(mjd)
        Converts MJD to `decimalyear` format.
    parse_args()
        Parses `ir_phot_monitor.py` command line arguments.
    resolve_targnames(targname, simplify, verbose, log)

Author
------
Mariarosa Marinelli, 2023
"""
import os
from argparse import ArgumentParser
from astropy.time import Time

from ir_logging import (check_preexisting_logging, display_message,
                       make_timestamp, setup_logging, MONITOR_DIR)

MONITOR_PROGRAMS = [11451, 11552, 11903, 11926, 11936, 12333,
                    12334, 12357, 12698, 12699, 12702, 13088, 13089,
                    13092, 13094, 13573, 13575, 13576, 13579, 13711,
                    14021, 14024, 14384, 14386, 14544, 14883, 14992,
                    14994, 15113, 15582, 16030, 16415, 16579, 17015]  # removed 11557
PAM = os.path.join(os.path.dirname(__file__), 'ir_wfc3_map.fits')
SIMPLE_TARGS = ['GD153', 'GRW70', 'GD71', 'P330E', 'G191B2B']

def make_phot_cols(hdr, dq_buffer):
    """Create batch-specific photometry column names.

    Parameters
    ----------
    hdr : dict
    dq_buffer : int

    Returns
    -------
    phot_cols : list
    """
    phot_cols = ['exposure_file', 'file_type', 'radius',
                 'annulus', 'dannulus', 'back_method',
                 'satellite_trail']  # 7 columns

    phot_cols.extend(list(hdr.keys()))

    phot_cols.extend(['x', 'y', 'recentered_x', 'recentered_y',
                      'RadialProfile.fwhm', 'Radial_Profile.chisquared',
                      'ltv1_adjusted', 'ltv2_adjusted', 'detx', 'dety',
                      'median_bg', 'mode_bg', 'mean_bg', 'std_bg',
                      'aperture_nonnan_area', 'phot_ap_area',
                      'mean_mag_flux', 'mean_flux_error',
                      'median_mag_flux', 'median_flux_error',
                      'mode_mag_flux', 'mode_flux_error',
                      'syn_mag', 'syn_cr',
                      'mean_obs_syn_cr', 'median_obs_syn_cr', 'mode_obs_syn_cr',
                      'photutils_sum', 'obs_syn_photutils',
                      'image_std', f'dq_count_{dq_buffer}'])

    return phot_cols


def display_args(args):
    """Prints and/or logs `args` values.

    Method displays arguments for an `InteractiveArgs`
    or `ArgParse` object.

    Usage
    -----
    Can either be called directly, ex:

            if __name__ == '__main__':
                args_to_display = parse_args()
                display_args(args_to_display)

    or called indirectly as a wrapper method of an
    `InteractiveArgs` object:

            notebook_args = InteractiveArgs()
            notebook_args.interactive_display_args()

    Parameter
    ---------
    args : `InteractiveArgs` or `ArgParse` object
        Arguments entered either at the command line (for
        an `ArgParse` object) or in a Jupyter notebook (for
        an `InteractiveArgs` object).
    """
    header = ["ARGUMENT", "VALUE"]
    display_message(verbose=args.verbose,
                    log=args.log,
                    log_type='info',
                    message='')
    display_message(verbose=args.verbose,
                    log=args.log,
                    log_type='info',
                    message=f'{header[0]:15} {header[1]:15}')
    display_message(verbose=args.verbose,
                    log=args.log,
                    log_type='info',
                    message=f'{len(header[0])*"-":15} {len(header[1])*"-":15}')
    for prop, val in vars(args).items():
        display_message(verbose=args.verbose,
                        log=args.log,
                        log_type='info',
                        message=f'{prop:15} {val}')



def get_decimalyear(mjd):
    """Converts MJD to `decimalyear` format.

    Helper function to convert a Modified Julian Date into
    the decimalyear format.

    Parameter
    ---------
    mjd : int or float
        Some date/time in MJD format.

    Returns
    -------
    decimalyear : float
        The date/time in decimalyear format.
    """
    decimalyear = Time(mjd, format='mjd').to_value(format='decimalyear')
    return decimalyear


def parse_args():
    """Parses `ir_phot_monitor.py` command line arguments.

    Parses command line arguments for the IR standard star
    monitor pipeline. In addition to the default `--help`
    flag, there are a total of 22 configurable arguments:
        4 pipeline settings
        5 pipeline flags
        13 pipeline parameters

    Pipeline Settings
    -----------------
    name : str
        Name of the directory for this pipeline run in
        central store. Defaults to the timestamp returned
        by `make_timestamp()`.
    trial : Boolean
        Run in 'trial' mode. Defaults to False if pipeline
        is run from command line (opposite default when
        pipeline is run through an `InteractiveArgs`
        object).
    verbose : Boolean
        Whether to print the message; defaults to False.
    log : Boolean
        Whether to log the message; defaults to False.

    Pipeline Execution Flags
    ------------------------
    get_new_data : Boolean
        If set, will query MAST and download data. If set
        to `True` and `redownload` is set to `False`,
        will only download new data. Default is `False`.
    redownload : Boolean
        If both this and `get_new_data` are set to `True`,
        then all data matching the search parameters will
        be downloaded. Previously-existing files will be
        overwritten. Default is `False`.
    drizzle : Boolean
        If set, will drizzle FLT files. Default is `False`.
        If this is set to `True` and `storm` is set to
        `False`, pipeline will only drizzle FLTs if there
        does not already exist a corresponding DRZ file in
        the same directories.
    storm : Boolean
        If both this and `drizzle` are set to `True`, will
        drizzle all FLTs, overwriting previously-drizzled
        files. Default is `False`.
    run_ap_phot : Boolean
        When set, indicates that aperture photometry should
        be calculated. Default is `False`.

    Pipeline Parameters
    -------------------
    proposals : str or list of str
        Program ID(s) to examine. If none are provided,
        this will default to the list of all IR staring
        mode calibration programs, `MONITOR_PROGRAMS`.
    targets : str or list of str
        Target(s) to select. If `download_new_data` is set
        to `True`, will download data for these targets;
        names will be resolved such that providing `GRW70`
        will also download data from MAST matching
        `GRW+70D` and `GRW+70D5824`. Otherwise, will only
        process (drizzle and/or perform photometry) data in
        the defined directory for given targets. If no
        targets are defined, all targets available will be
        processed.
    filters : str or list of str
        WFC3/IR filters to select. If `download_new_data`
        is set to `True`, will only download data in these
        filters. Otherwise, will only process (drizzle
        and/or perform photometry) data in the defined
        directory in these filters. If no filters are
        defined, all filters available will be processed.
    file_type : str
        File type with which to begin pipeline operations.
        Possible options are `flt` or `drz`; defaults to
        `flt`.
    radius : int
        Radius, in pixels, of the photometric aperture;
        defaults to 3.
    annulus: int
        Inner radius, in pixels, of the background annulus;
        default is 14.
    dannulus : int
        Width, in pixels, of the background annulus;
        default is 5.
    back_method : str
        Method to calculate the background from the sigma-
        clipped data. Options are `mean`, `median`, and
        `mode`; defaults to `median`. AKA `salgorithm`.
    ap_phot_drz : Boolean
        When set, indicates that aperture photometry should
        be performed on DRZ files. Default is False.
    ap_phot_flt : Boolean
        When set, indicates that aperture photometry should
        be performed on FLT files. Default is False.
    write_dir : str
        String representation of the directory where the
        photometry tables and/or detection plots should be
        written. Defaults to the `output` subdirectory in
        `MONITOR_DIR`.
    plot_sources : Boolean
        When set, creates and saves the source detection/
        selection plots. Default is False.
    helium_corr : Boolean
        When set, (re)downloads RAWs, corrects for TVB from
        Helium I, runs `calwf3`, and produces FLTs that
        have been corrected for helium. Only applicable for
        F105W and F110W. Default is False.

    Returns
    -------
    args : `argparse.Namespace`
        Namespace class object that has as attributes the
        20 configurable arguments.
    """
    parser = ArgumentParser(prog='ir_phot_pipeline',
                            description='WFC3/IR standard staring mode  '\
                                        'photometry monitor pipeline',
                            epilog = 'Author: Mariarosa Marinelli')

    # Settings:
    parser.add_argument("-n", "--name",
                        help="name for pipeline run/log; defaults to timestamp",
                        default=make_timestamp())
    parser.add_argument("-t", "--trial",
                        help="when set, runs pipeline in `trial` mode",
                        action="store_true")
    parser.add_argument("--local",
                        help='when set, runs pipeline to download stuff locally',
                        action='store_true')
    parser.add_argument("-v", "--verbose",
                        help="when set, prints statements to command line",
                        action="store_true")
    parser.add_argument("-l", "--log",
                        help="when set, logs statements to log file",
                        action="store_true")

    # Execution Flags:
    parser.add_argument("-g", "--get_new_data",                                 # keep the same
                        help="when set, get new data",
                        action='store_true')
    parser.add_argument("-r", "--redownload",                                   # new
                        help="when set, redownload data",
                        action='store_true')
    parser.add_argument("-d", "--drizzle",                                      # new
                        help="when set, drizzle FLT files",
                        action='store_true')
    parser.add_argument("-s", "--storm",                                        # new
                        help="when set, redrizzle FLT files",
                        action='store_true')
    parser.add_argument("-a", "--run_ap_phot",                                  # keep the same
                        help="when set, run aperture photometry",
                        action='store_true')

    # Pipeline Parameters:
    parser.add_argument("--proposals",
                        help="calibration proposal or list of proposals",
                        nargs="+",
                        type=int,
                        default=MONITOR_PROGRAMS)
    parser.add_argument("--targets",
                        help="target or list of targets (default is 'all')",
                        nargs="+",
                        default="all")
    parser.add_argument("--filters",
                        help="filter or list of filters (default is 'all')",
                        nargs="+",
                        default='all')
    parser.add_argument("--file_type",                                          # changed fcr to drz
                        help="file type to begin with (flt or drz)",
                        default="flt",
                        choices=["flt", "drz"])
    parser.add_argument("--radius",                                             # new
                        help="photometric aperture radius (pixels)",
                        type=int,
                        default=3)
    parser.add_argument("--annulus",                                            # new
                        help="inner radius of the background annulus (pixels)",
                        type=int,
                        default=14)
    parser.add_argument("--dannulus",
                        help="width of the background annulus (pixels)",        # new
                        type=int,
                        default=5)
    parser.add_argument("--back_method",                                        # new
                        help="method to calculate background from sigma-clipped data",
                        default="median",
                        choices=["mean", "median", "mode"])
    parser.add_argument("--ap_phot_drz",                                        # updated
                        help='when set, perform aperture photometry on DRZs',
                        action='store_true')
    parser.add_argument("--ap_phot_flt",                                        # keep the same
                        help='when set, perform aperture photometry on FLTs',
                        action='store_true')
    parser.add_argument("-w", "--write_dir",
                        help="directory where tables/plots should be saved",
                        default=os.path.join(MONITOR_DIR, 'output'))
    parser.add_argument("--plot_sources",
                        help="when set, save source detection/selection plots",
                        action='store_true')
    parser.add_argument("--helium_corr",
                        help="when set, correct for He I in F105W & F110W",
                        action='store_true')
    parser.add_argument("--update_refs",
                        help="when set, run bestrefs",
                        action='store_true')

    args = parser.parse_args()

    return args


def resolve_targnames(targname, simplify=True, verbose=True, log=False):
    """
    Helper functions to resolve target names. Sometimes
    what is put into APT is not the simplest form of a
    target's name so this helps make sure that everything
    is consistent. Or, in the case of searching MAST, we
    want all possible versions of the target name.

    Parameter
    ---------
    targname : string
        Name of target.
    simplify : Boolean
        When set to `True`, finds the simplified version of
        the input target name. If set to `False`, returns a
        list of all possible/accepted names for the input
        target name.
    verbose : Boolean
        Whether to print the message.
    log : Boolean
        Whether to log the message.

    Returns
    -------
    resolved : string or list of str
        Resolved target name(s). If unable to resolve the
        name (ex. it's a different target altogether, or a
        weird spelling), the original input `targname` will
        be returned instead. In `simplify` mode, it will
        return a string of the simplest version of the
        input. If not in `simplify` mode, it will return a
        list of possible names for searching in MAST (list
        may only contain one item).
    """
    targnames = {'GD153': 'GD153',
                 'GD-153': 'GD153',
                 'GRW+70D5824': 'GRW70',
                 'GRW+70D': 'GRW70',
                 'GRW70': 'GRW70',
                 'GD71': 'GD71',
                 'GD-71': 'GD71',
                 'P330E': 'P330E',
                 'GSC-02581-02323': 'P330E',
                 'G191B2B': 'G191B2B'}
    if simplify:
        try:
            resolved_targname = targnames[targname]
            resolved = resolved_targname
        except KeyError as key_error:
            display_message(verbose=verbose,
                            log=log,
                            message=f'Unable to resolve name for {key_error}',
                            log_type='warning')
            resolved = targname

    else:
        resolved_targnames = [k for k, v in targnames.items() if v == targname]
        if len(resolved_targnames) > 0:
            resolved = resolved_targnames
        else:
            resolved = targname

    return resolved


#class FontFilter(logging.Filter):
#    def filter(self, record):
#        if record.getMessage().startswith('findfont'):
#            return False



class InteractiveArgs:
    """Interactive pipeline arguments class.

    A class to enable easy handling of arguments in a
    parallel manner to argument parsing with `arg_parse()`.
    There are a total of 20 configurable arguments:
        4 pipeline settings
        6 pipeline flags
        10 pipeline parameters

    Pipeline Settings
    -----------------
    name : str
        Name of the directory for this pipeline run in
        central store. Defaults to the timestamp returned
        by `make_timestamp()`.
    trial : Boolean
        Run in 'trial' mode. Defaults to False if pipeline
        is run from command line (opposite default when
        pipeline is run through an `InteractiveArgs`
        object).
    verbose : Boolean
        Whether to print the message; defaults to True.
    log : Boolean
        Whether to log the message; defaults to False.

    Pipeline Execution Flags
    ------------------------
    get_new_data : Boolean
        If set, will query MAST and download data. If set
        to `True` and `redownload` is set to `False`,
        will only download new data. Default is `False`.
    redownload : Boolean
        If both this and `get_new_data` are set to `True`,
        then all data matching the search parameters will
        be downloaded. Previously-existing files will be
        overwritten. Default is `False`.
    drizzle : Boolean
        If set, will drizzle FLT files. Default is `False`.
        If this is set to `True` and `storm` is set to
        `False`, pipeline will only drizzle FLTs if there
        does not already exist a corresponding DRZ file in
        the same directories.
    storm : Boolean
        If both this and `drizzle` are set to `True`, will
        drizzle all FLTs, overwriting previously-drizzled
        files. Default is `False`.
    run_ap_phot : Boolean
        When set, indicates that aperture photometry should
        be calculated. Default is `False`.
    show_ap_plot : Boolean
        When set, indicates that detection plots should be
        shown. Default is `False`.

    Pipeline Parameters
    -------------------
    proposals : str or list of str
        Program ID(s) to examine. If none are provided,
        this will default to the list of all IR staring
        mode calibration programs, `MONITOR_PROGRAMS`.
    targets : str or list of str
        Target(s) to select. If `download_new_data` is set
        to `True`, will download data for these targets;
        names will be resolved such that providing `GRW70`
        will also download data from MAST matching
        `GRW+70D` and `GRW+70D5824`. Otherwise, will only
        process (drizzle and/or perform photometry) data in
        the defined directory for given targets. If no
        targets are defined, all targets available will be
        processed.
    filters : str or list of str
        WFC3/IR filters to select. If `download_new_data`
        is set to `True`, will only download data in these
        filters. Otherwise, will only process (drizzle
        and/or perform photometry) data in the defined
        directory in these filters. If no filters are
        defined, all filters available will be processed.
    file_type : str
        File type with which to begin pipeline operations.
        Possible options are `flt` or `drz`; defaults to
        `flt`.
    radius : int
        Radius, in pixels, of the photometric aperture;
        defaults to 3.
    annulus: int
        Inner radius, in pixels, of the background annulus;
        default is 14.
    dannulus : int
        Width, in pixels, of the background annulus;
        default is 5.
    back_method : str
        Method to calculate the background from the sigma-
        clipped data. Options are `mean`, `median`, and
        `mode`; defaults to `median`. AKA salgorithm.
    ap_phot_drz : Boolean
        When set, indicates that aperture photometry should
        be performed on DRZ files. Default is False.
    ap_phot_flt : Boolean
        When set, indicates that aperture photometry should
        be performed on FLT files. Default is False.
    write_dir : str
        String representation of the directory where the
        photometry tables and/or detection plots should be
        written. Defaults to the current working directory,
        since this class is intended to run interactively.
    plot_sources : Boolean
        When set, creates and saves the source detection/
        selection plots. Default is False.
    helium_corr : Boolean
        When set, (re)downloads RAWs, corrects for TVB from
        Helium I, runs `calwf3`, and produces FLTs that
        have been corrected for helium. Only applicable for
        F105W and F110W. Default is False.

    Notes
    -----
    There has got to be an easier way to set this up.

    Methods
    -------
    interactive_logging(local, log_dir)
        Enables logging when running pipeline in notebook;
        only needs to be run once per kernel and will warn
        user otherwise.
    interactive_display_args()
        Wrapper for `display_args()` function; displays
        (either in the output cell or printed to the log or
        both) the attribute names and values for the
        `InteractiveArgs` object.
    """
    def __init__(self,
                 name=make_timestamp(),
                 verbose=True,
                 log=False,
                 trial=True,
                 write_dir=os.getcwd(),
                 get_new_data=False,
                 redownload=False,
                 drizzle=False,
                 storm=False,
                 run_ap_phot=False,
                 proposals=MONITOR_PROGRAMS,
                 targets="all",
                 filters="all",
                 file_type="flt",
                 radius=3,
                 annulus=14,
                 dannulus=5,
                 back_method='median',
                 ap_phot_drz=False,
                 ap_phot_flt=False,
                 plot_sources=False,
                 helium_corr=False):
        self.name = name
        self.verbose = verbose
        self.log = log
        self.trial = trial
        self.write_dir = write_dir
        self.get_new_data = get_new_data
        self.redownload = redownload
        self.drizzle = drizzle
        self.storm = storm
        self.run_ap_phot = run_ap_phot
        self.proposals = proposals

        if targets == 'all':
            targets = None      # so none are filtered
        self.targets = targets

        if filters == 'all':
            filters = None      # so none are filtered
        self.filters = filters

        self.file_type = file_type
        self.radius = radius
        self.annulus = annulus
        self.dannulus = dannulus
        self.back_method = back_method
        self.ap_phot_drz = ap_phot_drz
        self.ap_phot_flt = ap_phot_flt
        self.plot_sources = plot_sources
        self.helium_corr = helium_corr

    def interactive_logging(self, local=True,
                            log_dir=os.getcwd()):
        """
        Method to enable logging when running the pipeline
        interactively. Note that this only needs to be run
        one time per session. If it is run multiple times,
        it will warn the user that the logging is already
        configured to a specific file location.

        Parameters
        ----------
        self : `InteractiveArgs` object
            Self.
        local : Boolean
            Whether to save the log locally (default) or to
            central storage log location.
        log_dir : str or path-like
            String representation or path to location where
            log should be saved. If no path is specified
            and `local` is set to `True`, then log will
            save to current working directory.
        """
        self.log = True

        if not check_preexisting_logging():
            setup_logging(local=local,
                          log_dir=log_dir,
                          log_name=self.name)


    def interactive_display_args(self):
        """Method wrapper for `display_args()` function.

        Parameter
        ---------
        self : `InteractiveArgs` object
            Self, passed as `args` parameter to
            `display_args()` function.
        """
        display_args(args=self)
