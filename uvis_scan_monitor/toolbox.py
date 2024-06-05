import argparse
from argparse import ArgumentParser
from datetime import datetime
import logging
import os
import sys

SCAN_PROGRAMS = [14878, 15398, 15583, 16021, 16416, 16580, 17016]
core_filters = ['F218W', 'F225W', 'F275W', 'F336W', 'F438W', 'F606W', 'F814W']
core_targets = ['GD153', 'GRW70']

class CaptureOutput(list):
    """
    Class to capture output from externally-imported
    functions.
    """
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()

        return self

    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio
        sys.stdout = self._stdout

def check_subdirectory(parent_dir, sub_name, verbose=True, log=False):
    """
    Helper function to check if a subdirectory exists.
    If it doesn't exist, the subdirectory is created.

    Parameters
    ----------
    parent_dir : str or path-like
        String representation of path or path-like object
        of parent directory (can be either relative or
        absolute path).
    sub_name : str
        Name of subdirectory.
    verbose : Boolean
    log : Boolean


    Returns
    -------
    sub_dir : str or None
        String representation of path to subdirectory.
        Returns `None` if parent directory does not exist.
    """
    sub_dir = os.path.join(parent_dir, sub_name)
    if os.path.exists(parent_dir):
        if os.path.exists(sub_dir):
            display_message(verbose=verbose,
                            log=log,
                            log_type='info',
                            message=f'Found existing directory at {sub_dir}')
        else:
            display_message(verbose=verbose,
                            log=log,
                            log_type='info',
                            message=f'Making new directory at {sub_dir}...')
            os.mkdir(sub_dir)

        return sub_dir

    else:
        display_message(verbose=args.verbose,
                        log=args.log,
                        log_type='critical',
                        message=f'Nonexistent parent directory: {parent_dir}\n'\
                                f'Cannot make new directory at {sub_dir}')
        return None


def make_timestamp():
    """
    Helper function to convert and format current datetime
    into a string. This string is then used for the name
    of the pipeline run directory in the scan monitor
    photometry central store location:
        /grp/hst/wfc3v/wfc3photom/data/uvis_scan_monitor

    Returns
    -------
    timestamp : str
        String representation of current time, in format
        `YYYY-MM-DD_hh-mm-ss`.
    """
    now = str(datetime.now()).split(' ')
    date = now[0]
    time = now[1].split('.')[0].replace(':', '-')
    timestamp = f'{date}_{time}'
    return timestamp


class InteractiveArgs:
    """
    A class to enable easy handling of arguments in a
    parallel manner to argument parsing with `arg_parse()`.
    There are a total of 19 configurable arguments:
        4 pipeline settings
        6 pipeline flags
        10 pipeline parameters

    Pipeline Settings
    -----------------
    name : str
        Name of the directory for this pipeline run in
        central store. Defaults to the timestamp returned
        by `make_timestamp()`.
    verbose : Boolean
        Defaults to False.
    log : Boolean
        Defaults to False.
    trial : Boolean
        Run in 'trial' mode. Defaults to True if pipeline
        is run interactively (opposite from default
        behavior when pipeline is run from command line).

    Pipeline Flags
    --------------
    get_new_data : Boolean
    redownload_data : Boolean
    run_cr_reject : Boolean
    reprocess_fcr : Boolean
    run_ap_phot : Boolean
    show_ap_plot : Boolean

    Pipeline Parameters
    -------------------
    proposals : str or list of str
        Program ID(s) to examine. If none are provided,
        this will default to the list of all UVIS scan
        calibration programs, `scan_programs`.
    targets : str or list of str
        Target(s) to select. If `download_new_data` is set
        to `True`, will download data for these targets;
        nameswill be resolved such that providing `GRW70`
        will also download data from MAST matching
        `GRW+70D` and `GRW+70D5824`. Otherwise, will only
        process (CR reject and/or perform photometry) data
        in the defined directory for given targets. If no
        targets are defined, the list of standard stars for
        the UVIS scan calibration programs, `core_targets`,
        will be used.
    filters : str or list of str
        WFC3/UVIS filters to select. If `download_new_data`
        is set to `True`, will only download data in these
        filters. Otherwise, will only process (CR reject
        and/or perform photometry) data in the defined
        directory in these filters. If no filters are
        defined, the list of core filters for the UVIS scan
        calibration programs, `core_filters`, will be used.
    file_type : str
        File type with which to begin pipeline operations.
        Possible options are `flt` or `fcr`; defaults to
        `flt`.
    ap_dim : tuple
        Dimenions of photometric aperture; defaults to
        (44, 268).
    sky_ap_dim : tuple
        Inner dimensions of sky background rind; defaults
        to (300, 400).
    sky_thickness : int
        Thickness of sky background rind in pixels;
        defaults to 30.
    back_method : str
        Which method to use for sky background calculation.
        Possible options are `median` and `mean`; defaults
        to `median`.
    ap_phot_fcr : Boolean
        Defaults to False.
    ap_phot_flt : Boolean
        Defaults to False.

    Notes
    -----
    There has got to be an easier way to set this up.
    """
    def __init__(self,
                 name=make_timestamp(),
                 verbose=True,
                 log=False,
                 trial=True,
                 get_new_data=False,
                 redownload_data=False,
                 run_cr_reject=False,
                 reprocess_fcr=False,
                 run_ap_phot=False,
                 show_ap_plot=False,
                 proposals=scan_programs,
                 targets="core",
                 filters="core",
                 file_type="flt",
                 ap_dim=[44, 268],
                 sky_ap_dim=[300, 400],
                 sky_thickness=30,
                 back_method='median',
                 ap_phot_fcr=False,
                 ap_phot_flt=False):
        self.name = name
        self.verbose = verbose
        self.log = log
        self.get_new_data = get_new_data
        self.redownload_data = redownload_data
        self.run_cr_reject = run_cr_reject
        self.reprocess_fcr = reprocess_fcr
        self.run_ap_phot = run_ap_phot
        self.show_ap_plot = show_ap_plot
        self.proposals = proposals

        if targets == 'core':
            targets = core_targets
        self.targets = targets

        if filters == 'core':
            filters = core_filters
        self.filters = filters

        self.file_type = file_type
        self.ap_dim = ap_dim
        self.sky_ap_dim = sky_ap_dim
        self.sky_thickness = sky_thickness
        self.back_method = back_method
        self.ap_phot_fcr = ap_phot_fcr
        self.ap_phot_flt = ap_phot_flt

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
        if not check_preexisting_logging():
            self.log = True
            setup_logging(local=local,
                          log_dir=log_dir,
                          log_name=self.name)


    def interactive_display_args(self):
        """
        Method wrapper for `display_args()` function.

        Parameter
        ---------
        self : `InteractiveArgs` object
            Self, passed as `args` parameter to
            `display_args()` function.
        """
        display_args(args=self)

def display_args(args):
    """
    Method displays arguments for an `InteractiveArgs`
    or `ArgParse` object, since the object is initialized
    with default settings.

    Use
    ---
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
    self : `InteractiveArgs` or `ArgParse` object
    """
    header = ["ARGUMENT", "VALUE"]
    display_message(verbose=args.verbose,
                    log=args.log,
                    log_type='info',
                    message=f'{header[0]:15} {header[1]:15}')
    for prop, val in vars(args).items():
        display_message(verbose=args.verbose,
                        log=args.log,
                        log_type='info',
                        message=f'{prop:15} {val}')


def parse_args():
    """
    Parses command line arguments for the UVIS scan monitor
    pipeline. In addition to the default `--help` flag,
    there are a total of 19 configurable arguments:
        4 pipeline settings
        6 pipeline flags
        10 pipeline parameters

    Pipeline Settings
    -----------------
    name : str
        Name of the directory for this pipeline run in
        central store. Defaults to the timestamp returned
        by `make_timestamp()`.
    verbose : Boolean
        Defaults to False.
    log : Boolean
        Defaults to False.
    trial : Boolean
        Run in 'trial' mode. Defaults to False if pipeline
        is run from command line (opposite default when
        pipeline is run through an `InteractiveArgs`
        object).

    Pipeline Execution Flags
    ------------------------
    get_new_data : Boolean
    redownload_data : Boolean
    run_cr_reject : Boolean
    reprocess_fcr : Boolean
    run_ap_phot : Boolean
    show_ap_plot : Boolean

    Pipeline Parameters
    -------------------
    proposals : str or list of str
        Program ID(s) to examine. If none are provided,
        this will default to the list of all UVIS scan
        calibration programs, `scan_programs`.
    targets : str or list of str
        Target(s) to select. If `download_new_data` is set
        to `True`, will download data for these targets;
        nameswill be resolved such that providing `GRW70`
        will also download data from MAST matching
        `GRW+70D` and `GRW+70D5824`. Otherwise, will only
        process (CR reject and/or perform photometry) data
        in the defined directory for given targets. If no
        targets are defined, the list of standard stars for
        the UVIS scan calibration programs, `core_targets`,
        will be used.
    filters : str or list of str
        WFC3/UVIS filters to select. If `download_new_data`
        is set to `True`, will only download data in these
        filters. Otherwise, will only process (CR reject
        and/or perform photometry) data in the defined
        directory in these filters. If no filters are
        defined, the list of core filters for the UVIS scan
        calibration programs, `core_filters`, will be used.
    file_type : str
        File type with which to begin pipeline operations.
        Possible options are `flt` or `fcr`; defaults to
        `flt`.
    ap_dim : tuple
        Dimenions of photometric aperture; defaults to
        (44, 268).
    sky_ap_dim : tuple
        Inner dimensions of sky background rind; defaults
        to (300, 400).
    sky_thickness : int
        Thickness of sky background rind in pixels;
        defaults to 30.
    back_method : str
        Which method to use for sky background calculation.
        Possible options are `median` and `mean`; defaults
        to `median`.
    ap_phot_fcr : Boolean
        Whether to run aperture photometry on FCR data.
        Defaults to False.
    ap_phot_flt : Boolean
        Whether to run aperture photometry on FLT data.
        Defaults to False.

    Returns
    -------
    args : `argparse.Namespace`
        Namespace class object that has as attributes the
        19 configurable arguments.
    """
    parser = ArgumentParser(prog='uvis_scan_monitor',
                            description='WFC3/UVIS calibration scan '\
                                        'photometry monitor pipeline',
                            epilog = 'Author: Mariarosa Marinelli')

    # Settings:
    parser.add_argument("-t", "--trial",
                        help="when set, runs pipeline in `trial` mode",
                        action="store_true")
    parser.add_argument("-v", "--verbose",
                        help="when set, prints statements to command line",
                        action="store_true")
    parser.add_argument("-l", "--log",
                        help="when set, logs statements to log file",
                        action="store_true")
    parser.add_argument("-n", "--name",
                        help="name for pipeline run; defaults to timestamp",
                        default=make_timestamp())

    # Execution Flags:
    parser.add_argument("-g", "--get_new_data",
                        help="when set, get new data",
                        action='store_true')
    parser.add_argument("-r", "--redownload_data",                              # new
                        help="when set, redownload data",
                        action='store_true')
    parser.add_argument("-c", "--run_cr_reject",
                        help="when set, run cosmic ray rejection",
                        action='store_true')
    parser.add_argument("-p", "--reprocess_fcr",                                # updated short flag
                        help="when set, reprocess existing FCR files",
                        action='store_true')
    parser.add_argument("-a", "--run_ap_phot",
                        help="when set, run aperture photometry",
                        action='store_true')
    parser.add_argument("-s", "--show_ap_plot",
                        help="when set, show source detection plots",
                        action='store_true')

    # Pipeline Parameters:
    parser.add_argument("--proposals",
                        help="calibration proposal or list of proposals",
                        nargs="+",
                        default=SCAN_PROGRAMS)
    parser.add_argument("--targets",
                        help="target or list of targets (default is 'core')",
                        nargs="+",
                        default="core")
    parser.add_argument("--filters",
                        help="filter or list of filters (default is 'core')",
                        nargs="+",
                        default="core")

    parser.add_argument("--file_type",
                        help="file type to begin with (flt or fcr)",
                        default="flt",
                        choices=["flt", "fcr"])
    parser.add_argument("--ap_dim",
                        help="photometric aperture dimensions: x_px y_px",
                        nargs=2,
                        type=int,
                        default=[44, 268])
    parser.add_argument("--sky_ap_dim",
                        help="sky rind inner dimensions: x_px y_px "\
                                "(default is 300px wide and 400px tall)",
                        nargs=2,
                        type=int,
                        default=[300,400])
    parser.add_argument("--sky_thickness",
                        help="sky background rind thickness: px (default is 30)",
                        default=30,
                        type=int)
    parser.add_argument("--back_method",
                        help='background subtraction method',
                        default="median",
                        choices=["mean", "median"])
    parser.add_argument("--ap_phot_fcr",
                        help='when set, perform aperture photometry on FCRs',
                        action='store_true')
    parser.add_argument("--ap_phot_flt",
                        help='when set, perform aperture photometry on FLTs',
                        action='store_true')

    args = parser.parse_args()

    return args


def check_preexisting_logging():
    """
    Helper function to verify that no logging is already
    set up, so that when this is run in interactive mode,
    there is no confusion about where the logs are. Not
    used when monitor is run from the command line.

    Returns
    -------
    preexisting_logging : Boolean
        Whether or not there is any preexisting logging
        configured in the current session.
    """
    existing_handlers = logging.getLogger().handlers
    if len(existing_handlers) == 0:
        preexisting_logging = False

    else:
        # then logging is already enabled for another file
        display_message(verbose=True, log=True,
                        log_type='critical',
                        message='Logging has already been enabled for file: '\
                                f'{existing_handlers[0].__str__().split(" ")[1]}')
        preexisting_logging = True

    return preexisting_logging


def setup_logging(local=False,
                  log_dir=os.getcwd(),
                  log_name=make_timestamp(),
                  verbose=True,
                  log=True):
    """
    Sets up logging. If pipeline is being run from the
    command line, this should be initialized at the
    beginning. Otherwise, this can be initialized when
    creating an `InteractiveArgs` object. Run only once.

    Parameters
    ----------
    local : Boolean
        Whether to write the log locally or to the monitor
        directory on central store. Default is `False`.
    log_dir : str or path-like
        String representation of path or path-like object
        indicating where the log should be saved. Set value
        only used if `local` is set to `True`; if not
        writing logs locally, `log_dir` is set to the log
        directory on central store. Defaults to current
        working directory.
    log_name : str
        Name for log. Defaults to timestamp at execution.
    verbose : Boolean
        Whether to print the message.
    log : Boolean
        Whether to log the message.
    """
    if local:
        if not os.path.exists(log_dir):
            log_dir = os.getcwd()
    else:
        log_dir = '/grp/hst/wfc3v/wfc3photom/data/uvis_scan_monitor/logs'

    log_file = os.path.join(log_dir, log_name)

    logging.basicConfig(filename=f'{log_file}.log', filemode='w',
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        level=logging.INFO)
    logging.info(f'Logging enabled; writing to {log_file}.log')
    display_message(verbose=verbose,
                    log=log,
                    log_type='info',
                    message=f'Logging enabled. Writing to file: {log_file}.log')

def display_message(verbose, log, message, log_type='info'):
    """
    Function that governs the display of messages.

    Parameters
    ----------
    verbose : Boolean
        Whether to print the message.
    log : Boolean
        Whether to log the message.
    message : str
        Message to be displayed.
    log_type : str
        Logging message type; defaults to `info`. Should
        be `info`, `warning`, `error`, or `critical`;
        otherwise, displays an additional warning message
        and logs original message as `info` type.
    """

    if verbose:
        print(message)

    if log:
        if log_type == 'info':
            logging.info(message)

        elif log_type == 'warning':
            logging.warning(message)

        elif log_type == 'error':
            logging.error(message)

        elif log_type == 'critical':
            logging.critical(message)

        else:
            log_type_message = '`display_message()` called with invalid '\
                               f'`log_type` = {log_type}\n'\
                               'Logging the following as `info` message:'
            if verbose:
                print(log_type_message)
            logging.warning(log_type_message)
            logging.info(message)
