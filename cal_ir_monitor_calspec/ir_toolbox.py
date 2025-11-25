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


Functions
---------
    display_args(args)
        Prints and/or logs `args` values.
    get_decimalyear(mjd)
        Converts MJD to `decimalyear` format.
    parse_args()
        Parses `ir_phot_monitor.py` command line arguments.
    resolve_targnames(targname, simplify)

Author
------
Mariarosa Marinelli, 2023
"""
from datetime import datetime
import os
from argparse import ArgumentParser
from astropy.time import Time

from ir_config import CONFIG


PAM = CONFIG['pam_file']
MONITOR_DIR = CONFIG['monitor_dir']
MONITOR_PROGRAMS = CONFIG['programs']
SIMPLE_TARGS = CONFIG['targets']
CR_PD = CONFIG['cr_pd']


# class CaptureOutput(list):
#     """
#     Class to capture output from externally-imported
#     functions.
#
#     Parameters
#     ----------
#     list : list of str
#         List of output strings.
#     """
#     def __enter__(self):
#         self._stdout = sys.stdout
#         sys.stdout = self._stringio = StringIO()
#
#         return self
#
#     def __exit__(self, *args):
#         self.extend(self._stringio.getvalue().splitlines())
#         del self._stringio
#         sys.stdout = self._stdout


def make_timestamp():
    """Creates string timestamp for current datetime.

    Helper function to convert and format current datetime
    into a string. This string is then used for the name
    of the pipeline run directory in the scan monitor
    photometry central store location:
        /grp/hst/wfc3v/wfc3photom/data/ir_staring_monitor

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
    phot_cols = ['exposure_file', 'radius',
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
                      'syn_cr', 'mean_obs_syn_cr',
                      'median_obs_syn_cr', 'mode_obs_syn_cr',
                      'photutils_sum', 'obs_syn_photutils',
                      'image_std', f'dq_count_{dq_buffer}',
                      'tds_mean_mag_flux', 'tds_median_mag_flux',
                      'tds_mode_mag_flux'])

    return phot_cols


def display_args(args):
    """Prints and/or logs `args` values.

    Method displays arguments for an `ArgParse` object.

    Usage
    -----
    Called directly, ex:

            if __name__ == '__main__':
                args_to_display = parse_args()
                display_args(args_to_display)

    Parameter
    ---------
    args : `ArgParse` object
        Arguments entered at the command line.
    """
    top = ["ARGUMENT", "VALUE"]
    print(f'\n{top[0]:15} {top[1]:15}')
    print(f'{len(top[0])*"-":15} {len(top[1])*"-":15}')
    for prop, val in vars(args).items():
        print(f'{prop:15} {val}')


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
        Run in 'trial' mode.

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
    helium : Boolean
    linearity: Boolean
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
        process data in the defined directory for given
        targets. If no targets are defined, all targets
        available will be processed.
    filters : str or list of str
        WFC3/IR filters to select. If `download_new_data`
        is set to `True`, will only download data in these
        filters. Otherwise, will only process data in the
        defined directory in these filters. If no filters
        are defined, all filters available will be
        processed.
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
    parser = ArgumentParser(prog='cal_ir_monitor_calspec',
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

    # Execution Flags:
    parser.add_argument("-g", "--get_new_data",
                        help="if set, get new data",
                        action='store_true')
    parser.add_argument("-r", "--redownload",
                        help="if set, redownload data",
                        action='store_true')
    parser.add_argument("-e", "--helium",
                        help="if set, correct for He I in F105W & F110W",
                        action='store_true')
    parser.add_argument("-l", "--linearity",
                        help="if set, update linearity correction coefficients",
                        action='store_true')
    parser.add_argument("-a", "--run_ap_phot",
                        help="if set, run aperture photometry",
                        action='store_true')

    # Pipeline Parameters:
    parser.add_argument("--proposals",
                        help="proposal ID(s) (default set in config.yaml)",
                        nargs="+",
                        type=int,
                        default=MONITOR_PROGRAMS)
    parser.add_argument("--targets",
                        help="target(s) (default set in config.yaml)",
                        nargs="+",
                        default=SIMPLE_TARGS)
    parser.add_argument("--filters",
                        help="filter(s) (default set in config.yaml)",
                        nargs="+",
                        default=list(CONFIG['tds_2024'].keys()))
    parser.add_argument("--radius",
                        help="photometric aperture radius (pixels)",
                        type=int,
                        default=3)
    parser.add_argument("--annulus",
                        help="inner radius of the background annulus (pixels)",
                        type=int,
                        default=14)
    parser.add_argument("--dannulus",
                        help="width of the background annulus (pixels)",
                        type=int,
                        default=5)
    parser.add_argument("--back_method",
                        help="method to calculate background from sigma-clipped data",
                        default="median",
                        choices=["mean", "median", "mode"])
    parser.add_argument("--ap_phot_flt",
                        help='when set, perform aperture photometry on FLTs',
                        action='store_true')
    parser.add_argument("--write_dir",
                        help="directory where tables/plots should be saved",
                        default=os.path.join(MONITOR_DIR, 'output'))
    parser.add_argument("--plot_sources",
                        help="when set, save source detection/selection plots",
                        action='store_true')
    parser.add_argument("--nlinfile",
                        help='location of linearity correction file, if reprocessing',
                        type=str,
                        default=None)
#    parser.add_argument("--update_refs",
#                        help="when set, run bestrefs",
#                        action='store_true')

    args = parser.parse_args()

    return args


def resolve_targnames(targname, simplify=True):
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
            print(f'Unable to resolve name for {key_error}')
            resolved = targname

    else:
        resolved_targnames = [k for k, v in targnames.items() if v == targname]
        if len(resolved_targnames) > 0:
            resolved = resolved_targnames
        else:
            resolved = targname

    return resolved
