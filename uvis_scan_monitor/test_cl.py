import argparse
from argparse import ArgumentParser
import sys
from housekeeping import _make_timestamp

def parse_args():
    """
    Parses command line arguments for the UVIS scan monitor
    pipeline. In addition to the default `--help` flag,
    there are a total of 18 configurable arguments:
        3 pipeline settings
        5 pipeline flags
        10 pipeline parameters

    Pipeline Settings
    -----------------
    verbose : Boolean
        Defaults to False.
    log : Boolean
        Defaults to False.
    name : str
        Name of the directory for this pipeline run in
        central store. Defaults to the timestamp returned
        by `_make_timestamp()`.

    Pipeline Flags
    --------------
    get_new_data : Boolean
    run_cr_reject : Boolean
    reprocess_fcr : Boolean
    run_ap_phot : Boolean
    show_ap_plot : Boolean

    Pipeline Parameters
    -------------------
    proposals : str or list of str

    targets : str or list of str
        Targets to select. If `download_new_data` is set to
        `True`, will download data for these targets; names
        will be resolved such that providing `GRW70` will
        also download data from MAST matching `GRW+70D` and
        `GRW+70D5824`. Otherwise, will only process (CR
        reject and/or perform photometry) data in the
        defined directory for given targets.
    filters : str or list of str
        WFC3/UVIS filters to select. If `download_new_data`
        is set to `True`, will only download data in these
        filters. Otherwise, will only process (CR reject
        and/or perform photometry) data in the defined
        directory in these filters.
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
        Thickness of sky background rind; defaults to 30px.
    back_method : str
        Which method to use for sky background calculation.
        Possible options are `median` and `mean`; defaults
        to `mean`.
    ap_phot_fcr : Boolean
        Defaults to False.
    ap_phot_flt : Boolean
        Defaults to False.

    Returns
    -------
    args : `argparse.Namespace`
        Namespace class object that has as attributes the
        18 configurable arguments.

    flags = {'get_new_data': False,
             'run_cr_reject': True,
             'reprocess_fcr': False,
             'run_ap_phot': True,
             'show_ap_plot': False}

    params = {'proposals': [14878, 15398, 15583, 16021, 16416, 16580, 17016],
              'targets': ['GRW70'], # 'GD153',
              'filters': ['F225W', 'F438W', 'F814W'],
              'file_type': 'flt',
              'ap_dim': (44, 268),
              'sky_ap_dim': (300, 400),
              'back_method': 'mean',
              'ap_phot_fcr': True,
              'ap_phot_flt': False}

    """
    parser = ArgumentParser(prog='uvis_scan_monitor',
                            description='WFC3/UVIS calibration scan '\
                                        'photometry monitor pipeline',
                            epilog = 'Author: Mariarosa Marinelli')

    # output viewing arguments
    parser.add_argument("-v", "--verbose",
                        help="when set, prints statements to command line",
                        action="store_true")
    parser.add_argument("-l", "--log",
                        help="when set, logs statements to log file",
                        action="store_true")
    parser.add_argument("-n", "--name",
                        help="name for pipeline run; defaults to timestamp",
                        default=_make_timestamp())

    # flags
    parser.add_argument("-g", "--get_new_data",
                        help="when set, get new data",
                        action='store_true')
    parser.add_argument("-c", "--run_cr_reject",
                        help="when set, run cosmic ray rejection",
                        action='store_true')
    parser.add_argument("-r", "--reprocess_fcr",
                        help="when set, reprocess existing FCR files",
                        action='store_true')
    parser.add_argument("-a", "--run_ap_phot",
                        help="when set, run aperture photometry",
                        action='store_true')
    parser.add_argument("-s", "--show_ap_plot",
                        help="when set, show source detection plots",
                        action='store_true')

    # parameters
    parser.add_argument("--proposals",
                        help="calibration proposal or list of proposals",
                        nargs="+",
                        default=[14878, 15398, 15583, 16021, 16416, 16580, 17016])
    parser.add_argument("--targets",
                        help="target or list of targets (default is 'all')",
                        nargs="+",
                        default="all")
    parser.add_argument("--filters",
                        help="filter or list of filters (default is 'all')",
                        nargs="+",
                        default="all")

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

    for prop, val in vars(args).items():
        print(f'{prop}: {val}')

    return args

#parse_args()
