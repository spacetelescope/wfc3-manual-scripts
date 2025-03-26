#!/usr/bin/env python
"""
Note: this script replaces scan_psf.py 

Purpose
-------
    1) Makes and saves the following filter- and chip-
    dependent point spread functions (PSFs):
        - `simple` : derived solely from published UVIS EE
           curves.
        - `blended` : derived from empirical PSF models
           that are extended to a 150 px radius by blending
           with published UVIS EE curves.
    2) Convolves `simple` and `blended` PSFs with a line
    corresponding to the trajectory of the UVIS calibration
    spatial scans to create and save  scan spread functions
    (SSFs).

Usage
-----
    This script can be run from the command line with one
    required argument, two :

        python uvis_convolve_psf.py --type TYPE (--verbose --log --name NAME)
                                    (--ee_dir EE_DIR
                                    --filters [FILTERS] --uvis [UVIS]
                                    --xlim [XLIM XLIM] --ylim [YLIM YLIM])

    where the arguments are:
        -t or --type : string
            Type of PSF to convolve. Required argument. Valid options are
            "simple" or "blended" (no quotation marks).
        -l or --log : Boolean
            Whether to print messages to command line. Optional argument.
        -v or --verbose : Boolean
            Whether to log messages to log file. Optional argument.
        -n or --name : str
            Name for pipeline run. Optional argument, since it defaults to
            timestamp.
        -e or --ee_dir : string
            String represenation of the directory wherein the WFC3/UVIS
            encircled energy tables can be found. Optional argument; defaults
            to /grp/hst/wfc3v/wfc3photom/data/uvis_scan_monitor/synphot.
        -f or --filters : string or list of strings
            Filters to create PSF/SSFs for. Optional argument, but it defaults
            to all core filters for "simple" PSFs, and all available filters
            for "blended" PSFs.
        -u or --uvis : int(s)
            Which CCD to make PSFs for. Optional argument; defaults to making
            PSFs for both (i.e. [1, 2]).
        -x or --xlim : list of int
            The x-location of (start, end) of scan, in pixels. Optional
            argument; defaults to [256, 256].
        -y or --ylim : list of int
            The y-location of (start, end) of scan, in pixels. Optional
            argument; defaults to [160, 352], which is appropriate for 59.9s
            scans (i.e. WD scans and scans of P330E/TYC in redder filters).

    This script can also be imported and used as such:

        from uvis_convolve_psf import make_convolved_psf

Authors
-------
    Mariarosa Marinelli, 2025
    Varun Bajaj, 2021
"""

from argparse import ArgumentParser
from astropy.io import fits
from astropy.table import Table
import matplotlib.pyplot as plt
import numpy as np
import os
from photutils.aperture import CircularAperture, aperture_photometry
from scipy.interpolate import interp1d
from scipy.signal import convolve2d
from skimage.draw import line_aa

from plotting_tools import MY_COLORS, MPL_SETTINGS
from toolbox import MONITOR_DIR, CORE_FILTERS, JAY_FILTERS
from toolbox import check_subdirectory, display_message, make_timestamp

SYNPHOT_DIR = '/grp/hst/wfc3v/wfc3photom/data/uvis_scan_monitor/synphot'


class PSF():
    """
    Class to create a blended PSF that is radially
    symmetric, and generated from both an empirical PSF
    (referred to throughout as `BlendedPSF` as they were made
    by Jay Anderson) and the published EE curve.

    This can be useful for convolving with various shapes
    to get better estimates of aperture corrections of non
    point-like objects, such as spatial scans

    NOTE: For now this only is guaranteed to work for UVIS
    data taken from wfc3uvis1_aper_007_syn.csv or
    wfc3uvis2_aper_007_syn.csv, because of a small quirk in
    how the radii for the EE are recorded in the table.

    Parameters
    ----------
    filt : str
        The filter which to generate the BlendedPSF (EE curves
        are filter dependent)
    ee_table : astropy Table
        The table containing the EE measurements (in
        arcsec) for the UVIS channel
    norm_rad : float
        The radius at which the PSF hits 1. If not
        provided, assumed to be the last value in the
        radii. Probably actually required to be the last
        value.
    psf_file : fits
        FITS file, i.e. for Jay's PSFs (calculated for UVIS
        2 only.)

    Useful attributes
    -----------------
    psf : numpy.ndarry
        Array containing the radially symmetric PSF
        function.
    filt : str
        The filter corresponding to the EE values extracted
        from the table
    radii : numpy.ndarry
        Array of radii in pixels the EE was tabulated at
        (see note above, these radii do not match the input
        table)
    ee_array : numpy.ndarry
        Array of EE values corresponding to radii above
    ee_interp : scipy.interpolate.interpolate.interp1d
        Interpolated encircled energy object, input in
        pixels
    center : (float, float)
        X and Y coordinates of the center of the PSF in the
        psf image. 0 indexed.
    """
    def __init__(self, filt, ee_table, uvis,
                 verbose=True, log=False, norm_rad=None):
        """The constructor method

        Note
        ----
            When assigning the `radii` attribute, rounding
            is done because the radii in the EE table are
            rounded to two decimals, so this gets the
            integer pixel positions back. This kludge is
            likely unnecessary when more precision is
            available.

        Parameters
        ----------
        filt :
        ee_table :
        norm_rad :
        """
        self.uvis = uvis
        self.verbose = verbose
        self.log = log
        self.ees = ee_table
        self.filt = filt

        radii = np.array([0] + [float(cn.split('#')[-1]) for cn in self.ees.colnames[2:]])
        self.radii = np.around(radii / .0396, 0) # convert to pixels

        if norm_rad is None:
            self.rad = self.radii[-1]
        else:
            self.rad = norm_rad

        self.ee_array = self.get_ee()
        self.ee_interp = interp1d(self.radii, self.ee_array, fill_value=1.,
                                  bounds_error=False)
        self._ris = np.arange(1., self.rad)
        self._cenx = float(self.rad)
        self._ceny = float(self.rad)
        self.center = (self._cenx, self._ceny)
        self.diameter = int(2 * self.rad)
        self.psf = np.zeros((self.diameter + 1, self.diameter + 1))


    def get_ee(self):
        """
        Retrieves the EE curve for given filter as an array
        from the read-in EE table.

        Returns
        -------
        vals : `numpy.ndarray`
            Array of encircled energy values.
            shape 52,
        """
        rowind = np.where(self.ees['FILTER'] == self.filt.upper())[0][0]
        row = self.ees[rowind]
        vals = [row[cn] for cn in self.ees.colnames[2:]]
        vals = [0] + vals
        vals = np.array(vals)

        return vals


    def _pixval(self, dist):
        """
        Calculates pixel value of a pixel some distance
        from center of star, assuming radially symmetric
        PSF.

        This computes the energy inside an annulus at
        r_in = dist -.5 and r_out = dist + .5 and divides
        by the area of the annulus to get average energy
        per pixel in the annulus.  This breaks down for the
        central pixels due to the approximations made, but
        is corrected in via 'correct_psf()'.

        Parameter
        ---------
        dist : float

        Returns
        -------
        avpix : float
        """
        # Calculate delta EE from a 1 pixel step
        inner = np.amax([0., dist - .5])
        outer = dist + .5
        delta_ee = self.ee_interp(outer) - self.ee_interp(inner)

        # divide delta EE by area of annulus
        avpix = delta_ee / (np.pi * (outer**2. - inner**2.))

        return avpix

    def calc_ee(self, data):
        """
        This creates an encircled energy curve for the
        simple or blended PSF using apertures of increasing
        radii.

        Parameter
        ---------
        data : `numpy.ndarray`
            shape 305,305

        Returns
        -------
        psf_ee : `numpy.ndarray`
            Array of encircled energy curve values.
            shape 151,
        """
        aps = [CircularAperture(self.center, r=ri) for ri in self._ris]

        phots = aperture_photometry(data, aps, method='exact')
        psf_ee = np.array([phots[cn][0] for cn in phots.colnames[3:]])

        return psf_ee


class SimplePSF(PSF):
    """
    Parameters
    ----------
    PSF : obj
        The class that initializes a convolved PSF.

    Attributes
    ----------
    psf_ee :
    """
    def __init__(self, filt, ee_table, uvis, norm_rad=None):
        """The constructor method.

        Parameters
        ---------
        """
        PSF.__init__(self, filt, ee_table, uvis, norm_rad)

        self.generate_image()
        self.psf_ee = self.calc_ee(self.psf)
        self.correct_psf()

    def _average_dist(self, x_i, y_i, x_0=0., y_0=0.):
        """
        Calculate the average distance to some pixel
        (x_i, y_i) from point (x_0, y_0).

        This is kind of important for getting the values of
        the PSF close to the center correct. Since a pixel
        isn't a point, the average distance to that pixel
        can vary depending where (x0, y0) are. When the
        distance is large, this is well approximated by the
        distance to the center of the pixel.
        """
        if (np.abs(x_i - x_0) > 30.) or (np.abs(y_i - y_0) > 30.):
            return np.sqrt((x_i - x_0)**2. + (y_i - y_0)**2.)

        d_y, d_x = np.mgrid[-50:51, -50:51].astype(float)/100.
        posx, posy = x_i + d_x, y_i + d_y
        dists = np.sqrt((posx - x_0)**2. + (posy - y_0)**2.)

        return np.mean(dists)

    def generate_image(self):#, radius=None):
        """Generate image of SimplePSF.
        """
#        self.setup_image(radius=radius)
        for i in range(0, self.diameter + 1):
            for j in range(0, self.diameter + 1):
                # Calculate average distance
                average_dist = self._average_dist(i, j, self._cenx, self._ceny)
                self.psf[j, i] = self._pixval(average_dist)

    def correct_psf(self):
        """
        This calculates the EE curve of the original simple
        PSF, finds the offset between that and the real EE
        curve, and puts that offset back into the central
        pixel.
        """
        display_message(self.verbose, self.log,
                        message='computing offset from real EE via '\
                                'aperture photometry')

        self.offsets = self.ee_interp(self._ris) - self.psf_ee
        self.correction = np.nanmedian(self.offsets)
        self.psf[int(self._ceny), int(self._cenx)] += self.correction

        for message in [f'Calculated offset: {str(self.correction)}',
                        f'STD of correction: {str(np.nanstd(self.offsets))}',
                        'STD of correction r>10 pix: '\
                        f'{str(np.nanstd(self.offsets[10:]))}']:
            display_message(self.verbose, self.log, message)


    def plot_ees(self, show=True, save=False, save_dir=None):
        """Plot encircled energy of SimplePSF
        """
        plt.rcParams.update(MPL_SETTINGS)

        fig, axes = plt.subplots(figsize=(8,4))
        axes.grid(alpha=0.2, zorder=0)
        axes.plot(self.radii, self.ee_array, c=MY_COLORS[0], alpha=0.7, lw=2,
                  label='real EE', zorder=2)
        axes.plot(self._ris, self.ee_interp(self._ris) - self.offsets, lw=2,
                  c=MY_COLORS[5], alpha=0.7, label='uncorr PSF EE', zorder=3)
        axes.set_xlabel('Radius (pixels)')
        axes.set_ylabel('Encircled Energy')
        axes.set_title(f'SimplePSF: WFC3/UVIS {self.uvis[-1]} {self.filt}')
        axes.legend(loc='best')
        fig.tight_layout()

        if save:
            if isinstance(save_dir, type(None)):
                save_dir = os.getcwd()
            filename = os.path.join(save_dir, f'simple_{self.uvis}_'\
                                    f'{self.filt}_ee.png')
            plt.savefig(filename, dpi=250)
        if show:
            plt.show()

        plt.close()


class BlendedPSF(PSF):
    """
    Parameters
    ----------
    PSF : obj
        The class that initializes a convolved PSF.

    Attributes
    ----------
    jpsf_data :
    jrad :
    self._ris :
    psf_ee :
    """
    def __init__(self, filt, ee_table, uvis, psf_file, norm_rad=None):
        """The constructor method.

        Parameters
        ---------
        """
        PSF.__init__(self, filt, ee_table, uvis, norm_rad)

        self.jpsf_data = fits.getdata(psf_file)
        self.jrad = int((self.jpsf_data.shape[0] - 1) / 2)
        self._ris = np.arange(1., self.jrad)

        self.psf_ee = self.calc_ee(self.jpsf_data)
        self._norm_jpsf()
        self.generate_image()
        self.update_ee()


    def _norm_jpsf(self):
        """Normalize BlendedPSF.

        Pins the EE of empirical PSF to the EE curve value
        at the radius of the empirical PSF.
        """
        psf_rat = self.ee_interp(self.jrad) / self.psf_ee[-1]
        self.jpsf_data *= psf_rat


    def generate_image(self):
        """Generate image of BlendedPSF.
        """
        # Insert empirical PSF:
        diff_rad = int(self.rad - self.jrad)
        sum_rad = int(self.rad + self.jrad)

        self.psf[diff_rad:(sum_rad + 1), diff_rad:(sum_rad + 1)] = self.jpsf_data

        for i in range(0, self.diameter + 1):
            for j in range(0, self.diameter + 1):
                dist = np.sqrt((i - self._cenx)**2. + (j - self._ceny)**2.)

                if dist <= 100.:
                    continue
                else:
                #if dist > 100.:
                    self.psf[j, i] = self._pixval(dist)


    def plot_ees(self, show=True, save=False, save_dir=None):
        """Plot encircled energy of BlendedPSF
        """
        plt.rcParams.update(MPL_SETTINGS)

        fig, axes = plt.subplots(figsize=(8,4))
        axes.grid(alpha=0.2, zorder=0)
        axes.plot(self.radii[1:], self.ee_array[1:], c=MY_COLORS[0], alpha=0.7,
                  lw=2, zorder=2, label='real EE')
        axes.plot(self._ris, self.psf_ee, c=MY_COLORS[5], alpha=0.7, lw=2,
                  zorder=3, label='JPSF PSF EE')
        axes.set_xlabel('Radius (pixels)')
        axes.set_ylabel('Encircled Energy')
        axes.set_title(f'BlendedPSF: WFC3/UVIS {self.uvis[-1]} {self.filt}')
        axes.legend(loc='best')
        fig.tight_layout()

        if save:
            if isinstance(save_dir, type(None)):
                save_dir = os.getcwd()
            filename = os.path.join(save_dir, f'blended_{self.uvis}_'\
                                    f'{self.filt}_ee.png')
            plt.savefig(filename, dpi=250)
        if show:
            plt.show()

        plt.close()


    def update_ee(self):
        """
        Now that we've blended the PSFs, we need to
        recalculate the encircled energy.
        """
        self._ris = np.arange(1., self.rad)
        self.psf_ee = self.calc_ee(self.psf)


def make_scan_line(img_shape, x_start, x_end, y_start, y_end):
    """Construct line to convolve PSF with.

    This makes a line in a numpy array. Because the
    `line_aa()` function antialiases lines, they could be
    made at an angle, but this may not preserve the "flux"
    represented by the line across the whole scan (i.e.
    the antialiasing probably makes it brighter by adding
    soft edges). Might be best to keep the lines straight
    as a result, or might not matter. Probably the latter.

    Note
    ----
        Insignificant difference between convolving with
        vertical line and convolving with a line 4 degrees
        from the vertical. Since these calibration scans
        are less than 1 degree tilted with respect to the
        vertical, we're fine.

    Parameters
    ----------
    image_shape : (int, int)
        Shape of image to be convolved (ypix, xpix). Should
        probably leave enough space for the edges of the
        kernel not to be cut off. i.e. if feature in image
        is 200 pixels, and kernel size is 300x300, then
        dimension should be aat least 200+150+150 = 500 to
        not cut off edges.
    x_start: int
        x-position of line start, in pixels.
    x_end: int
        x-position of line end, in pixels.
    y_start: int
        y-position of line start, in pixels.
    y_end: int
        y-position of line end, in pixels.

    Returns
    -------
    img : `numpy.ndarray`
        A numpy array with pixels along line set to 1, and
        potential antialiased pixels set to the antialiased
        value.
    """
    img = np.zeros(img_shape, dtype=np.float64)

    rr, cc, val = line_aa(y_start, x_start, y_end, x_end)
    img[rr, cc] = val

    return img


def make_dirs(parent_dir, verbose, log, make_plots_dir=True):
    """
    Helper functions to create directories as needed.

    Parameter
    ---------
    parent_dir : str

    Returns
    -------
    psf_dir : str
        String representation of path where PSF files
        should be saved.
    ssf_dir : str
        String representation of path where SSF files
        should be saved.
    """
    psf_dir = check_subdirectory(parent_dir, "psf", verbose, log)

    ssf_dir = check_subdirectory(parent_dir, "ssf", verbose, log)

    if make_plots_dir:
        save_dir = check_subdirectory(parent_dir, "plots", verbose, log)
    else:
        save_dir = None

    return [psf_dir, ssf_dir, save_dir]


def write_numpy_file(filename, numpy_object, file_type, delimiter=',',
                     verbose=True, log=False):
    """Helper function to save numpy object to file.

    Parameters
    ----------
    filename : str
        String representation of the full filepath and
        name for the file to be written.
    numpy_object :
        What object is being saved to file.
    file_type : str
        Used for the displayed messages. Should be either
        'PSF' or 'SSF'.
    delimiter : str
        Default is comma-delimited (',').
    """
    np.savetxt(filename, numpy_object, delimiter=delimiter)

    if os.path.exists(filename):
        display_message(verbose, log,
                        message=f'{file_type} file {filename} saved.')
    else:
        display_message(verbose, log, log_type='error',
                        message=f'Could not save {file_type} file to {filename}')


def make_convolved_psf(psf_type, filt, ee_table, uvis, parent_dir,
                       x_lim=(256, 256), y_lim=(160, 352), psf_file=None,
                       verbose=True, log=False,
                       show_plots=False, save_plots=False):
                      #x_start=256, x_end=256, y_start=160, y_end=352):
    """
    Main function to create a filter- and chip-dependent
    PSF, convolve with the scan trajectory line, and save
    the resulting 'scan spread function' (SSF).

    Parameters
    ----------
    psf_type : str
        Either `simple` or `blended`.
    filt : str
        String name of filter.
    ee_table : `astropy.table.table.Table`
        Encircled energy and aperture table derived from
        staring mode data (filter- and chip-dependent).
    uvis : str
        Which CCD; either `uvis1` or `uvis2`.
    save_dir_name : str
        Name of directory in `synphot_dir` where files
        should be saved.
    """
    [psf_dir, ssf_dir, save_dir] = make_dirs(parent_dir, verbose, log, save_plots)

    if psf_type == 'simple':
        model_psf = SimplePSF(filt, ee_table, uvis)

    else:
        if isinstance(psf_file, type(None)):
            # abort
            display_message(self.verbose, self.log, log_type='critical',
                            message=f'PSF file is NoneType: {psf_file}')
        else:
            model_psf = BlendedPSF(filt, ee_table, uvis, psf_file)

    model_psf.plot_ees(show_plots, save_plots, save_dir)

    psf_filename = os.path.join(psf_dir, f'{psf_type}_{uvis}_{filt}_psf.csv')
    write_numpy_file(psf_filename, model_psf.psf, file_type='PSF')

    line_model = make_scan_line(img_shape=(512,512),
                                x_start=x_lim[0], x_end=x_lim[1],
                                y_start=y_lim[0], y_end=y_lim[1])

    model_ssf = convolve2d(line_model, model_psf.psf, mode='same')

    ssf_filename = os.path.join(ssf_dir, f'{psf_type}_{uvis}_{filt}_ssf.csv')
    write_numpy_file(ssf_filename, model_ssf, file_type='SSF')


def parse_args():
    """
    Parses command line arguments.

    Returns
    -------
    args : `argparse.Namespace`
        Object where the attributes correspond to the
        arguments given at the command line (and the
        default values for optional arguments, if
        applicable).
    """
    parser = ArgumentParser(prog='uvis_convolve_psf',
                            description='creates a point spread function from '\
                                        'published EE values & empirical PSF models',
                            epilog = 'Authors: Mariarosa Marinelli & Varun Bajaj')

    parser.add_argument("-v", "--verbose",
                        help="when set, prints statements to command line",
                        action="store_true")
    parser.add_argument("-l", "--log",
                        help="when set, logs statements to log file",
                        action="store_true")
    parser.add_argument("-n", "--name",
                        help="name for pipeline run; defaults to timestamp",
                        default=make_timestamp())

    parser.add_argument("-t", "--type",
                        choices=["simple", "blended"],
                        help="type of convolved PSF: simple or blended",
                        required=True)
    parser.add_argument("-e", "--ee_dir",
                        help="directory location of encircled energy tables",
                        default=SYNPHOT_DIR)

    parser.add_argument("-f", "--filters",
                        help="filter or list of filters (default is `all`)",
                        nargs="+",
                        default=["all"])
    parser.add_argument("-u", "--uvis",
                        help="list integer number(s) for UVIS CCD(s)",
                        nargs="+",
                        type=int,
                        default=[1, 2])
    parser.add_argument("-x", "--x_lim",
                        help="x location of (start, end) of scan in pixels",
                        nargs=2,
                        type=int,
                        default=[256, 256])
    parser.add_argument("-y", "--y_lim",
                        help="y location of (start, end) of scan in pixels",
                        nargs=2,
                        type=int,
                        default=[160, 352])
    parser.add_argument("--show_plots",
                        help="when set, shows plots",
                        action="store_true")
    parser.add_argument("--save_plots",
                        help="when set, saves plots",
                        action="store_true")

    args = parser.parse_args()

    # Some minor reformatting:
    filt_dict = {'simple': CORE_FILTERS, 'blended': JAY_FILTERS}
    if args.filters == ['all']:
        args.filters = filt_dict[args.type]
    else:
        args.filters = [f for f in args.filters if f in filt_dict[args.type]]

    args.uvis = [f'uvis{u}' for u in args.uvis]

    args.dir = check_subdirectory(SYNPHOT_DIR, args.name,
                                  args.verbose, args.log)
    return args


if __name__ == '__main__':

    args = parse_args()

    for uvis in args.uvis:
        pub_ee_path = os.path.join(args.ee_dir, f'wfc3{uvis}_aper_007_syn.csv')
        pub_ee = Table.read(pub_ee_path, format='csv')

        for filt in args.filters:
            if args.type == 'blended':
                jpsf = os.path.join(SYNPHOT_DIR, 'jpsf', f'psfnrm_{filt}.fits')
            else:
                jpsf = None

            make_convolved_psf(args.type, filt, pub_ee, uvis, args.dir,
                               args.x_lim, args.y_lim, jpsf,
                               args.verbose, args.log,
                               args.show_plots, args.save_plots)
