"""
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
    This script can be run from the command line with two
    required arguments and two optional arguments:

        python scan_psf.py -t <type>
                           -d <directory>
                           (-f <filters>
                           -u <uvis>)

    where the arguments are:
        -t or --type : string
            Type of PSF to convolve. Required argument.
            Valid options are "simple" or "blended" (no
            quotation marks).
        -d or --directory : string
            Name of directory in /grp/hst/wfc3v/wfc3photom/
            data/uvis_scan_monitor/synphot/ where files
            will be saved. If the directory does not exist,
            it will be created. Required argument.
        -f or --filters : string or list of strings
            Filters to create PSF/SSFs for. Optional
            argument (defaults to all core filters for
            simple and all available filters for blended).
        -u or --uvis : int or string
            Which CCD to make PSF/SSFs for. Optional
            argument (defaults to "both"). Valid options
            are 1, 2, or "both".

    This script can also be imported and used as such:

        from scan_psf import make_convolvedpsf

Authors
-------
    Mariarosa Marinelli
    Varun Bajaj
"""
#!/usr/bin/env python
import argparse
from argparse import ArgumentParser
from astropy.io import fits
from astropy.table import Table
import matplotlib as mpl
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import numpy as np
import os
from photutils.aperture import CircularAperture, aperture_photometry
from scipy.interpolate import interp1d
from scipy.signal import convolve2d
from skimage.draw import line_aa

monitor_dir = '/grp/hst/wfc3v/wfc3photom/data/uvis_scan_monitor'
synphot_dir = os.path.join(monitor_dir, 'synphot')
#save_dir = os.path.join(synphot_dir, '2023_03_14_test1')


# Main class for generating Toy PSF to convolve with scan
class ToyPSF():
    """
    Class to create a simple PSF (referred to as a `Toy
    PSF`) that is radially symmetric and generated just
    from the EE curve.

    This can be useful for convolving with various shapes
    to get better estimates of aperture corrections of non
    point-like objects, such as spatial scans

    NOTE: For now this only is guaranteed to work for UVIS
    data taken from wfc3uvis1_aper_007_syn.csv or
    wfc3uvis2_aper_007_syn.csv, because of a small quirk in
    how the radii for the EE are recorded in the table.

    Author
    ------
        Varun Bajaj, 2021

    Parameters
    ----------
    filt : str
        The filter which to generate the ToyPSF (EE curves
        are filter dependent)
    ee_table : astropy Table
        The table containing the EE measurements (in
        arcsec) for the UVIS channel
    norm_rad : float
        The radius at which the PSF hits 1. If not
        provided, assumed to be the last value in the
        radii. Probably actually required to be the last
        value.

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

    def __init__(self, filt, ee_table, norm_rad=None):
        self.ees = ee_table
        self.filt = filt

        self.radii = np.array([0]+[float(cn.split('#')[-1]) for cn in self.ees.colnames[2:]])
        self.radii = np.around(self.radii/.0396, 0) # convert to pixels
        # NOTE: rounding is done because the radii in the file are rounded to 2 decimals, so this gets
        # the integer pixel positions back.  This kludge is likely unnecessary when more precision is
        # available

        if norm_rad is None:
            self.rad = self.radii[-1]
            print('norm_rad is None')
        else:
            self.rad = norm_rad
            print('norm_rad is not None')
        print(f'self.rad = {self.rad}')

        self.ee_array = self._get_ee(filt)
        self.ee_interp = interp1d(self.radii, self.ee_array, fill_value=1., bounds_error=False)

        self.generate_image()
        self.correct_psf()


    def _get_ee(self, filt):
        """
        Gets the EE curve for filt as an array from the
        Table.
        """
        rowind = np.where(self.ees['FILTER'] == self.filt.upper())[0][0]
        row = self.ees[rowind]
        vals = [row[cn] for cn in self.ees.colnames[2:]]
        vals = [0] + vals
        vals = np.array(vals)
        return vals

    def _average_dist(self, x, y, x0=0., y0=0.):
        """
        Calculates the average distance to some pixel
        (x,y) from point (x0,y0)

        This is kind of important for getting the values of
        the PSF close to the center
        correct. Since a pixel isnt a point, the average
        distance to that pixel can vary depending where
        (x0,y0) are. When the distance is large, this is
        well approximated by the distance to the center
        of the pixel.
        """
        if (np.abs(x-x0)>30.) or (np.abs(y-y0)>30.):
            return np.sqrt((x-x0)**2. + (y-y0)**2.)
        dy, dx = np.mgrid[-50:51, -50:51].astype(float)/100.
        posx, posy = x+dx, y+dy
        dists = np.sqrt((posx-x0)**2. + (posy-y0)**2.)
        return np.mean(dists)

    def _pixval(self, dist):
        """
        Calculates pixel value of a pixel some distance
        from center of star, assuming radially symmetric
        PSF.

        This computes the energy inside an annulus at
        r_in = dist -.5 and r_out = dist + .5 and divides
        by the area of the annulus to get average energy
        per pixel in the annulus.  This breaks down for
        the central pixels due to the approximations made,
        but is corrected in via 'correct_psf()'.
        """
        # Calculate delta EE from a 1 pixel step
        inner = np.amax([0., dist-.5])
        outer = dist+.5
        dEE = self.ee_interp(outer) - self.ee_interp(inner)

        # divide delta EE by area of annulus
        avpix = dEE/(np.pi*(outer**2. - inner ** 2.))
        return avpix

    def generate_image(self, radius=None):
        """
        This is the way to generate the simple PSF as an
        image. There's probably a smarter way to do this.
        """

        if radius is None:
            radius = int(self.rad)
        print("creating Toy PSF image")
        self._cenx = float(radius)
        self._ceny = float(radius)
        self.center = (self._cenx, self._ceny)

        self.psf = np.zeros((2*radius+1, 2*radius+1))
        for i in range(0, 2*radius+1):
            for j in range(0, 2*radius+1):
                self.psf[j, i] = self._pixval(self._average_dist(i, j, self._cenx, self._ceny))

    def calc_toy_ee(self):
        """
        This creates an encircled energy curve for the
        simple PSF using apertures of increasing radii.
        """
        self._ris = np.arange(1., self.rad)
        aps = [CircularAperture(self.center, r=ri) for ri in self._ris]

        phots = aperture_photometry(self.psf, aps, method='exact')
        toy_ee = np.array([phots[cn][0] for cn in phots.colnames[3:]])
        return toy_ee


    def correct_psf(self):
        """
        This calculates the EE curve of the original simple
        PSF, finds the offset between that and the real EE
        curve, and puts that offset back into the central
        pixel.
        """
        print('computing offset from real EE via aperture photometry')

        toy_ee = self.calc_toy_ee()

        self.offsets = self.ee_interp(self._ris)-toy_ee
        self.correction = np.nanmedian(self.offsets)
        self.psf[int(self._ceny), int(self._cenx)] += self.correction
        print('Calculated offset: {}'.format(self.correction))
        print('STD of correction: {}'.format(np.nanstd(self.offsets)))
        print('STD of correction r>10 pix: {}'.format(np.nanstd(self.offsets[10:])))

    def plot_ees(self):
        '''This just plots the real and uncorrected Toy EE'''
        plt.plot(self.radii, self.ee_array, label='real EE')
        plt.plot(self._ris, self.ee_interp(self._ris)-self.offsets, label='uncorr PSF EE')


class JayPSF():
    """
    Class to create a blended PSF that is radially
    symmetric, and generated from both an empirical PSF
    (referred to throughout as `JayPSF` as they were made
    by Jay Anderson) and the published EE curve.

    This can be useful for convolving with various shapes
    to get better estimates of aperture corrections of non
    point-like objects, such as spatial scans

    NOTE: For now this only is guaranteed to work for UVIS
    data taken from wfc3uvis1_aper_007_syn.csv or
    wfc3uvis2_aper_007_syn.csv, because of a small quirk in
    how the radii for the EE are recorded in the table.

    Author
    ------
        Varun Bajaj, 2021

    Parameters
    ----------
    filt : str
        The filter which to generate the JayPSF (EE curves
        are filter dependent)
    ee_table : astropy Table
        The table containing the EE measurements (in
        arcsec) for the UVIS channel
    jpsf_file : fits
        Fits file for Jay's PSFs which were calculated for
        UVIS 2 only.
    norm_rad : float
        The radius at which the PSF hits 1. If not
        provided, assumed to be the last value in the
        radii. Probably actually required to be the last
        value.

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
        psf image. 0 indexed."""

    def __init__(self, filt, ee_table, jpsf_file, norm_rad=None):
        self.ees = ee_table
        self.filt = filt

        self.radii = np.array([0]+[float(cn.split('#')[-1]) for cn in self.ees.colnames[2:]])
        self.radii = np.around(self.radii/.0396, 0) # convert to pixels
        # NOTE: rounding is done because the radii in the file are rounded to 2 decimals, so this gets
        # the integer pixel positions back.  This kludge is likely unnecessary when more precision is
        # available

        if norm_rad is None:
            self.rad = self.radii[-1]
        else:
            self.rad = norm_rad

        self.ee_array = self._get_ee(filt)
        self.ee_interp = interp1d(self.radii, self.ee_array, fill_value=1., bounds_error=False)

        self.jpsf_data = fits.getdata(jpsf_file)

        self.jrad = int((self.jpsf_data.shape[0]-1)/2)

        self._norm_jpsf(self.jrad)
        self.generate_image()


    def _get_ee(self, filt):
        """
        Gets the EE curve for filt as an array from the
        Table.
        """
        rowind = np.where(self.ees['FILTER'] == self.filt.upper())[0][0]
        row = self.ees[rowind]
        vals = [row[cn] for cn in self.ees.colnames[2:]]
        vals = [0] + vals
        vals = np.array(vals)
        return vals

    def _jee(self, jrad=100.):
        """"""
        ris = np.arange(1., jrad)
        center = (jrad, jrad)
        aps = [CircularAperture(center, r=ri) for ri in ris]

        phots = aperture_photometry(self.jpsf_data, aps, method='exact')
        jee = np.array([phots[cn][0] for cn in phots.colnames[3:]])
        return jee

    def _norm_jpsf(self, jrad=100.):
        """
        Pins the EE of empirical PSF to the EE curve value
        at the radius of the empirical PSF.
        """
        jee = self._jee()
        psf_rat = self.ee_interp(jrad)/jee[-1]
        self.jpsf_data *= psf_rat

    def _pixval(self, dist):
        """
        Calculates pixel value of a pixel some distance
        from center of star, assuming radially symmetric
        PSF.

        This computes the energy inside an annulus at
        r_in = dist -.5 and r_out = dist + .5 and divides
        by the area of the annulus to get average energy
        per pixel in the annulus.  This breaks down for
        the central pixels due to the approximations made,
        but is corrected in via 'correct_psf()'.
        """
        # Calculate delta EE from a 1 pixel step
        inner = np.amax([0., dist-.5])
        outer = dist+.5
        dEE = self.ee_interp(outer) - self.ee_interp(inner)

        # divide delta EE by area of annulus
        avpix = dEE/(np.pi*(outer**2. - inner ** 2.))
        return avpix

    def generate_image(self, radius=None):
        """
        This is the way to generate the Jay PSF as an
        image. There's probably a smarter way to do this.
        """
        if radius is None:
            radius = int(self.rad)
        print("creating Jay PSF image")
        self._cenx = float(radius)
        self._ceny = float(radius)
        self.center = (self._cenx, self._ceny)

        self.psf = np.zeros((2*radius+1, 2*radius+1))
        jrad = self.jrad
        self.psf[radius-jrad:radius+jrad+1, radius-jrad:radius+jrad+1] = self.jpsf_data

        for i in range(0, 2*radius+1):
            for j in range(0, 2*radius+1):
                dist = np.sqrt((i-self._cenx)**2. + (j-self._ceny)**2.)
                if dist <= 100.:
                    continue
                else:
                    self.psf[j, i] = self._pixval(dist)

    def calc_jay_ee(self):
        """
        This creates an encircled energy curve for the
        Jay PSF using apertures of increasing radii.

        Returns
        -------
        jay_ee : `numpy.array`
            Array of encircled energy curve values.
        """
        self._ris = np.arange(1., self.rad)
        aps = [CircularAperture(self.center, r=ri) for ri in self._ris]

        phots = aperture_photometry(self.psf, aps, method='exact')
        jay_ee = np.array([phots[cn][0] for cn in phots.colnames[3:]])
        return jay_ee

    def plot_ees(self):
        """
        This just plots the real JayPSF EE
        """
        psf_ee = self.calc_jay_ee()
        plt.plot(self.radii[1:], self.ee_array[1:], label='real EE')
        plt.plot(self._ris, psf_ee, label='JPSF PSF EE')



def make_scan_line(img_shape, xstart, xend, ystart, yend):
    """Construct line to convolve PSF with.

    This makes a line in a numpy array.  Because the
    line_aa function antialiases lines they could be made
    at an angle, though, this might not preserve the "flux"
    represented by the line across the whole scan (i.e.
    the antialiasing probably makes it brighter by adding
    soft edges).  Might be best to keep the lines straight
    as a result, or might not matter.  Probably the latter.

    MM Note: Insignificant difference between convolving with
    vertical line and convolving with a line 4 degrees from
    the vertical. Since these calibration scans are less than
    1 degree tilted with respect to the vertical, then we're
    fine.

    Parameters
    ----------
    image_shape : (int, int)
        Shape of image to be convolved (ypix, xpix). Should
        probably leave enough space for the edges of the
        kernel not to be cut off. i.e. if feature in image
        is 200 pixels, and kernel size is 300x300, then
        dimension should be aat least 200+150+150 = 500 to
        not cut off edges.
    xstart: int
        x-position of line start, in pixels.
    xend: int
        x-position of line end, in pixels.
    ystart: int
        y-position of line start, in pixels.
    yend: int
        y-position of line end, in pixels.


    Returns
    -------
    img : `numpy.ndarray`
        A numpy array with pixels along line set to 1, and
        potential antialiased pixels set to the antialiased
        value.
    """
    img = np.zeros(img_shape, dtype=np.float64)

    rr, cc, val = line_aa(ystart, xstart, yend, xend)
    img[rr, cc] = val

    return img

def setup_dirs(save_dir_name):
    """
    Helper functions to create directories as needed.

    Parameter
    ---------
    save_dir_name : str
        Name of directory in `synphot_dir` where files
        should be saved.

    Returns
    -------
    psf_dir : str
        String representation of path where PSF files
        should be saved.
    ssf_dir : str
        String representation of path where SSF files
        should be saved.
    """
    save_dir = check_subdirectory(parent_dir=synphot_dir,
                                  sub_name=save_dir_name)

    psf_dir = check_subdirectory(parent_dir=save_dir,
                                 sub_name="psf")

    ssf_dir = check_subdirectory(parent_dir=save_dir,
                                 sub_name="ssf")

    return psf_dir, ssf_dir

def make_convolvedpsf(psf_type, filt, ee_table, uvis_name, save_dir_name):
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
    uvis_name : str
        Which CCD; either `uvis1` or `uvis2`.
    save_dir_name : str
        Name of directory in `synphot_dir` where files
        should be saved.
    """
    psf_dir, ssf_dir = setup_dirs(save_dir_name)

    if psf_type == 'simple':
        res = ToyPSF(filt, ee_table)
        res.plot_ees()
        fixed_ee = res.calc_toy_ee()

        psf_fname = f'{uvis_name}_{filt}_psf.csv'

        np.savetxt(f'{psf_dir}/{psf_fname}', res.psf, delimiter=',')
        print(f'File {psf_fname} saved.')

        # spatial scans for this program are 92 pixels long
        # so we center that in the image and find the start
        # and end points of the line to be 160 and 352
        img = make_scan_line((512,512), 256, 256, 160, 352)
        out = convolve2d(img, res.psf, mode='same')

        ssf_fname = f'{uvis_name}_{filt}_convolvedpsf.csv'

        np.savetxt(f'{ssf_dir}/{ssf_fname}', out, delimiter=',')
        print(f'File {ssf_fname} saved.')

    else:
        jpsf_dir = os.path.join(synphot_dir, "jpsf")
        jpsf_file = f'{jpsf_dir}/psfnrm_{filt}.fits'

        res = JayPSF(filt, ee_table, jpsf_file)
        res.plot_ees()
        fixed_ee = res.calc_jay_ee()

        psf_fname = f'uvis2_{filt}_jpsf.csv'

        np.savetxt(f'{psf_dir}/{psf_fname}', res.psf, delimiter=',')
        print(f'File {psf_fname} saved.')

        img = make_scan_line((512,512), 256, 256, 160, 352)
        out = convolve2d(img, res.psf, mode='same')

        ssf_fname = f'jay_uvis2_{filt}_convolvedpsf.csv'

        np.savetxt(f'{ssf_dir}/{ssf_fname}', out, delimiter=',')
        print(f'File {ssf_fname} saved.')

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
    parser = ArgumentParser(prog='uvis_make_ssf',
                            description='creates a scan spread function from '\
                                        'published EE values & empirical PSF models',
                            epilog = 'Authors: Mariarosa Marinelli & Varun Bajaj')

    parser.add_argument("-t", "--type",
                        choices=["simple", "blended"],
                        help="type of convolved PSF to make: simple or blended",
                        required=True)
    parser.add_argument("-d", "--dir_name",
                        help="directory name in /grp/hst/wfc3v/wfc3photom/data"\
                              "/uvis_scan_monitor/synphot/",
                        required=True)

    parser.add_argument("-f", "--filters",
                        nargs="+",
                        help="filter or list of filters (default is `all`)",
                        default=["all"])
    parser.add_argument("-u", "--uvis",
                        help="either 1, 2, or both",
                        choices=['1', '2', 'both'],
                        default="both")

    args = parser.parse_args()

    return args


def clean_cl_args(args):
    """
    Helper function to clean up command line arguments.
    (Probably there's a better way to do this but I'll
    stick with what works for now.)

    Parameter
    ---------
    args : `argparse.Namespace`
        Object where the attributes correspond to the
        arguments given at the command line (and the
        default values for optional arguments, if
        applicable).

    Returns
    -------
    filters : list of str
        List of WFC3/UVIS filters to create SSFs for.
    uvises : list of str
        Which CCDs to make SSFs for. Either [`uvis1`],
        [`uvis`], or [`uvis1`, `uvis2`].
    """
    all_ee_filters = ['F218W', 'F225W', 'F275W', 'F336W', 'F438W', 'F606W', 'F814W']
    all_jpsf_filters = ['F275W', 'F336W', 'F438W', 'F606W', 'F814W']

    if args.filters == ['all']:
        if args.type == 'simple':
            filters = all_ee_filters
        else:
            filters = all_jpsf_filters

    else:
        filters = args.filters
        if args.type == "simple":
            for filt in filters:
                if filt not in all_ee_filters:
                    print("Non-core filter listed. Removing "\
                          f"{filt} from filter list...")
                    filters.remove(filt)
        else:
            for filt in filters:
                if filt not in all_jpsf_filters:
                    print("No empirical PSF is available for "\
                          f"{filt}. Removing from filter list...")
                    filters.remove(filt)

    uvis_dict = {'1': ['uvis1'],
                 '2': ['uvis2'],
                 'both': ['uvis1', 'uvis2']}

    uvises = uvis_dict[args.uvis]

    return filters, uvises


if __name__ == '__main__':

    args = parse_args()
    filters, uvises = clean_cl_args(args)

    for uvis in uvises:
        ee_from_stare = Table.read(f'{synphot_dir}/wfc3{uvis}_aper_007_syn.csv',
                                   format='csv')

        for filt in filters:
            make_convolvedpsf(psf_type=args.type,
                              filt=filt,
                              ee_table=ee_from_stare,
                              uvis_name=uvis,
                              save_dir_name=args.dir_name)
