import os
import time
import re
import h5py
import numpy as np
from numpy.fft import fft2, ifft2, fftshift
import scipy.signal
import matplotlib.pyplot as plt


class Streak:
    """
    This object describes a single streak that is found using the FRT,
    including the raw results of the finding algorithm, and the streak
    parameters derived from the results.

    Usually a Finder object will have a list of these objects saved
    after it has finished going over some images.

    The Streak object can optionally keep a copy of the original image and
    the Radon image where it was detected, as well as housekeeping data such
    as the filename, batch number and frame number of that image.

    The Streak object can also be used to subtract the streak from the input image,
    which is useful for iteratively finding faint streaks next to bright ones.

    """

    def __init__(
        self,
        snr=None,
        transpose=False,
        threshold=None,
        peak=None,
        foldings=None,
        subframe=None,
        section=None,
    ):

        self.image = None  # subtracted image, as given to finder.
        self.radon_image = None  # full Radon image for the correct transposition.
        # subframe where streak is detected. For non-short streaks, equal to "radon_image".
        self.subframe = subframe
        # smaller region of the original image, or just equal to original image (if no sectioning is done)
        self.image_section_raw = None
        # same as section, only PSF filtered, and streak subtracted (all processing is applied)
        self.image_section_proc = section
        self.corner_section = (0, 0)  # the corner of the section inside the full image
        self.size_section = None  # size of section (2-tuple)
        # size of section after transpose (if not transposed, equal to size_section)
        self.size_section_tr = None
        self.image_cutout = None  # small region around streak
        self.cut_size = 128
        # self.image_cutout_proc = None  # small region around streak with full processing applied
        self.corner_cutout = None  # the corner of the cutout inside the full image
        self.size_cutout = None  # size of the cutout (2-tuple)
        # the PSF image itself, normalized, that was used to do the filter.
        self.psf = None

        # these help find the correct image where the streak exists
        self.frame_num = None  # which frame in the batch.
        self.batch_num = None  # which batch in the run.
        self.filename = None  # which file it came from.

        self.threshold = threshold  # which threshold was used.
        self.is_short = None  # if yes, subframe is smaller than the full Radon image.

        # how bright the streak was
        self.I = None  # intensity, or brightness per unit length.
        self.snr = snr  # calculated from subframe and index.
        self.snr_fwhm = None  # S/N per resolution element
        self.count = None  # how many photons across the whole streak.

        # calculated parameters
        self.L = None  # length of streak (pixels).
        self.th = None  # angle of streak (th=0 is on the y axis).
        self.x0 = None  # intercept of line with the x axis.
        self.a = None  # slope parameter (y=ax+b).
        self.b = None  # intercept parameter (y=ax+b).
        self.x1 = None  # streak starting point in x.
        self.x2 = None  # streak end point in x.
        self.y1 = None  # streak starting point in y.
        self.y2 = None  # streak end point in y.
        self.dx = None  # difference of x's
        self.dy = None  # difference of y's

        # raw coordinates from the Radon result / subframe
        self.transposed = transpose  # was the image transposed?
        self.foldings = foldings  # in what step in the FRT the streak was found.
        self.radon_max_idx = peak  # 3D index of the maximum of the subframe (streak position).
        self.radon_x0 = None  # position coordinate in the Radon image.
        self.radon_dx = None  # slope coordinate in the Radon image.
        self.radon_x_var = None  # error estimate on "radon_x"
        self.radon_dx_var = None  # error estimate on radon_dx"
        self.radon_xdx_cov = None  # cross correlation of the slope-position errors
        self.radon_y1 = None  # start of the subframe
        self.radon_y2 = None  # end of the subframe

        # switches and parameters for user
        self.var_scalar = None  # mean value of the variance
        self.psf_sigma = None  # PSF width, equivalent to gaussian sigma
        # how many PSF widths to remove around streak position (overriden by finder)
        self.subtract_psf_widths = 3

        # internal switches and parameters
        self.was_expanded = False  # check if original image was expanded before FRT
        self.was_convolved = False  # check if original image was convolved with PSF
        # rough estimate of the region around the Radon peak we want to cut for error estimates
        self.num_psfs_peak_region = 5
        # how many S/N units below maximum is still inside the peak region
        self.num_snr_peak_region = 2
        # a map of the peak region,
        # with only the part above the cut
        # (peak-num_snr_peak_region)
        # not zeroed out (used for error estimates)
        self.peak_region = None

        # keep track of the time when this object was initialized
        self._version_timestamp: float = time.time()

    @property
    def radon_dy(self):
        """Difference between the y coordinates in Radon space."""
        return self.radon_y2 - self.radon_y1

    @property
    def x1f(self):
        """Position of x1 (streak left) in full image."""
        return self.x1 + self.corner_section[1]

    @property
    def x2f(self):
        """Position of x2 (streak right) in full image."""
        return self.x2 + self.corner_section[1]

    @property
    def y1f(self):
        """Position of y1 (streak bottom) in full image."""
        return self.y1 + self.corner_section[0]

    @property
    def y2f(self):
        """Position of y2 (streak top) in full image."""
        return self.y2 + self.corner_section[0]

    @property
    def x1c(self):
        """Position of x1 (streak left) in cutout."""
        if self.corner_cutout is None:
            return None
        else:
            return self.x1 + self.corner_section[1] - self.corner_cutout[1]

    @property
    def x2c(self):
        """Position of x2 (streak right) in cutout."""
        if self.corner_cutout is None:
            return None
        else:
            return self.x2 + self.corner_section[1] - self.corner_cutout[1]

    @property
    def y1c(self):
        """Position of y1 (streak left) in cutout."""
        if self.corner_cutout is None:
            return None
        else:
            return self.y1 + self.corner_section[0] - self.corner_cutout[0]

    @property
    def y2c(self):
        """Position of y2 (streak top) in cutout."""
        if self.corner_cutout is None:
            return None
        else:
            return self.y2 + self.corner_section[0] - self.corner_cutout[0]

    @property
    def mid_x(self):
        """Center of the streak, in x coordinate."""
        if self.x1 is None or self.x2 is None:
            return None
        else:
            return (self.x1 + self.x2) / 2

    @property
    def mid_y(self):
        """Center of the streak, in y coordinate."""
        if self.y1 is None or self.y2 is None:
            return None
        else:
            return (self.y1 + self.y2) / 2

    @property
    def mid_x_full(self):
        """Center of the streak, in x coordinate, in the full image."""
        if self.x1f is None or self.x2f is None:
            return None
        else:
            return (self.x1f + self.x2f) / 2

    @property
    def mid_y_full(self):
        """Center of the streak, in x coordinate, in the full image."""
        if self.y1f is None or self.y2f is None:
            return None
        else:
            return (self.y1f + self.y2f) / 2

    def update_from_finder(self, finder):
        """load all attributes in self that exist in finder.data or finder.pars"""
        if finder is None:  # quietly skip if no finder is given
            return

        for att in self.__dict__.keys():
            if hasattr(finder.data, att) and not callable(getattr(finder.data, att)):
                setattr(self, att, getattr(finder.data, att))

            if hasattr(finder.pars, att) and not callable(getattr(finder.pars, att)):
                setattr(self, att, getattr(finder.pars, att))

        self.corner_section = finder.data._current_section_corner

        self.was_convolved = bool(finder.pars.use_conv)
        self.was_expanded = bool(finder.pars.use_expand)

    def calculate(self):
        """
        Translate the raw coordinates form the Radon plane
        into usable x/y coordinates, length, and angle
        of the streak.
        Also figure out the streak brightness, S/N, etc.
        """
        # these are the raw results from this subframe
        self.size_section = self.image_section_proc.shape
        if self.transposed:
            self.size_section_tr = tuple(reversed(self.size_section))
        else:
            self.size_section_tr = self.size_section

        # part of the orignal image that is defined as the cutout (without processing!)
        if self.image is not None:
            self.image_section_raw = self.image[
                self.corner_section[0] : self.size_section[0],
                self.corner_section[1] : self.size_section[1],
            ]

        # size of each slice in y, times the slice number (index)
        self.radon_y1 = (2 ** (self.foldings - 1)) * self.radon_max_idx[1]
        # same thing, top index (should we include the last pixel?)
        self.radon_y2 = (2 ** (self.foldings - 1)) * (self.radon_max_idx[1] + 1)
        # this is added on either side if was_expanded
        offset = (self.subframe.shape[2] - self.size_section_tr[1]) // 2
        # position of x0 in the subframe (removing the expanded pixels)
        self.radon_x0 = self.radon_max_idx[2] - offset
        # the angle is in dim0, needs to offset for negative angles
        self.radon_dx = self.radon_max_idx[0] - self.subframe.shape[0] // 2

        # signal to noise from the subframe maximum
        self.snr = self.subframe[self.radon_max_idx]
        self.is_short = self.subframe.ndim > 2 and self.subframe.shape[1] > 1

        # assume there is no transpose (then add it if needed)
        self.y1 = self.radon_y1
        self.y2 = self.radon_y2
        self.dy = self.y2 - self.y1
        self.dx = self.radon_dx
        if self.radon_dx != 0:
            self.a = self.dy / self.dx
            self.x0 = self.radon_x0 - self.y1 / self.a
            self.b = -self.a * self.x0
            self.th = np.degrees(np.arctan(self.a))
            self.x1 = (self.y1 - self.b) / self.a
            self.x2 = (self.y2 - self.b) / self.a
        else:
            self.a = np.nan
            self.x0 = self.radon_x0
            self.b = np.nan
            self.th = 90
            self.x1 = self.radon_x0
            self.x2 = self.radon_x0

        self.L = abs(self.radon_dy / np.sin(np.radians(self.th)))
        self.I = self.snr * np.sqrt(self.var_scalar * 2 * np.sqrt(np.pi) * self.psf_sigma / self.L)
        self.snr_fwhm = self.I * 0.81 / np.sqrt(self.var_scalar)

        if self.transposed:
            self.x1, self.y1 = self.y1, self.x1
            self.x2, self.y2 = self.y2, self.x2
            self.a = 1 / self.a
            self.b, self.x0 = self.x0, self.b
            self.th = 90 - self.th

        # self.x1 = round(self.x1)
        # self.x2 = round(self.x2)
        # self.y1 = round(self.y1)
        # self.y2 = round(self.y2)
        # self.x0 = round(self.x0)

        # calculate the errors on the Radon parameters
        radon_subframe = self.subframe[:, self.radon_max_idx[1], :]  # as a 2D image
        xmax = self.radon_max_idx[2]
        ymax = self.radon_max_idx[0]

        # size of area of a few PSF widths around the peak
        size_pix = round(self.psf_sigma * self.num_psfs_peak_region)
        x1 = int(max(xmax - size_pix, 0))
        x2 = int(min(xmax + size_pix, radon_subframe.shape[1] - 1))
        y1 = int(max(ymax - size_pix, 0))
        y2 = int(min(ymax + size_pix, radon_subframe.shape[0] - 1))
        # a copy of the region around the streak, in Radon space
        peak_region = np.copy(radon_subframe[y1:y2, x1:x2])
        xgrid, ygrid = np.meshgrid(range(peak_region.shape[1]), range(peak_region.shape[0]))

        idx = np.unravel_index(np.nanargmax(peak_region), peak_region.shape)
        mx = peak_region[idx]

        peak_region[peak_region < mx - self.num_snr_peak_region] = 0

        xgrid = xgrid - idx[1]
        ygrid = ygrid - idx[0]

        radon_sum = np.sum(peak_region)
        self.radon_x_var = np.sum(peak_region * xgrid**2) / radon_sum
        self.radon_dx_var = np.sum(peak_region * ygrid**2) / radon_sum
        self.radon_xdx_cov = np.sum(peak_region * xgrid * ygrid) / radon_sum

        self.peak_region = peak_region

        self.make_cutout()

    def make_cutout(self, cut_size=None):
        """
        Generate a cutout around the position of the
        streak in the image. The result is saved in
        self.image_cutout.

        Parameters
        ----------
        cut_size: scalar int or 2-tuple of int
            Size of the small cutout around the streak,
            in units of pixels.
            If None, will default to self.cut_size.
            If given as a scalar, will create a
            square cutout.
            If a 2-tuple, will generate a rectangle
            with a shape equal to the size input.

        """
        if self.image is None:
            return

        if cut_size is None:
            cut_size = self.cut_size

        if np.isscalar(cut_size):
            cut_size = (cut_size, cut_size)
        elif not isinstance(cut_size, tuple) or len(cut_size) != 2:
            raise TypeError('Input "cut_size" must be a scalar or 2-tuple')

        d1 = tuple(int(np.floor(s / 2)) for s in cut_size)
        d2 = tuple(int(np.ceil(s / 2)) for s in cut_size)

        x1 = int(round(self.mid_x_full)) - d1[1]
        x2 = int(round(self.mid_x_full)) + d2[1]
        y1 = int(round(self.mid_y_full)) - d1[0]
        y2 = int(round(self.mid_y_full)) + d2[0]

        self.image_cutout = self.image[y1:y2, x1:x2]

        self.size_cutout = cut_size
        self.corner_cutout = (y1, x1)

    def write_to_disk(self, filename=None):
        """
        Write the cutout image to an HDF5 file.
        The file's group will be named streak_<number>
        where the <number> is a serial number that
        increases every time a streak is saved.
        If the file doesn't have any such groups,
        the numbering will start at 0.

        Parameters
        ----------
        filename: str
            The full path and filename to write to.
            If None, will use self.filename as the
            path and name of the output file, but
            replace the extension to .h5
            (i.e., generate an auxiliary file).

        """
        if filename is None:
            if self.filename is None:
                return
            else:
                filename = os.path.splitext(self.filename)[0] + ".h5"

        with h5py.File(filename, "a") as hf:
            numbers = []
            for k in hf.keys():
                string = re.search(r"streak_\d+$", k).group()
                numbers.append(int(string.replace("streak_", "")))

            if not numbers:
                new_number = 0
            else:
                new_number = max(numbers)

            ds = hf.create_dataset(f"streak_{new_number:03d}", data=self.image_cutout)

            ds.attrs["corner"] = self.corner_cutout
            # add additional metadata later on

    def subtract_streak(self, im, replace_value=np.nan, image_type="subsection"):
        """
        Replace any pixels touched by this streak with
        a new value (default is NaN).
        The pixels "touched" are defined by instantiating
        a streak model with this streak's parameters
        with some margin given by a few times the
        width of the PSF.
        See the documentation for model().

        Parameters
        ----------
        im: np.array of floats
            The image given, will be modified IN PLACE
            to remove the streak.

        replace_value: float int
            The value to put instead of the streak pixels.
            Default is NaN, which lets the finder use
            background subtraction, no counting this pixels.
            other popular choices are zero or the b/g mean value.

        image_type: str
            The type of image to use for the streak model.
            Can be 'subsection' or 'full' or 'cutout'.
        """

        if image_type == "subsection":
            mask = model(im.shape, self.x1, self.x2, self.y1, self.y2, self.psf_sigma) > 0
        elif image_type == "full":
            mask = model(im.shape, self.x1f, self.x2f, self.y1f, self.y2f, self.psf_sigma) > 0
        elif image_type == "cutout":
            mask = model(im.shape, self.x1c, self.x2c, self.y1c, self.y2, self.psf_sigma) > 0
        else:
            raise ValueError(f"image_type {image_type} not recognized... use 'subsection', 'full', or 'cutout'. ")

        im[mask] = replace_value

    def show(self, ax=None, **kwargs):
        """
        Show the processed image, with the streak
        highlighted with two lines on either side of it.

        Parameters
        ----------
        ax: matplotlib.axes._subplots.AxesSubplot
            The current axes to plot into.
            If not given (or None), will
            default to using plt.gca().

        """
        if ax is None:
            ax = plt.gca()

        ax.imshow(self.image_section_raw, **kwargs)
        self.plot_lines(ax)

    def plot_lines(self, ax=None, offset=10, im_type="section", line_format="--m", linewidth=2.0):
        """
        Show two guiding lines around the position of this streak
        placed on top of an already existing image.

        Parameters
        ----------
        ax: matplotlib.axes._subplots.AxesSubplot
            The axes object to plot into.
            If not given, or None, will
            default to plt.gca().
        offset: scalar int
            Number of pixels to offset the
            lines from the center of the streak
        im_type: str
            Choose the image that is plotted,
            can be "section" (default), "full" or "cutout".
        line_format: str
            Specify the line format used in the plot function.
            Default is "--m" for magenta dashed lines.
        linewidth: scalar float
            Width of the plotted lines.
            Default is 2.0.

        """
        if ax is None:
            ax = plt.gca()

        if im_type == "section":
            x1 = self.x1
            x2 = self.x2
            y1 = self.y1
            y2 = self.y2
        elif im_type == "full":
            x1 = self.x1f
            x2 = self.x2f
            y1 = self.y1f
            y2 = self.y2f
        elif im_type == "cutout":
            x1 = self.x1c
            x2 = self.x2c
            y1 = self.y1c
            y2 = self.y2c
        else:
            raise KeyError(f'Unknown im_type: "{im_type}". Use "section" or "full" or "cutout".')

        if self.transposed:
            ax.plot([x1, x2], [y1 + offset, y2 + offset], line_format, linewidth=linewidth)
            ax.plot([x1, x2], [y1 - offset, y2 - offset], line_format, linewidth=linewidth)
        else:
            ax.plot([x1 + offset, x2 + offset], [y1, y2], line_format, linewidth=linewidth)
            ax.plot([x1 - offset, x2 - offset], [y1, y2], line_format, linewidth=linewidth)

    def print(self):
        print(
            f"CALCULATED: S/N= {self.snr:.2f} | "
            f"I= {self.I:.2f} | L= {self.L:.1f} | "
            f"th= {self.th:.2f} | x0= {self.x0:.2f} "
        )


def model(im_size, x1, x2, y1, y2, psf_sigma=2, replace_value=0, threshold=1e-10, oversample=4):
    """
    Generate a model streak using the given coordinates,
    and the PSF sigma width.
    The streak is normalized to
    The streak is embedded in an image
    of size given by the im_size parameter.
    Pixels far from the streak that have very low
    values are replaced with zero (or another value).

    Parameters
    ----------
    im_size: scalar int or 2-tuple of int
        Size of the image to produce, in pixels.
        Can be a scalar, in which case the output
        is a square image. Otherwise the image
        would have a shape equal to im_size.
    x1: scalar float
        Initial x position of the streak,
        in image coordiantes.
    x2: scalar float
        Final x position of the streak,
        in image coordinates.
    y1: scalar float
        Initial y position of the streak,
        in image coordinates.
    y2: scalar float
        Final y position of the streak,
        in image coordinates.
    psf_sigma: scalar float
        The width of the gaussian PSF
        for the streak model.
        Default is 2 pixels.

    replace_value: scalar float
        Value to replace pixels that are
        very far from the streak core,
        that would have values below
        the threshold.
        Default is zero.
        Another popular choice is np.nan.
        If None, will not replace anything.

    threshold: scalar float
        A small value that signifies which
        pixels are too far away from the streak
        core and should be replaced.
        Default is 1e-10.
        If None, will not replace anything.

    oversample: scalar int
        The image is produced with every
        aspect of the model scaled up,
        and only after placing the streak
        the image is down-sampled to the
        expected size.
        This allows higher resolution
        of the streak values to be
        effectively integrated on
        each pixel by averaging the
        values from the high-resolution
        grid, into the final, low-resolution
        image. Default is 4.

    Returns
    -------
        An image with a streak embedded into it.
    """
    if np.isscalar(im_size):
        im_size = (im_size, im_size)
    elif not isinstance(im_size, tuple) or len(im_size) != 2:
        raise TypeError('Input "im_size" must be a scalar or 2-tuple')

    if not all(isinstance(s, int) for s in im_size):
        raise TypeError('Input "im_size" must have only int type values. ')

    if oversample:
        im_size = tuple(s * oversample for s in im_size)
        x1 = (x1 - 0.5) * oversample + 0.5
        x2 = (x2 - 0.5) * oversample + 0.5
        y1 = (y1 - 0.5) * oversample + 0.5
        y2 = (y2 - 0.5) * oversample + 0.5
        psf_sigma = psf_sigma * oversample

    #    (x,y) = np.meshgrid(range(im_size[1]), range(im_size[0]), indexing='xy')
    (y, x) = np.indices(im_size, dtype="float32")

    if x1 == x2:
        a = float("Inf")  # do we need this?
        b = float("NaN")  # do we need this?
        d = np.abs(x - x1)  # distance from vertical line
    else:
        a = (y2 - y1) / (x2 - x1)  # slope parameter
        b = (y1 * x2 - y2 * x1) / (x2 - x1)  # impact parameter
        d = np.abs(a * x - y + b) / np.sqrt(1 + a**2)  # distance from line

    # an image of an infinite streak with gaussian width psf_sigma
    im0 = (1 / np.sqrt(2.0 * np.pi) / psf_sigma) * np.exp(-0.5 * d**2 / psf_sigma**2)

    # must clip this streak:
    if x1 == x2 and y1 == y2:  # this is extremely unlikely to happen...
        im0 = np.zeros(im0.shape)
    elif x1 == x2:  # vertical line (a is infinite)
        if y1 > y2:
            im0[y > y1] = 0
            im0[y < y2] = 0
        else:
            im0[y < y1] = 0
            im0[y > y2] = 0

    elif y1 == y2:  # horizontal line
        if x1 > x2:
            im0[x > x1] = 0
            im0[x < x2] = 0
        else:
            im0[x < x1] = 0
            im0[x > x2] = 0

    elif y1 < y2:
        im0[y < (-1 / a * x + y1 + 1 / a * x1)] = 0
        im0[y > (-1 / a * x + y2 + 1 / a * x2)] = 0
    else:
        im0[y > (-1 / a * x + y1 + 1 / a * x1)] = 0
        im0[y < (-1 / a * x + y2 + 1 / a * x2)] = 0

    # make point-source gaussians at either end of the streak
    im1 = (1 / np.sqrt(2 * np.pi) / psf_sigma) * np.exp(-0.5 * ((x - x1) ** 2 + (y - y1) ** 2) / psf_sigma**2)
    im2 = (1 / np.sqrt(2 * np.pi) / psf_sigma) * np.exp(-0.5 * ((x - x2) ** 2 + (y - y2) ** 2) / psf_sigma**2)

    # "attach" the point sources by finding the maximum pixel value
    out_im = np.fmax(im0, np.fmax(im1, im2))

    if oversample > 1:
        out_im = downsample(out_im, oversample) / oversample

    # apply threshold and replace value
    if threshold is not None and replace_value is not None:
        out_im[out_im < threshold] = replace_value

    return out_im


def downsample(im, factor=2, normalization="sum"):
    """
    Scale down an image by first smoothing
    it with a square kernel, then subsampling
    the pixels in constant steps.
    Both the kernel size and steps are
    equal to the downsample factor.

    Parameters
    ----------
    im: np.array
        The image that should be downsampled.
        The original image is not altered.

    factor: scalar int
        How many pixels need to be combined
        to produce each of the output pixels.
        The input image is effectively scaled
        down by this number.

    normalization: str
        Choose "sum" or "mean" to determine
        if pixels that are combined should
        be simply summed (default) or averaged.

    Returns
    -------
        A smaller image array, where the scale
        of the image is smaller by "factor".

    """

    if factor is None or factor < 1:
        return im

    if not isinstance(factor, int):
        raise TypeError('Input "factor" must be a scalar integer. ')

    k = np.ones((factor, factor), dtype=im.dtype)
    if normalization == "mean":
        k = k / np.sum(k)
    elif normalization != "sum":
        raise KeyError('Input "normalization" must be "mean" or "sum". ' f'Got "{normalization}" instead. ')

    im_conv = scipy.signal.convolve2d(im, k, mode="same")

    return im_conv[factor - 1 :: factor, factor - 1 :: factor]


def upsample(im, factor=2):
    """
    Use FFT interpolation (sinc interp) to up-sample
    the given image by a factor (default 2).
    """
    before = [int(np.floor(s * (factor - 1) / 2)) for s in im.shape]
    after = [int(np.ceil(s * (factor - 1) / 2)) for s in im.shape]

    im_f = fftshift(fft2(fftshift(im)))
    im_pad_f = np.pad(im_f, [(before[0], after[0]), (before[1], after[1])])
    im_new = fftshift(ifft2(fftshift(im_pad_f)))

    return im_new
