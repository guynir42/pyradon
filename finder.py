import sys
import os
import time
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Union

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from streak import Streak, upsample
from frt import FRT


class Finder:
    """
    Streak finding tool.
    Give it some images, using self.input(images, ...)
    This will do all the internal calculations needed to find streaks
    inside each frame. Optional arguments to "input" are:
     -variance: give a scalar (average) variance *or* a variance map.
     -psf: give the point spread function as a scalar width (in this
      case the PSF is a 2D Gaussian with sigma equal to the input scalar)
      *or* a map for the image.
     -frame_num: housekeeping parameter saved to output Streak objects.
     Helps keep track of where each streak was found.
     -batch_num: housekeeping parameter saved to output Streak objects.
     Helps keep track of where each streak was found.
     -filename: housekeeping parameter saved to output Streak objects.
     Helps keep track of where each streak was found.

    SWITCHES AND SETTINGS (see __init__() for more details)
    -Image pre-processing: use_subtract_mean, use_conv.
    -Search options: use_short, min_length
    -Post processing: use_exclude, exclude_dx, exclude_dy
    -memory considerations: use_save_images, use_clear_memory.
    -housekeeping: filename, batch_num, frame_num.
    -display defaults: use_show, display_index, display_which, line_offset,
     rect_size, display_monochrome.

    ALGORITHMS:
    1) Individual images are scanned with the Fast Radon Transform (FRT)
       that does something similar to FFT to sum pixel values along straight
       lines, finding the peaks and saving them as Streak objects.
       The input images must be matched-filtered using an estimate
       of the system PSF. This is also true for point-source detection.
       The FRT of the input image must be divided by an FRT of a
       variance map to normalize the noise and the number of pixels
       in the streak, before any S/N calculations can be made.
       See https://ui.adsabs.harvard.edu/abs/2018AJ....156..229N/abstract
    2) To make sure all streaks are found, regardless of brightness,
       there are some steps you need to take.
       Each image must be background subtracted, and once streaks
       are found they need to be subtracted (and b/g resubtracted)
       before searching for other, possibly fainter streaks.
       This object will search a set of thresholds, starting with
       the S/N of the brightest objects in the image, and scanning
       with a threshold that is 2 times smaller on every iteration.
       Before applying a smaller threshold, all pixels in the image
       with point-wise S/N above half the threshold are truncated.
       This removes artefacts and stars, but does not remove streaks,
       that have an aggregated S/N that is above threshold, even
       though each pixel in the streak is below threshold.

    NOTES:
        -the input image can be altered by the process (streak subtraction).
         If you need that image, make a copy of it before inputting.
        -the Radon image that is saved in each streak is already normalized
         by the variance map.
        -If you remove stars or streaks that were found in previous iterations,
         make sure to replace them with NaN. Then use_subtract_mean will ignore
         these pixels when finding the mean. Finally, the NaNs are replaced
         by zero after reducing the mean and before entering the transfrom.

    """

    @dataclass
    class Pars:
        """
        A class to hold the user-parameters for the Finder class.
        """

        # image pre-processing
        use_subtract_mean: bool = True  # subtract any residual bias
        use_conv: bool = True  # match filter with PSF

        # cut the incoming images into sections to run faster
        use_sections: bool = False
        size_sections: int = 1024  # can be a scalar or a 2-tuple
        # can use this to trim the image before making sections
        offset_sections: Union[int, Tuple[int, int]] = 0

        # search options
        use_short: bool = True  # search for short streaks
        min_length: int = 32  # minimal length (along the axis) for short streaks
        threshold: float = 5  # in units of S/N

        # times to search same image for streaks (per threshold, per section)
        num_iterations: int = 5
        use_exclude: bool = True
        exclude_x_pix: Optional[Tuple[int, int]] = (-50, 50)
        exclude_y_pix: Optional[Tuple[int, int]] = None

        use_show: bool = False  # display the images

        # save to disk the cutouts for any detected streaks
        use_write_cutouts: bool = False
        # only save cutouts for streaks shorter than this
        # set to None for no limits
        max_length_to_write: Optional[int] = 128

        # if no PSF is given, assume this as the width of a Gaussian PSF (pixels)
        default_psf_sigma: float = 1
        # if no variance is given, this is used as the variance scalar (counts^2)
        default_var_scalar: float = 1

        verbosity: int = 1  # level of output feedback (higher: more outputs, zero: silent operations)

        @property
        def use_expand(self):
            # when doing short streaks, no need to expand the axes
            return not self.use_short

    @dataclass
    class Data:
        """
        A class to hold the input/output/intermediate data for the Finder class
        """

        _pars = None  # must give the Pars object to allow default values of PSF and variance

        # images #
        image: np.ndarray = None  # image as given to finder (not altered in any way!)
        radon_image: np.ndarray = None  # final FRT result (normalized by the var-map)
        radon_image_tr: np.ndarray = None  # final FRT of the transposed image

        # S/N values found in the images
        # list of all S/N values found in this run (use "reset()" to clear them)
        snr_values: List[float] = field(default_factory=list)
        best_snr: float = 0.0

        # housekeeping variables #
        filename: Optional[str] = None  # which file we are currently scanning
        frame_num: Optional[int] = None  # which frame in the batch
        batch_num: Optional[int] = None  # which batch in the run
        section_num: Optional[int] = None  # which section in the current image

        # PSF width and image #
        # the map of the PSF, either as given or generated as a Gaussian from "sigma_psf"
        _psf_image: Optional[np.ndarray] = None
        # the width of a Gaussian PSF, either given as scalar or fit to PSF map
        _psf_scalar: Optional[float] = None

        # variance maps and scalars
        # either given as scalar or the median of the var map
        _var_scalar: Optional[float] = None
        # either given as map or just expanded from the scalar
        _var_image: Optional[np.ndarray] = None
        # did we expand the input variance map
        _expanded_var: Optional[bool] = None
        # which part of the var image did we use?
        _var_corner: Tuple[int, int] = (0, 0)
        # the size of the extracted section from the var image
        _var_size: Tuple[int, int] = (0, 0)
        # a dictionary keyed to a 6-tuple
        # (corner, corner, size, size, transpose, expand)
        # that keeps RadonVariance objects for each
        # section and transposition.
        # these are kept until a call to reset()
        # or when calling clear_var_cache()
        # or when the underlying variance map is changed.
        _radon_variance_cache: dict = field(default_factory=dict)

        # other things we keep track of
        # keep track of where the current section starts
        _current_section_corner: Tuple[int, int] = (0, 0)
        _num_frt_calls: int = 0

        @property
        def var_uniform(self):
            """
            If no variance image is given,
            the variance is a uniform map
            with some scalar value as average.
            """
            return self._var_image is None

        @property
        def variance(self):  # the variance value given by user (or the default value)
            if self.var_uniform:
                return self.var_scalar
            else:
                return self._var_image

        @variance.setter
        def variance(self, val: Union[None, float, np.ndarray]):
            # if working with scalar variance, only need to rescale the var maps
            if val is None:  # to clear the variance, set it to None
                self._var_scalar = None
                self._var_image = None
                self.clear_radon_var_cache()
                return

            if not np.isscalar(val) and not isinstance(val, np.ndarray):
                raise TypeError(
                    "Input to variance must be a scalar or np.ndarray. "
                    f"Got {type(val)} instead. "
                )

            if np.isscalar(val) and self.is_var_cache_uniform():
                for v in self._radon_variance_cache.values():
                    v.rescale(val)
            else:  # new or old variance is not scalar, must recalculate var maps
                self.clear_radon_var_cache()

            if np.isscalar(val):
                self._var_scalar = val
                self._var_image = None
            else:
                self._var_scalar = float(np.median(val))
                self._var_image = val

        @property
        def var_scalar(self):
            if self._var_scalar is not None:
                return self._var_scalar
            else:
                return self._pars.default_var_scalar

        def get_var_image(self, corner, size):
            if self.var_uniform:
                return self.var_scalar * np.ones(size)
            else:  # in this case, there MUST BE a _var_image
                return self._var_image[
                    corner[0] : corner[0] + size[0], corner[1] : corner[1] + size[1]
                ]

        def is_var_cache_uniform(self):
            """
            Assume that all items in this dictionary have the same
            value for the "uniform" field.
            If the cache is empty, can assume it is also uniform
            """
            return (
                not self._radon_variance_cache
                or next(iter(self._radon_variance_cache.values())).uniform
            )

        def get_radon_variance(self, sec_corner, sec_size, transpose, expand):
            """
            Get a Radon variance map of the section
            of the variance image given by the section
            corner and size, and whether that section was
            transposed and/or expanded.
            If such a list of radon maps exist, return it.
            If not, produce it and cache it for later use.

            Parameters
            ----------
            sec_corner: 2-tuple of int
                The indices of the lower left corner of the section
                that is now being processed.
                When not sectioning or trimming, this would be (0,0)
            sec_size: 2-tuple of int
                The size of the section of the image that is now
                being processed. If not sectioning this would just
                be equal to the size of the input image.
            transpose: scalar boolean
                Choose whether or not the image now being processed
                has been transposed or not.
            expand: scalar boolean
                Choose whether or not the image now being processed
                has been expanded by the FRT funtion or not.

            Returns
            -------
                A list of partial Radon transforms of the variance
                map (for the correct part of it, if sectioning)
                and with the proper transpose and expansion.
                This list should be used to normalize the
                partial Radon transforms from the FRT
                to get the S/N for any streaks.
            """

            # if the variance is made from a scalar,
            # and it is therefore uniform,
            # then there is no point in specifying the
            if self.var_uniform:
                sec_corner = (0, 0)
                if not self.is_var_cache_uniform():
                    self.clear_radon_var_cache()  # just to make sure cache is not non-uniform

            # this is the value we are keying on
            id_tuple = (*sec_corner, *sec_size, bool(transpose), bool(expand))

            # lazy calculate the radon maps for this combination
            if id_tuple not in self._radon_variance_cache:
                self._radon_variance_cache[id_tuple] = Finder.RadonVariance(
                    self.variance,
                    transpose,
                    expand,
                    sec_corner,
                    sec_size,
                )

            return self._radon_variance_cache[id_tuple].radon_var_list

        def clear_radon_var_cache(self):
            self._radon_variance_cache = {}

        @property
        def psf(self):
            if self._psf_image is None:
                self.psf = self.psf_sigma
            return self._psf_image

        @psf.setter
        def psf(self, val: Union[float, np.ndarray]):
            if np.isscalar(val):
                if val != self._psf_scalar:
                    self._psf_image = gaussian2D(val, norm=2)
                    self._psf_scalar = val
            elif isinstance(val, np.ndarray) and val.ndim == 2:
                if not np.ndarray(self._psf_image, val):
                    val /= np.sqrt(np.sum(val**2))
                    if not np.array_equal(self._psf_image, val):
                        self._psf_image = val
                        # fit_results = fit_gaussian(val)  # run a short minimization 2D fitter to gaussian
                        # print(fit_results)
                        # self._psf_scalar = (fit_results.x[1] + fit_results.x[2]) / 2  # average the x and y sigma
                        self._psf_scalar = gaussian_width(val)
            else:
                raise TypeError("psf must be a scalar or a 2D numpy array.")

        @property
        def psf_sigma(self):
            if self._psf_scalar is not None:
                return self._psf_scalar
            else:
                return self._pars.default_psf_sigma

        def clear_psf(self):
            self._psf_scalar = None
            self._psf_image = gaussian2D(self.psf_sigma, norm=2)

    class RadonVariance:
        """
        Keep track of the Radon transformed variance maps.
        Each instance of this class refers to one part of
        the input variance map (a section) or the whole map.
        It contains a list of partial transforms of the map,
        to be used against real images and their partial Radon
        transforms, to calculate the S/N.

        If the object is created from a uniform map
        (i.e., all the values are the same as the mean value)
        then it can also be rescaled without recalculating anything,
        in case the average variance has been changed.
        """

        def __init__(
            self, variance, transpose=False, expand=False, corner=(0, 0), im_size=None
        ):
            """
            Generate a cached Radon variance map to be
            used for calculating S/N of Radon images
            with potential streaks in them.

            Parameters
            ----------
            variance: scalar float or np.ndarray
                The input variance. Can be a scalar or a 2D array.
                If scalar, will generate a uniform variance map
                and transform that. In such cases the object
                can be rescaled easily.
                If given a 2D array (image) it will
                just transform a section of that and use it whenever
                that section is queried for a variance map.

            transpose: scalar boolean
                Choose if the input should be transposed
                before the transform is applied.
            expand: scalar boolean
                Choose if the input should be expanded
                before the transform is applied.
            corner: 2-tuple of int
                The bottom left corner of the subsection
                from the given variance that should be used
                to produce the Radon maps.
                If (0, 0), will just use the corner of the
                input variance map. If the input variance is
                a scalar, this has no effect.
            im_size: 2-tuple of int
                The size of the image that should be taken out
                of the original variance map, or in case of a
                scalar input variance, the size of the uniform
                variance map that should be then transformed
                to create the Radon variance maps.
                If None, will just use the top-right edge
                of the input variance image.
                If the variance is given as a scalar,
                the im_size cannot be None.

            """

            if np.isscalar(variance):
                self.uniform = True
                if im_size is None:
                    raise ValueError("Must supply an im_size if variance is a scalar.")
                self.var_map = np.ones(im_size) * variance
                self.var_scalar = variance
            elif isinstance(variance, np.ndarray):
                if np.ndim(variance) != 2:
                    raise ValueError(
                        "variance must be a 2D array. "
                        f"Got a {np.ndim(variance)}D array."
                    )
                self.uniform = False

                if im_size is None:
                    upper_corner = variance.shape
                else:
                    upper_corner = (im_size[0] + corner[0], im_size[1] + corner[1])

                self.var_map = variance[
                    corner[0] : upper_corner[0], corner[1] : upper_corner[1]
                ]
                self.var_scalar = np.nanmedian(self.var_map)

            else:
                raise TypeError(
                    "Input to variance parameter must be"
                    "either a scalar or an array. "
                    f"Instead got a {type(variance)}"
                )

            if transpose:
                self.var_map = self.var_map.T

            self.transposed = transpose

            self.radon_var_list = FRT(
                self.var_map,
                partial=True,
                expand=expand,
                transpose=transpose,
            )

            self.expanded = expand

        def rescale(self, new_var_scalar):
            if not self.uniform:
                raise RuntimeError("Cannot rescale a non-uniform RadonVariance object.")

            factor = new_var_scalar / self.var_scalar

            self.var_map *= factor

            for i, v in enumerate(self.radon_var_list):
                self.radon_var_list[i] = v * factor

            self.var_scalar = new_var_scalar

    def __init__(self, **kwargs):

        # all the user-defined parameters live here:
        self.pars = Finder.Pars(**kwargs)

        # all the input/outputs/intermediate data products live here:
        self.data = Finder.Data()
        self.data._pars = self.pars

        # objects or lists of objects
        # streaks saved from latest call to input()
        self.streaks: List[Streak] = field(default_factory=list)

        # a list of Streak objects that passed the threshold,
        # saved from all scans (use reset() to remove these)
        self.streaks_all: List[Streak] = field(default_factory=list)

        # keep track of the time when this object was initialized
        self._version_timestamp: float = time.time()

    # reset methods #
    def reset(self):
        """
        Reset all the long-persisting
        data fields and objects.
        Do this at the start of a new run.
        """

        self.data.snr_values = []
        self.data.total_runtime = 0
        self.streaks_all = []

        self.data.clear_radon_var_cache()
        self.data.clear_psf()

        self.clear()

    def clear(self):
        """
        Clears intermediate results.
        Do this each time you input new images.
        """
        self.data.image = None
        self.data.radon_image = None
        self.data.radon_image_tr = None
        self.data.best_snr = 0.0
        self.streaks = []

        self.data.batch_num = 0
        self.data.frame_num = 0
        self.data.section_num = 0

        self.data._current_section_corner = (0, 0)
        self.data._num_frt_calls = 0

    # getters #
    def get_radon_variance_old(self, transpose=False):
        """
        Get the partial Radon transforms of the background noise for some transpose.
        Lazy Reloading: only delete old var-maps if input size changed
        (or if we changed expansion mode) and then calculate the var-maps on demand.
        """
        # check if we need to recalculate the var map
        if (
            self.data.image is not None
            and self.data.var_image is not None
            and (
                self.data.image.shape != self.data.var_image.shape
                or self.pars.use_expand != self.data._expanded_var
            )
        ):
            if self.pars.verbosity > 1:
                print("Clearing the Radon var-maps")
            # clear these, to be lazy loaded with the right size
            self.data._radon_var_map = []
            self.data._radon_var_map_tr = []

        # if there is no var map, we need to lazy load it
        if not self.data._radon_var_map:
            self.data._expanded_var = self.pars.use_expand
            self.data._radon_var_map = FRT(
                self.data.var_image,
                partial=True,
                expand=self.data._expanded_var,
                transpose=False,
            )
            self.data._radon_var_map_tr = FRT(
                self.data.var_image,
                partial=True,
                expand=self.data._expanded_var,
                transpose=True,
            )

        if transpose:
            return self.data._radon_var_map_tr
        else:
            return self.data._radon_var_map

    def get_norm_factor_psf(self):
        """
        Factors the normalization of the PSF when calculating S/N
        """
        return np.sum(self.data.psf) * np.sqrt(np.sum(self.data.psf**2))

    def get_geometric_factor(self, foldings):
        """
        Gets the geometric factor that has to do
        with the slope of the streak inside each pixel.
        This depends on the stage in the algorithm
        (the "foldings") that determines the height
        of the current slab of the data we are working on.

        Parameters
        ----------
        foldings: scalar int
            The number of logarithmic folds the data has taken.
            There are 2**(foldings-1) lines of the original image
            in each sub-image that was folded this many times.

        Returns
        -------
        The geometric factor that adjusts the S/N of the results
        based on the streak angle through the pixels.
        """
        height = 2 ** (foldings - 1)
        th = np.arctan(np.arange(-height + 1, height) / float(height))
        geom_fact = np.maximum(np.fabs(np.cos(th)), np.fabs(np.sin(th)))
        geom_fact = geom_fact[:, np.newaxis, np.newaxis]
        return geom_fact

    # STREAK FINDING #
    def make_streak(self, snr, transpose, threshold, peak, foldings, subframe, section):
        """
        Generate a Streak object based on
        the data recovered from the Radon image.

        Parameters
        ----------
        snr: scalar float
            The adjusted S/N for this streak.
        transpose: scalar boolean
            Checks if the streak was found in the
            original image or in it's transpose.
        threshold: scalar float
            The threshold on S/N used to detect this streak.
        peak: 3-tuple of int
            The 3D index of the maximum of the subframe
            where the streak was found (encodes streak position).
        foldings: scalar int
            The number logarithmic folds the data has undergone
            to allow detection of this streak.
            There are 2**(foldings-1) lines of the original
            image in each sub-image with this many foldings.
        subframe: np.ndarray of floats
            Partially transformed image where Streak was found.
            For long streaks this is the same as the final
            Radon image.
        section: np.ndarray of floats
            ?

        Returns
        -------
        A Streak object, with all values calculated.
        """
        s = Streak(
            snr=snr,
            transpose=transpose,
            threshold=threshold,
            peak=peak,
            foldings=foldings,
            subframe=subframe,
            section=section,
        )

        s.update_from_finder(self)  # get additional values not given in __init__()
        s.calculate()  # calculate all internal quantities for this streak

        return s

    def find_single(self, im, transpose=False, threshold=None):
        """
        Find a single streak in a single image.

        Parameters
        ----------
        im: 2D np.ndarray of floats (or None)
            The image to be scanned for streaks.
            Should be 2D, with stars and background
            fully subtracted.
        transpose: scalar boolean
            Has this image been transposed.
            If so, that has some bearing on
            the output Streak object (e.g.,
            replacing x->y values).
            This DOES NOT APPLY A TRANSPOSE
            on the input image, but assumes
            it has already been transposed.
        threshold: scalar float
            The minimal S/N for finding streaks.

        Returns
        -------
        A single Streak object, if found.
        If no streaks are detected, returns None.
        If input "im" is None, silently returns None.
        If a streak is detected, it is also added to
        self.streaks, that can be cleared using
        self.clear().

        Note on iterations
        ------------------
        This can be called multiple times on the same
        image (or section of the image).
        Using transpose, and a cascade of different thresholds,
        and removing each individual streak can allow
        identification of multiple streaks in the same image.
        In that case, find_single() will be called many times.
        Use self.data._num_frt_calls to figure out how many
        calls to this function were done on the entire image.
        (this number is set to zero in self.clear()).

        """
        if im is None or len(im) == 0:  # why do we need this short-circuit?
            return None

        if threshold is None:
            threshold = self.pars.threshold  # use default

        streak = None
        # self.data.image = im
        self.data._num_frt_calls += 1

        radon_variance_maps = self.data.get_radon_variance(
            self.data._current_section_corner,
            im.shape,
            transpose,
            self.pars.use_expand,
        )

        if self.pars.use_short:
            # these are raw Radon partial transforms:
            radon_partial = FRT(im, transpose=transpose, partial=True, expand=False)

            # divide by the variance map, geometric factor, and PSF norm for each level
            # m counts the number of foldings, partials start at 2
            geometric_factors = [
                self.get_geometric_factor(m) for m in range(2, len(radon_partial) + 2)
            ]

            psf_factor = self.get_norm_factor_psf()

            # correct the radon images for all these factors
            for i in range(len(radon_partial)):
                radon_partial[i] /= np.sqrt(
                    radon_variance_maps[i] * geometric_factors[i] * psf_factor
                )

            # get the final Radon image as 2D map
            radon_image = radon_partial[-1][:, 0, :]

            # best index for each folding
            snrs_idx = [np.nanargmax(r) for r in radon_partial]
            # snrs_max = np.array([r[i] for i, r in zip(snrs_idx, radon_partial)]) # best S/N for each folding
            # best S/N for each folding
            snrs_max = np.array([np.nanmax(r) for r in radon_partial])

            best_idx = np.nanargmax(snrs_max)  # which folding has the best S/N
            best_snr = snrs_max[best_idx]  # what is the best S/N of all foldings

            if best_snr >= threshold and 2**best_idx >= self.pars.min_length:
                # the x,y,z of the peak in that subframe:
                peak_coord = np.unravel_index(
                    snrs_idx[best_idx], radon_partial[best_idx].shape
                )

                streak = self.make_streak(
                    snr=best_snr,
                    transpose=transpose,
                    threshold=threshold,
                    peak=peak_coord,
                    foldings=best_idx + 2,
                    subframe=radon_partial[best_idx],
                    section=im,
                )

        else:  # don't use short streaks

            radon_image = FRT(im, transpose=transpose, partial=False, expand=True)

            # the length tells you how many foldings we need
            foldings = len(radon_variance_maps) + 1

            # get the last folding and flatten it to 2D
            radon_variance = radon_variance_maps[-1][:, 0, :]
            geom_factor = self.get_geometric_factor(foldings)
            psf_factor = self.get_norm_factor_psf

            radon_image /= np.sqrt(radon_variance * geom_factor * psf_factor)

            # this is how it would look from a partial transpose output
            radon_partial = radon_image[:, np.newaxis, :]

            idx = np.argmax(radon_image)
            best_snr = radon_image[idx]

            peak_coord = np.unravel_index(idx, radon_image.shape)
            # added zero for y start position that is often non-zero in the partial transforms
            peak_coord = (peak_coord[0], 0, peak_coord[1])

            if best_snr >= threshold:
                streak = self.make_streak(
                    snr=best_snr,
                    transpose=transpose,
                    threshold=threshold,
                    peak=peak_coord,
                    foldings=foldings,
                    subframe=radon_partial,
                    section=im,
                )

        # this will always have the best S/N until "clear" is called
        self.data.best_snr = max(best_snr, self.data.best_snr)

        if streak:
            self.streaks.append(streak)
            if self.pars.use_write_cutouts:
                if (
                    self.pars.max_length_to_write is None
                    or streak.L < self.pars.max_length_to_write
                ):
                    streak.write_to_disk()

        # store the final FRT result
        # (this is problematic once we start iterating over find_single!)
        if transpose:
            self.data.radon_image_tr = radon_image
        else:
            self.data.radon_image = radon_image

        if self.pars.verbosity > 1:
            print(
                f"Running FRT {self.data._num_frt_calls} times, "
                f"transpose= {transpose}, thresh= {threshold:.2f}, "
                f"found streak: {bool(streak)}"
            )

        return streak

    def find_multi(self, im, threshold=None, num_iter=None):
        """
        Run streak detection on the same image (or subsection)
        multiple times, for both the transposed and original image.
        At each iteration, any streaks found are removed from
        the original image, before calling find_single() again.
        The original and transposed images each get processed
        up to "num_iter" times. If no streaks are found, the
        iterations are cut short (but the image is always
        checked at least twice, for the two transpositions).
        Streak detection is done on a single threshold value.

        Parameters
        ----------
        im: np.ndarray of floats
            An image (or subset of images) to be scanned
            for streaks. The image will be modified
            by removing streaks and adjusting the mean
            background level.
        threshold: scalar float
            The minimal S/N for streaks to be detected in
            the given image. Defaults to self.pars.threshold (7.5).

        num_iter: scalar int
            The maximum number of streaks to find in each
            transposition of the image. When no streaks are
            found, the number of iterations is smaller.
            Defaults to self.pars.num_iterations (5).

        Returns
        -------
        The image given, after subtracting any found streaks.

        """
        if threshold is None:
            threshold = self.pars.threshold

        if num_iter is None:
            num_iter = self.pars.num_iterations

        for trans in [False, True]:
            for i in range(num_iter):
                new_streak = self.find_single(im, transpose=trans, threshold=threshold)
                if not new_streak:
                    break
                else:
                    new_streak.subtract_streak(im)  # remove the streak from input image
                    if self.pars.use_subtract_mean:
                        im -= np.nanmean(im)

                    if self.pars.use_show:
                        new_streak.plot_lines()
                        plt.show()
                        # fig = plt.gcf()
                        # fig.canvas.draw()
                        # fig.canvas.flush_events()
        return im

    def scan_thresholds(self, im, min_threshold=None):
        """
        Iteratively call find_multi() on several values of
        the threshold. This enables detection of both
        bright and faint streaks in the same image.
        Before running a scan with a given threshold,
        pixels with point-wise S/N above half the threshold
        are truncated, leaving only streaks able to pass the
        threshold because the aggregated S/N of their pixels
        is generally higher than the S/N of each pixel.

        Any streaks that are found are subtracted inside
        the find_multi() function.

        Parameters
        ----------
        im: np.ndarray of floats
            An image (or subset of images) to be scanned
            for streaks. The image will be modified
            by removing streaks and adjusting the mean
            background level.
        min_threshold: scalar float
            The minimal threshold level to scan.
            This determines the total number of iterations
            and also the faintest streaks to be detected.
            Defaults to self.pars.threshold.

        Returns
        -------
        The image is returned after all adjustments
        and streak subtraction. The input image is
        modified directly, but returned as a convenience.
        """

        if min_threshold is None:
            min_threshold = self.pars.threshold

        var_image = self.data.get_var_image(self.data._current_section_corner, im.shape)
        mx = np.nanmax(im / np.sqrt(var_image))

        dynamic_range = np.log2(mx / min_threshold)

        thresholds = np.flip(min_threshold * 2 ** np.arange(dynamic_range + 1))
        if self.pars.verbosity > 1:
            print(
                f"mx= {mx:.2f} | dynamic_range= {dynamic_range:.2e} "
                f"| thresholds: {np.round(thresholds, 2)}"
            )

        for t in thresholds:
            mask = np.zeros(im.shape, dtype=bool)
            np.greater(
                im / np.sqrt(var_image), t / 2, where=np.isnan(im) == 0, out=mask
            )
            # clip to threshold (scaled by noise):
            im[mask] = t / 2 * np.sqrt(self.data.var_scalar)
            if self.pars.use_subtract_mean:
                im -= np.nanmean(im)

            if self.pars.use_show:
                plt.clf()
                plt.imshow(im)
                plt.title(f"section corner: {self.data._current_section_corner}")
                plt.xlabel(f"psf_sigma= {self.data.psf_sigma:.2f} | threshold= {t:.2f}")
                plt.show()
                # f = plt.gcf()
                # f.canvas.draw()
                # f.canvas.flush_events()

            im = self.find_multi(im, t)

        return im

    def preprocess(self, im):
        """
        Run basic processing required before submitting the image
        into the streak detection code.
        The two operations are
        1) background subtraction (by removing the image mean).
           This only happens if self.pars.use_subtract_mean=True.
        2) Matched-filter using an estimate of the image PSF.
           This is done by running a 2D convolution using the PSF
           image (which defaults to a Gaussian with sigma=2).

        Parameters
        ----------
        im: np.ndarray of floats
            The image to be processed before sending it to
            the streak detection algorithm.

        Returns
        -------
        Another np.ndarray of the same size as "im",
        after background subtraction and PSF filtering.

        """

        if self.pars.use_subtract_mean:
            im = im - np.nanmean(im)  # remove the mean
        # np.nan_to_num(im, copy=False)
        # should we also remove cosmic rays/bad pixels at this level?

        im_conv = scipy.signal.convolve2d(im, self.data.psf, mode="same")

        return im_conv

    # def scan_sections(self):  # to be depricated!
    #     """
    #
    #     """
    #     corners = []
    #     sections = jigsaw(self.image, self.size_sections, output_corners=corners)
    #
    #     for i in range(sections.shape[0]):
    #         this_section = self.preprocess(sections[i])
    #         self.current_section_corner = corners[i]
    #         # treat the var map for each section right here!
    #
    #         self.scanThresholds(this_section)

    # User interface #

    def input(
        self,
        image,
        variance=None,
        psf=None,
        filename=None,
        batch_num=None,
        frame_num=None,
    ):
        """
        Input an image and search for streaks in it.
        This function runs the full suite of processing
        the Finder object can run, doing all the work
        to find as many streaks at multiple brightness
        levels as possible.
        Ideally, this should be the direct function called
        by the user.

        Parameters
        ----------
        image: np.ndarray
            The image to be processed.
        variance: scalar float or 2D np.ndarray of floats
            An estimate of the noise variance, including
            noise from the background and const sources.
            Can be a scalar representing the average noise,
            or a map of the same size as the input image.
            If not given, will use the existing variance map
            (i.e., it can be given once for a series of different
            images) or use the default in self.pars.default_var_scalar.
        psf: scalar float or 2D np.ndarray of floats
            The Point Spread Function of the image (or images).
            Can give either a scalar (gaussian sigma) or a 2D np.ndarray.
            The array is usually smaller than the input image.
        filename: string
            Used for tracking the origin of any discovered streaks.
        batch_num: scalar int
            Used for tracking the origin of any discovered streaks.
            Useful if we are running many batches in this run.
        frame_num: scalar int
            Used for tracking the origin of any discovered streaks.
            Useful if each batch contains multiple frames/images.

        """

        if not isinstance(image, np.ndarray):
            raise Exception("Cannot do streak finding without an image!")

        self.clear()  # get rid of intermediate results

        self.data.image = image

        # input the variance, if given!
        if variance:
            self.data.variance = variance

        # input the PSF if given!
        if psf:
            self.data.psf = psf

        # housekeeping
        self.data.filename = filename
        self.data.batch_num = batch_num
        self.data.frame_num = frame_num

        corners = []  # this can be modified by jigsaw()

        if self.pars.use_sections:
            sections = jigsaw(
                image,
                self.pars.size_sections,
                self.pars.offset_sections,
                output_corners=corners,
            )
        else:
            sections = np.expand_dims(np.copy(image), 0)

        for i in range(sections.shape[0]):

            if corners:
                self.data._current_section_corner = corners[i]
            else:
                self.data._current_section_corner = (0, 0)

            sec = sections[i, :, :]
            sec = self.preprocess(sec)
            self.scan_thresholds(sec)

        # must return this to zero in case future users
        # of this object want to use unsectioned images
        self.data._current_section_corner = (0, 0)

        if self.pars.use_show:
            plt.clf()
            h = plt.imshow(self.data.image)
            h.set_clim(0, 5 * np.sqrt(self.data.var_scalar))
            [streak.plot_lines(im_type="full") for streak in self.streaks]
            plt.title("full frame image")
            plt.xlabel(self.data.filename)
            f = plt.gcf()
            f.canvas.draw()
            f.canvas.flush_events()


def gaussian2D(
    sigma_x=2.0,
    sigma_y=None,
    rotation_degrees=0,
    offset_x=0,
    offset_y=0,
    size=None,
    norm=1,
):
    """
    Generate an image of a gaussian distrubtion.

    Parameters
    ----------
    sigma_x: scalar float
        The width parameter of the gaussian.
        If sigma_y is not given, the shape is
        circularly symmetric (i.e., sigma_y=sigma_x)
        If both sigma_x and sigma_y are given,
        then this parameter controls the x-axis.

    sigma_y: scalar float
        The width parameter for the y axis.
        If not given (or None) will just be
        equal to the sigma_x parameter.

    rotation_degrees: scalar float
        The rotation angle to be applied to the
        x and y axes of the gaussian.

    offset_x: scalar float
        The number of pixels of offset to the gaussian center
        in the x direction (after it had been rotated).

    offset_y: scalar float
        The number of pixels of offset to the gaussian center
        in the y direction (after it had been rotated).

    size: scalar int or 2-tuple of int
        Shape of the image to be generated.
        If None (default) will use 20 times the
        longer axis sigma parameter.

    norm: scalar int
        The power of the values to use as normalization.
        A few options exists:
        0: no normalization at all, the peak is equal to 1.
        1: the sum of the gaussian is equal to 1.
        2: the sqrt of the sum of squares is equal to 1.

    Returns
    -------
        A two-dimensional array (image) of a gaussian.
    """
    if sigma_y is None:
        sigma_y = sigma_x

    if size is None:
        size = max(sigma_x, sigma_y) * 20

    if np.isscalar(size):
        size = (size, size)

    size = (round(size[0]), round(size[1]))

    (y0, x0) = np.indices(size, dtype="float32")

    x0 -= size[1] / 2
    y0 -= size[0] / 2

    rotation_radians = np.radians(rotation_degrees)

    x = x0 * np.cos(rotation_radians) + y0 * np.sin(rotation_radians) - offset_x
    y = -x0 * np.sin(rotation_radians) + y0 * np.cos(rotation_radians) - offset_y

    output_gaussian = np.exp(-0.5 * ((x / sigma_x) ** 2 + (y / sigma_y) ** 2))
    if norm == 1:
        output_gaussian /= np.sum(output_gaussian)
    elif norm == 2:
        output_gaussian /= np.sqrt(np.sum(output_gaussian**2))

    return output_gaussian


def gaussian_width(im):
    """
    Estimate the width of an image "im" with
    a 2D gaussian, using the FWHM of the image
    and calculating the gaussian sigma from that.
    """

    sizes = np.array(im.shape)
    factor = np.round(min(300.0 / sizes))
    if factor > 1:
        im = upsample(im, factor)  # make sure to calculate this on an upsampled image

    pix_above_half = np.sum(im > np.max(im) * 0.5)
    fwhm = 2 * np.sqrt(pix_above_half / np.pi)  # assume circular peak

    return fwhm / 2.355 / factor


def jigsaw(im, cut_size, trim_corner=None, pad_value=None, output_corners=None):
    """
    Cut an image into small cutouts and return them in a 3D array.

    Parameters
    ----------
    im: np.ndarray
        A 2D image that needs to be cut into pieces.

    cut_size: scalar int or 2-tuple of int
        The size of the output cutouts.
        Can be a scalar (in which case the cutouts are square)
        or a 2-tuple so that the shape of each cutout is equal
        to this parameter.

    trim_corner: scalar int or 2 tuple of int
        Number of rows and columns to trim from the initial
        image before cutting it into pieces.
        The corner specifies the bottom left pixel
        where the cutting should begin.
        If given as a scalar, will be copied into
        a 2-tuple with identical values.
        If None (default) will not trim anything,
        i.e., equal to (0,0).
        Note that the top right edge of the image
        may still be trimmed if the cut_size does
        not fit an integer number of times in the
        image size.

    pad_value: scalar float or None
        If given a value, will use that as
        a filler for any part of the cutouts
        that lie outside the original image
        (this can happen if the last pixels
        reach out of the image or if the
        trim_corner has negative values).
        If None, any cutouts that have any
        pixels outside the original image
        will not be included in the output.

    output_corners: list
        If not None, use this list to output
        tuples of the corners of each cutout.
        The list is expected to be empty when
        the function is called.

    Returns
    -------
        A 3D array where the first dimension is the number of cutouts,
        and the other two dimensions are the cutout height and width
        (equal to the input cut_size).
    """

    S = im.shape  # size of the input
    if np.isscalar(cut_size):
        C = (cut_size, cut_size)
    elif isinstance(cut_size, tuple) and len(cut_size) == 2:
        C = cut_size
    else:
        raise TypeError("cut_size must be a scalar or 2-tuple")

    if trim_corner is None:
        T = (0, 0)
    else:
        if np.isscalar(trim_corner):
            T = (trim_corner, trim_corner)
        elif isinstance(trim_corner, tuple) and len(trim_corner) == 2:
            T = trim_corner
        else:
            raise TypeError("trim_corner must be a scalar or 2-tuple")

    left = np.arange(
        T[1], S[1], C[1]
    )  # x position of the corner of each cutout in the input image
    right = left + C[1]
    x = list(zip(left, right))
    bottom = np.arange(
        T[0], S[0], C[0]
    )  # y position of the corner of each cutout in the input image
    top = bottom + C[0]
    y = list(zip(bottom, top))

    num_cut = len(x) * len(y)  # number of cutouts

    if pad_value is None:  # get rid of any coordinates outside the image
        for i, coords in enumerate(x):
            if coords[0] < 0 or coords[1] >= S[1]:
                del x[i]
        for i, coords in enumerate(y):
            if coords[0] < 0 or coords[1] >= S[0]:
                del y[i]

        num_cut = len(x) * len(y)  # number of cutouts after culling
        im_out = np.empty((num_cut, C[0], C[1]))  # make an array to hold viable cutouts
    elif np.isnan(pad_value):  # generate a NaN padded output
        im_out = np.empty((num_cut, C[0], C[1]), dtype=im.dtype)
        im_out[:] = np.nan
    elif pad_value == 0:  # generate a zero padded output
        im_out = np.zeros((num_cut, C[0], C[1]), dtype=im.dtype)
    else:  # generate a scalar value padded output
        im_out = np.ones((num_cut, C[0], C[1]), dtype=im.dtype) * pad_value

    num_cut = len(x) * len(y)  # number of cutouts

    counter = 0
    for cy in y:
        for cx in x:
            low_x = max(cx[0], 0)  # in case it is negative
            high_x = min(cx[1], S[1])  # in case bigger than image
            low_y = max(cy[0], 0)  # in case it is negative
            high_y = min(cy[1], S[0])  # in case bigger than image

            im_out[
                counter, low_y - cy[0] : high_y - cy[0], low_x - cx[0] : high_x - cx[0]
            ] = im[low_y:high_y, low_x:high_x]

            if output_corners is not None:
                output_corners.append((cy[0], cx[0]))

            counter += 1

    return im_out


if __name__ == "__main__":
    f = Finder()
    f.pars.threshold = 3
    f.pars.use_show = True
    im = np.random.normal(0, 1, (512, 512))
    # f.find_single(im)
    # f.find_multi(im)
    f.input(im)
