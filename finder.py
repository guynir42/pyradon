# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 15:43:28 2017

@author: guyn
"""
import sys
import os
import time
import math
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Union

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from streak import Streak
from frt import FRT
from utils import empty, scalar, compare_size, imsize, crop2size, gaussian2D, fit_gaussian, gaussian_width, jigsaw


class Finder:
    """
    Streak finding tool. 
    Give it some images, using self.input(images, ...)
    This will do all the internal calculations needed to find streaks
    inside each frame. Optional arguments to "input" are:
     -variance: give a scalar (average) variance *or* a variance map. 
      At some point we may also allow giving a 3D matrix of
      variance maps for each image in the batch. 
     -psf: give the point spread function as a scalar width (in this 
      case the PSF is a 2D Gaussian with sigma equal to the input 
      scalar) *or* a map for the image. PSF can be 3D matrix with a 
      number of slices as the number of images. 
     -batch number: housekeeping parameter. Helps keep track of where 
      each streak was found.
     -filename: housekeeping parameter. Helps keep track of where 
      each streak was found.
   
    SWITCHES AND SETTINGS (see comments in properties block for more details)
    -Image pre-processing: use_subtract_mean, use_conv, use_crop_image, crop_size.
    -Search options: use_short, min_length
    -Post processing: use_exclude, exclude_dx, exclude_dy
    -memory considerations: use_save_images, use_clear_memory.
    -housekeeping: filename, batch_num, frame_num.
    -display defaults: show_bit, display_index, display_which, line_offset,
     rect_size, display_monochrome.

    NOTES: -the Radon image that is saved in each streak is already normalized
          by the variance map.
         -If you remove stars or streaks that were found in previous iterations,
          make sure to replace them with NaN. Then use_subtract_mean will ignore
          these pixels when finding the mean. Finally, the NaNs are replaced
          by zero after reducing the mean.
 
    """

    @dataclass
    class Pars:
        """
        A class to hold the user-parameters for the Finder class.
        """
        # image pre-processing
        use_subtract_mean: bool = True  # subtract any residual bias
        use_conv: bool = True  # match filter with PSF
        use_crop: bool = False  # do we want to crop the input images (e.g., to power of 2)
        crop_size: int = 2048  # size to crop to (does not grow the array)

        use_sections: bool = False  # cut the incoming images into sections to run faster
        size_sections: int = 1024  # can be a scalar or a 2-tuple

        # search options
        use_short: bool = True  # search for short streaks
        min_length: int = 32  # minimal length (along the axis) for short streaks
        threshold: float = 5  # in units of S/N
        num_iterations: int = 5  # how many times to go over the same image to look for streaks (per threshold/per section)
        use_exclude: bool = True
        exclude_x_pix: Optional[Tuple[int, int]] = (-50, 50)
        exclude_y_pix: Optional[Tuple[int, int]] = None

        use_show: bool = True  # display the images
        use_save_images: bool = True  # ...

        use_write_cutouts: bool = False  # ...
        max_length_to_write: int = 128  # ...

        default_psf_sigma: float = 1  # if no PSF is given, assume this as the width of a Gaussian PSF
        default_var_scalar: float = 1  # if no variance is given, this is used as the variance scalar

        verbosity: int = 1

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
        image: np.array = None  # image as given to finder
        im_size: Optional[Tuple[int, int]] = None  # two-element tuple
        radon_image: np.array = None  # final FRT result (normalized by the var-map)
        radon_image_tr: np.array = None  # final FRT of the transposed image (normalized)

        # S/N values found in the images
        # list of all S/N values found in this run (use "reset()" to clear them)
        snr_values: List[float] = field(default_factory=list)
        best_snr: float = 0.0

        # housekeeping variables #
        filename: Optional[str] = None  # which file we are currently scanning
        batch_num: int = 0  # which batch in the run
        frame_num: int = 0  # which frame in the batch
        section_num: int = 0  # which section in the current image

        # PSF width and image #
        # the map of the PSF, either as given or generated as a Gaussian from "sigma_psf"
        _psf_image: Optional[np.array] = None
        # the width of a Gaussian PSF, either given as scalar or fit to PSF map
        _psf_scalar: Optional[float] = None

        # variance maps and scalars
        _var_scalar: Optional[float] = None  # either given as scalar or the median of the var map
        _var_image: Optional[np.array] = None  # either given as map or just expanded from the scalar
        _expanded_var: Optional[bool] = None  # did we expand the input variance map

        # list of partial Radon variances from either a var_scalar or var_map
        _radon_var_map: List[np.array] = field(default_factory=list)
        _radon_var_map_tr: List[np.array] = field(default_factory=list)  # same thing, for the transposed FRT

        # other things we keep track of
        _current_section_corner: Tuple[int, int] = (0, 0)  # keep track of where the current section starts
        _num_frt_calls: int = 0

        @property
        def variance(self):  # the variance value given by user (or the default value)
            if self._var_image is not None:
                return self._var_image
            else:
                return self.var_scalar

        @variance.setter
        def variance(self, val: Union[float, np.array]):
            # if working with scalar variance, only need to rescale the var maps
            if scalar(val) and self._var_scalar is not None:
                if self._radon_var_map:  # list of length > 0
                    self._radon_var_map = [im * val / self._var_scalar for im in self._radon_var_map]
                if self._radon_var_map_tr:  # list of length > 0
                    self._radon_var_map_tr = [im * val / self._var_scalar for im in self._radon_var_map_tr]

            else:  # new or old variance is not scalar, must recalculate var maps
                self.clear_var_map()

            if scalar(val):
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

        @property
        def var_image(self):
            if self._var_image is not None:
                return self._var_image
            elif self.im_size is not None:
                return self.var_scalar * np.ones(self.im_size)
            else:  # cannot make a map without an image size
                return None

        @property
        def psf(self):
            return self._psf_image

        @psf.setter
        def psf(self, val: Union[float, np.array]):
            if scalar(val):
                if val != self._psf_scalar:
                    self._psf_image = gaussian2D(val, norm=2)
                    self._psf_scalar = val
            elif isinstance(val, np.ndarray) and val.ndim == 2:
                if not np.array_equal(self._psf_image, val):
                    val /= np.sqrt(np.sum(val ** 2))
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

        def clear_var_map(self):  # when we switch to a new image frame
            self._var_scalar = None  # either given as scalar or the median of the var map
            self._var_image = None  # either given as map or just expanded from the scalar
            self._expanded_var = None  # did we expand the input variance map
            self._radon_var_map = []  # list of partial Radon variances from either a var_scalar or var_map
            self._radon_var_map_tr = []  # same thing, for transposed FRT

        def clear_psf(self):
            self._psf_scalar = None
            self._psf_image = gaussian2D(self.psf_sigma, norm=2)

    def __init__(self, **kwargs):

        # all the user-defined parameters live here:
        self.pars = Finder.Pars(**kwargs)

        # all the input/outputs/intermediate data products live here:
        self.data = Finder.Data()

        # objects or lists of objects
        self.streaks: List[Streak] = field(default_factory=list)  # streaks saved from latest call to input()

        # a list of Streak objects that passed the threshold,
        # saved from all scans (use reset() to remove these)
        self.streaks_all: List[Streak] = field(default_factory=list)

        # keep track of the time when this object was initialized
        self._version_timestamp: float = time.time()

    # reset methods #
    def reset(self):  # do this at the start of a new run

        self.data.snr_values = []
        self.data.total_runtime = 0
        self.streaks_all = []

        self.data.clear_var_map()
        self.data.clear_psf()

        self.clear()

    def clear(self):  # do this each time we have new images

        self.data.image = None
        self.data.im_size = None
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
    def get_radon_variance(self, transpose=False):
        """
        Get the partial Radon transforms of the background noise for some transpose.
        Lazy Reloading: only delete old var-maps if input size changed (or
        if we changed expansion mode) and then calculate the var-maps on demand.
        """

        # check if we need to recalculate the var map
        if (
                self.data.im_size != self.data.var_image.shape
                or self.pars.use_expand != self.data._expanded_var
        ):
            if self.pars.verbosity > 1:
                print("Clearing the Radon var-maps")
            self.data._radon_var_map = []  # clear this to be lazy loaded with the right size
            self.data._radon_var_map_tr = []

        # if there is no var map, we need to lazy load it
        if not self.data._radon_var_map:
            self.data._expanded_var = self.useExpand()
            self.data._radon_var_map = FRT(self.data.var_image, partial=True, expand=self._expanded_var, transpose=False)
            self.data._radon_var_map_tr = FRT(self.data.var_image, partial=True, expand=self._expanded_var, transpose=True)

        if transpose:
            return self._radon_var_map_tr
        else:
            return self._radon_var_map

    def get_norm_factor_psf(self):
        """
        Factors the normalization of the PSF when calculating S/N
        """
        return np.sum(self.psf) * np.sqrt(np.sum(self.psf ** 2))

    def get_geometric_factor(self, foldings):
        """
        Gets the geometric factor that has to do
        with the slope of the streak inside each pixel.
        This depends on the stage in the algorithm
        (the "foldings") that determines the height
        of the current slab of the data we are working on.
        """
        height = 2 ** (foldings - 1)
        th = np.arctan(np.arange(-height + 1, height) / float(height))
        geom_fact = np.maximum(np.fabs(np.cos(th)), np.fabs(np.sin(th)))
        geom_fact = geom_fact[:, np.newaxis, np.newaxis]
        return geom_fact

    # STREAK FINDING #
    def make_streak(self, snr, transpose, threshold, peak, foldings, subframe, section):
        """

        """
        s = Streak(snr=snr, transpose=transpose, threshold=threshold, peak=peak,
                   foldings=foldings, subframe=subframe, section=section)

        s.update_from_finder(self)
        s.calculate()

        return s

    def find_single(self, im, transpose=False, threshold=None):
        """

        """
        if im is None or len(im) == 0:
            return None

        self.im_size = im.shape

        if threshold is None:
            threshold = self.pars.threshold  # use default

        streak = None

        self.data._num_frt_calls += 1

        if self.pars.use_short:
            # these are raw Radon partial transforms:
            radon_partial = FRT(im, transpose=transpose, partial=True, expand=False)

            # divide by the variance map, geometric factor, and PSF norm for each level
            radon_variance_maps = self.get_radon_variance(transpose)

            # m counts the number of foldings, partials start at 2
            geometric_factors = [self.get_geometric_factor(m) for m in range(2, len(radon_partial) + 2)]

            psf_factor = self.get_norm_factor_psf()

            for i in range(len(radon_partial)):  # correct the radon images for all these factors
                radon_partial[i] /= np.sqrt(radon_variance_maps[i] * geometric_factors[i] * psf_factor)

            radon_image = radon_partial[-1][:, 0, :]  # get the final Radon image as 2D map

            snrs_idx = [np.nanargmax(r) for r in radon_partial]  # best index for each folding
            # snrs_max = np.array([r[i] for i, r in zip(snrs_idx, radon_partial)]) # best S/N for each folding
            snrs_max = np.array([np.nanmax(r) for r in radon_partial])  # best S/N for each folding

            best_idx = np.nanargmax(snrs_max)  # which folding has the best S/N
            best_snr = snrs_max[best_idx]  # what is the best S/N of all foldings

            if best_snr >= threshold and 2 ** best_idx >= self.pars.min_length:
                # the x,y,z of the peak in that subframe:
                peak_coord = np.unravel_index(snrs_idx[best_idx], radon_partial[best_idx].shape)

                streak = self.make_streak(snr=best_snr, transpose=transpose, threshold=threshold, peak=peak_coord,
                                          foldings=best_idx + 2, subframe=radon_partial[best_idx], section=im)

        else:

            radon_image = FRT(im, transpose=transpose, partial=False, expand=True)

            radon_variance = self.get_radon_variance(transpose)
            foldings = len(radon_variance) + 1  # the length tells you how many foldings we need

            radon_variance = radon_variance[-1][:, 0, :]  # get the last folding and flatten it to 2D
            geom_factor = self.get_geometric_factor(foldings)
            psf_factor = self.get_norm_factor_psf

            radon_image /= np.sqrt(radon_variance * geom_factor * psf_factor)

            radon_partial = radon_image[:, np.newaxis, :]  # this is how it would look from a partial transpose output

            idx = np.argmax(radon_image)
            best_snr = radon_image[idx]

            peak_coord = np.unravel_index(idx, radon_image.shape)
            # added zero for y start position that is often non-zero in the partial transforms
            peak_coord = (peak_coord[0], 0, peak_coord[1])

            if best_snr >= threshold:
                streak = self.make_streak(snr=best_snr, transpose=transpose, threshold=threshold, peak=peak_coord,
                                          foldings=foldings, subframe=radon_partial, section=im)

        # this will always have the best S/N until "clear" is called
        self.data.best_snr = max(best_snr, self.data.best_snr)

        if streak:
            self.streaks.append(streak)
            if self.pars.use_write_cutouts:
                if self.pars.max_length_to_write is None or streak.L < self.pars.max_length_to_write:
                    streak.write_to_disk()

        # store the final FRT result (this is problematic once we start iterating over findSingle!)
        if transpose:
            self.data.radon_image_tr = radon_image
        else:
            self.data.radon_image = radon_image

        if self.pars.verbosity > 1:
            print(f'Running FRT {self.data._num_frt_calls} times, '
                  f'transpose= {transpose}, thresh= {threshold:.2f}, '
                  f'found streak: {bool(streak)}')

        return streak

    def find_multi(self, im, threshold=None, num_iter=None):
        """

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
                    new_streak.subtract_streak(im)
                    if self.pars.use_subtract_mean:
                        im -= np.nanmean(im)
                    # np.nan_to_num(M, copy=False)

                    if self.pars.use_show:
                        new_streak.plot_lines()
                        plt.show()
                        # fig = plt.gcf()
                        # fig.canvas.draw()
                        # fig.canvas.flush_events()
        return im

    def scan_thresholds(self, im, min_threshold=None):
        """

        """
        self.im_size = im.shape

        if min_threshold is None:
            min_threshold = self.pars.threshold

        mx = np.nanmax(im / np.sqrt(self.data.var_image))

        dynamic_range = np.log2(mx / min_threshold)

        thresholds = np.flip(min_threshold * 2 ** np.arange(dynamic_range + 1))
        if self.pars.verbosity > 1:
            print(f'mx= {mx:.2f} | dynamic_range= {dynamic_range:.2e} | thresholds: {np.round(thresholds, 2)}')

        for t in thresholds:
            mask = np.zeros(im.shape, dtype=bool)
            np.greater(im / np.sqrt(self.data.var_image), t / 2, where=np.isnan(im) == 0, out=mask)
            im[mask] = t * np.sqrt(self.data.var_scalar)  # clip to threshold (scaled by noise)
            if self.pars.use_subtract_mean:
                im -= np.nanmean(im)

            if self.pars.use_show:
                plt.clf()
                plt.imshow(im)
                plt.title(f'section corner: {self.data._current_section_corner}')
                plt.xlabel(f'psf_sigma= {self.data.psf_sigma:.2f} | threshold= {t:.2f}')
                plt.show()
                # f = plt.gcf()
                # f.canvas.draw()
                # f.canvas.flush_events()

            im = self.find_multi(im, t)

    def preprocess(self, M):
        """

        """
        M_new = M

        if self.use_subtract_mean: M_new = M_new - np.nanmean(M_new)  # remove the mean
        # np.nan_to_num(M, copy=False)
        # should we also remove cosmic rays/bad pixels at this level?

        M_conv = scipy.signal.convolve2d(M_new,self.psf,mode='same')

        return M_conv

    def scan_sections(self): # to be depricated!
        """

        """
        corners = []
        sections = jigsaw(self.image, self.size_sections, output_corners=corners)

        for i in range(sections.shape[0]):
            this_section = self.preprocess(sections[i])
            self.current_section_corner = corners[i]
            # treat the var map for each section right here!

            self.scanThresholds(this_section)

    # User interface #

    def input(self, image, variance=None, psf=None, filename=None, batch_num=None):
        """
        Input an image and search for streaks in it.
        Inputs: -images (expect numpy array, can be 3D)
                -variance (can be scalar or map of noise variance)
                -psf: point spread function of the image (scalar gaussian sigma or map)
                -filename: for tracking the source of discovered streaks
                -batch_num: if we are running many batches in this run
        """

        if empty(image):
            raise Exception("Cannot do streak finding without an image!")

        self.clear()

        if self.use_crop:
            image = crop2size(image, self.crop_size)

        self.im_size = imsize(image)
        self.image = image

        self.filename = filename

        # input the variance, if given!
        if not empty(variance):
            if not scalar(variance) and self.use_crop:
                self.input_var = crop2size(variance, self.crop_size)
            else:
                self.input_var = variance

        # input the PSF if given! 
        if not empty(psf):
            self.input_psf = psf

        # housekeeping 
        self.filename = filename
        self.batch_num = batch_num

        corners = []

        if self.use_sections:
            sections = jigsaw(image, self.size_sections, output_corners=corners)
        else:
            sections = image[np.newaxis,...]

        for i in range(sections.shape[0]):

            if not empty(corners):
                self.current_section_corner = corners[i]

            sec = sections[i,:,:]
            # (m,v) = image_stats(sec)
            sec = self.preprocess(sec)
            self.scanThresholds(sec)

        if self.use_show:
            plt.clf()
            h = plt.imshow(self.image)
            h.set_clim(0,5*np.sqrt(self.var_scalar))
            [streak.plotLines(im_type='full') for streak in self.streaks]
            plt.title("full frame image")
            plt.xlabel(self.filename)
            f = plt.gcf()
            f.canvas.draw()
            f.canvas.flush_events()


if __name__ == "__main__":
    f = Finder()
    f.find_single(np.random.normal(0, 1, (512, 512)))
