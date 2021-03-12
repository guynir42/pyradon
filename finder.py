# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 15:43:28 2017

@author: guyn
"""

from streak import Streak
from frt import FRT
from utils import empty, scalar, compare_size, imsize, crop2size, gaussian2D, fit_gaussian, jigsaw, image_stats
import matplotlib.pyplot as plt
import scipy.signal
import numpy as np
import math
import sys
import time


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

    def __init__(self):

        ################# switches #################

        # image pre-processing
        self.use_subtract_mean = 1  # subtract any residual bias
        self.use_conv = 1  # match filter with PSF
        self.use_crop = 0  # do we want to crop the input images (e.g., to power of 2)
        self.crop_size = 2048  # does not grow the array

        self.use_sections = 0  # cut the incoming images into sections to run faster
        self.size_sections = 1024  # can be a scalar or a 2-tuple
        self.current_section_corner = (0, 0)  # keep track of where the current section starts

        # search options
        self.use_short = 1  # search for short streaks
        self.min_length = 32  # minimal length (along the axis) for short streaks
        self.threshold = 5  # in units of S/N
        # how many times to go over the same image to look for streaks (per threshold/per section)
        self.num_iterations = 5
        self.use_exclude = 1
        self.exclude_x_pix = (-50, 50)
        self.exclude_y_pix = None

        self.use_show = 1
        self.use_save_images = 1

        self.use_write_cutouts = 0
        self.max_length_to_write = 128

        self.debug_bit = 1
        self._version = 2.01

        ################## inputs/outputs ######################

        self.image = None  # image as given to finder
        self.im_size = None  # two-element tuple
        self.radon_image = None  # final FRT result (normalized by the var-map)
        self.radon_image_trans = None  # final FRT of the transposed image (normalized)
        # list of all SNR values found in this run (use "reset()" to clear them)
        self.snr_values = []
        # self._im_size_tr = []  # size of image after transposition (if not transposed, equal to im_size) (do we need this??)
        self.best_SNR = 0

        # objects
        self.streaks = []  # streaks saved from latest call to input()
        # a list of Streak objects that passed the threshold,  saved from all scans (use reset() to remove these)
        self.streaks_all = []

        # housekeeping variables
        self.filename = ''  # which file we are currently scanning
        self.batch_num = 0  # which batch in the run
        self.frame_num = 0  # which frame in the batch
        self.section_num = 0  # which section in the current image

        ################ PSF width and image ################

        # original PSF as given (PSF image or scalar width parameter "sigma")
        self._input_psf = None
        self._psf = None  # the map of the PSF, either as given or generated as a Gaussian from "sigma_psf"
        self._sigma_psf = None  # the width of a Gaussian PSF, either given as scalar or fit to PSF map
        self._default_sigma_psf = 1  # if no PSF is given, assume this as the width of a Gaussian PSF

        ################## variance maps and scalars ####################

        self._input_var = None  # input variance (can be scalar or 2D array)
        self._var_scalar = None  # either given as scalar or the median of the var map
        self._var_map = None  # either given as map or just expanded from the scalar
        self._size_var = None  # the size of the input variance map
        self._expanded_var = None  # did we expand the input variance map
        self._default_var_scalar = 1  # if no variance is given, this is used as the variance scalar
        self._radon_var_map = []  # list of partial Radon variances from either a var_scalar or var_map
        self._radon_var_map_trans = []  # same thing, for the transposed FRT

        ####################### Other hidden values #####################

        self.num_frt_calls = 0

    @property
    def input_var(self):
        if empty(self._input_var):
            return self._default_var_scalar
        else:
            return self._input_var

    @input_var.setter
    def input_var(self, val):
        # if working with scalar variance, only need to update the scaling of var maps
        if scalar(val) and scalar(self._input_var):
            if not empty(self._radon_var_map):
                self._radon_var_map = [m * val / self.input_var for m in self._radon_var_map]
            if not empty(self._radon_var_map_trans):
                self._radon_var_map_trans = [
                    m * val / self.input_var for m in self._radon_var_map_trans]

        else:  # new or old input_var is not scalar, must recalculate var maps
            self.clearVarMap()

        self._input_var = val
        if scalar(val):
            self._var_scalar = val
            self._var_map = None
        else:
            self._var_scalar = np.median(val)
            self._var_map = val

    @property
    def var_scalar(self):
        if empty(self._var_scalar):
            return self._default_var_scalar
        else:
            return self._var_scalar

    @property
    def var_map(self):
        if empty(self._var_map) and not empty(self.im_size):
            return self.var_scalar*np.ones(self.im_size)
        else:
            return self._var_map

    def __get_input_psf(self):

        return self._input_psf

    def __set_input_psf(self, val):

        if scalar(val):
            if empty(self.input_psf) or not scalar(self.input_psf) or val != self.input_psf:
                self._psf = gaussian2D(val)
                self._sigma_psf = val

        elif isinstance(val, np.ndarray) and val.ndim == 2:
            if scalar(self.input_psf) or val.shape != self._input_psf.shape or np.any(val != self.input_psf):
                self._psf = val
                a = fit_gaussian(val)  # run a short minimization 2D fitter to gaussian
                self._sigma_psf = (a.x[1] + a.x[2]) / 2  # average the x and y sigma

        else:
            raise ValueError("input_psf must be a scalar or a numpy array")

        self._input_psf = val
        self._psf = self._psf / np.sqrt(np.sum(self._psf ** 2))  # normalized PSF

    # just trying out different methods for 'property'
    input_psf = property(__get_input_psf, __set_input_psf)

    @property
    def psf(self):
        if empty(self._input_psf):
            # go through the setter for input_psf and calculate psf and sigma_psf
            self.input_psf = self._default_sigma_psf

        return self._psf

    @psf.setter
    def psf(self, val):
        self.input_psf = val

    @property
    def sigma_psf(self):
        if empty(self._input_psf):
            # go through the setter for input_psf and calculate psf and sigma_psf
            self.input_psf = self._default_sigma_psf

        return self._sigma_psf

    @sigma_psf.setter
    def sigma_psf(self, val):
        self.input_psf = val

    ################# reset methods ###################

    def reset(self):  # do this at the start of a new run

        self.snr_values = []
        self.streaks_all = []
        self.total_runtime = 0

        self.clearVarMap()
        self.clearPSF()

        self.clear()

    def clear(self):  # do this each time we have new images

        self.image = None
        self.radon_image = None
        self.radon_image_trans = None
        self.best_SNR = []
        self.streaks = []

        self.batch_num = None
        self.frame_num = None
        self.section_num = None

        self.current_section_corner = (0, 0)

        self.num_frt_calls = 0

    def clearVarMap(self):  # when we switch to a new image frame

        self._input_var = []  # input variance (can be scalar or 2D array)
        self._var_scalar = []  # either given as scalar or the median of the var map
        self._var_map = []  # either given as map or just expanded from the scalar
        self._size_var = []  # the size of the input variance map
        self._expanded_var = []  # did we expand the input variance map
        self._default_var_scalar = 1  # if no variance is given, this is used as the variance scalar
        self._radon_var_map = []  # list of partial Radon variances from either a var_scalar or var_map
        self._radon_var_map_trans = []  # same thing, for transposed FRT

    def clearPSF(self):

        self._input_psf = []
        self._psf = []
        self._sigma_psf = []

    ################ getters #####################

    def useExpand(self):
        return not self.use_short

    def getRadonVariance(self, transpose=0):
        """ Get the partial Radon transforms of the background noise for some transpose.
            Lazy Reloading: only delete old var-maps if input size changed (or 
            if we changed expansion mode) and then calculate the var-maps on demand.

        """

        # check if we need to recalculate the var map
        if not empty(self._size_var) and (
                not compare_size(self.im_size, self._size_var) or self.useExpand() != self._expanded_var):
            if self.debug_bit:
                print("Clearing the Radon var-maps")
            self._radon_var_map = []  # clear this to be lazy loaded with the right size
            self._radon_var_map_trans = []

        # if there is no var map, we need to lazy load it
        if empty(self._radon_var_map):

            # do we have a variance map or scalar??
            # if empty(self._input_var):
            #     self._input_var = self._default_var_scalar
            #
            # if scalar(self._input_var):
            #     self._var_scalar = self._input_var
            #     self._var_map = self._input_var * np.ones(self.im_size, dtype='float32')
            # else:
            #     self._var_scalar = np.median(self.input_var)
            #     self._var_map = self._input_var

            self._size_var = self.im_size
            self._expanded_var = self.useExpand()
            self._radon_var_map = FRT(self.var_map, partial=True,
                                      expand=self._expanded_var, transpose=False)
            self._radon_var_map_trans = FRT(
                self.var_map, partial=True, expand=self._expanded_var, transpose=True)

        if transpose:
            return self._radon_var_map_trans
        else:
            return self._radon_var_map

    def getNormFactorPSF(self):

        return np.sum(self.psf) * np.sqrt(np.sum(self.psf ** 2))

    def getGeometricFactor(self, foldings):

        height = 2 ** (foldings - 1)
        th = np.arctan(np.arange(-height + 1, height) / np.float(height))
        G = np.maximum(np.fabs(np.cos(th)), np.fabs(np.sin(th)))
        G = G[:, np.newaxis, np.newaxis]
        return G

    ########################## STREAK FINDING #####################################

    def makeStreak(self, snr, transpose, threshold, peak, foldings, subframe, section):
        s = Streak(snr=snr, transpose=transpose, threshold=threshold, peak=peak,
                   foldings=foldings, subframe=subframe, section=section)

        s.update_from_finder(self)
        s.calculate()

        return s

    def findSingle(self, M, transpose=False, threshold=None):

        if empty(M):
            return None

        self.im_size = imsize(M)

        if empty(threshold):
            threshold = self.threshold

        streak = None

        self.num_frt_calls += 1

        if self.use_short:

            # these are raw Radon partial transforms
            R_partial = FRT(M, transpose=transpose, partial=True, expand=False)

            # divide by the variance map, geometric factor, and PSF norm for each level
            V = self.getRadonVariance(transpose)
            # m counts the number of foldings, partials start at 2
            G = [self.getGeometricFactor(m) for m in range(2, len(R_partial) + 2)]
            P = self.getNormFactorPSF()

            R_partial = [R_partial[i] / np.sqrt(V[i] * G[i] * P) for i in range(len(R_partial))]

            R = R_partial[-1][:, 0, :]  # get the final Radon image as 2D map

            snrs_idx = [np.nanargmax(r) for r in R_partial]  # best index for each folding
            # snrs_max = np.array([r[i] for i,r in zip(snrs_idx,R_partial)]) # best SNR for each folding
            snrs_max = np.array([np.nanmax(r) for r in R_partial])  # best SNR for each folding

            best_idx = np.nanargmax(snrs_max)  # which folding has the best SNR
            best_snr = snrs_max[best_idx]  # what is the best SNR of all foldings

            if best_snr >= threshold and 2**best_idx >= self.min_length:
                # the x,y,z of the peak in that subframe
                peak_coord = np.unravel_index(snrs_idx[best_idx], R_partial[best_idx].shape)

                streak = self.makeStreak(snr=best_snr, transpose=transpose, threshold=threshold, peak=peak_coord,
                                         foldings=best_idx + 2, subframe=R_partial[best_idx], section=M)

        else:

            R = FRT(M, transpose=transpose, partial=False, expand=True)

            V = self.getRadonVariance(transpose)
            foldings = len(V) + 1  # the length tells you how many foldings we need

            V = V[-1][:, 0, :]  # get the last folding and flatten it to 2D
            G = self.getGeometricFactor(foldings)
            P = self.getNormFactorPSF

            R = R / np.sqrt(V * G * P)

            # this is how it would look from a partial transpose output
            R_partial = R[:, np.newaxis, :]

            idx = np.argmax(R)
            best_snr = R[idx]

            peak_coord = np.unravel_index(idx, R.shape)
            # added zero for y start position that is often non-zero in the partial transforms
            peak_coord = (peak_coord[0], 0, peak_coord[1])

            streak = self.makeStreak(snr=best_snr, transpose=transpose, threshold=threshold, peak=peak_coord,
                                     foldings=foldings, subframe=R_partial, section=M)

        # this will always have the best S/N until "clear" is called
        self.best_SNR = max(best_snr, self.best_SNR)

        if not empty(streak):
            self.streaks.append(streak)
            if self.use_write_cutouts:
                if empty(self.max_length_to_write) or streak.L < self.max_length_to_write:
                    streak.write_to_disk()

        # store the final FRT result (this is problematic once we start iterating over findSingle!)
        if transpose:
            self.radon_image_trans = R
        else:
            self.radon_image = R

        if self.debug_bit > 1:
            print("Running FRT %d times, trans= %d, thresh= %f, found streak: %d"
                  % (self.num_frt_calls, transpose, threshold, not empty(streak)))

        return streak

    def findMulti(self, M, threshold=None, num_iter=None):

        if threshold is None:
            threshold = self.threshold

        if num_iter is None:
            num_iter = self.num_iterations

        for trans in range(2):
            for i in range(num_iter):
                new_streak = self.findSingle(M, transpose=trans, threshold=threshold)
                if empty(new_streak):
                    break
                else:
                    new_streak.subtractStreak(M)
                    if self.use_subtract_mean:
                        M -= np.nanmean(M)
                    # np.nan_to_num(M, copy=False)

                    if self.use_show:
                        new_streak.plotLines()
                        f = plt.gcf()
                        f.canvas.draw()
                        f.canvas.flush_events()
        return M

    def scanThresholds(self, M):

        self.im_size = imsize(M)

        mx = np.nanmax(M/np.sqrt(self.var_map))

        N = math.log2(mx/self.threshold)

        thresholds = np.flip(self.threshold * 2**np.arange(N+1))
        if self.debug_bit > 1:
            print("mx= %f | N= %f | thresholds: %s" % (mx, N, str(thresholds)))

        for t in thresholds:
            # M[M>t] = t
            mask = np.zeros(M.shape, dtype=bool)
            np.greater(M/np.sqrt(self.var_map), t/2, where=~np.isnan(M), out=mask)
            M[mask] = t*np.sqrt(self.var_scalar)
            if self.use_subtract_mean:
                M -= np.nanmean(M)
            # np.nan_to_num(M, copy=False)

            if self.use_show:
                plt.clf()
                plt.imshow(M)
                plt.title("section corner: %s" % (str(self.current_section_corner)))
                plt.xlabel("psf_sigma= %f | threshold= %f" % (self.sigma_psf, t))
                f = plt.gcf()
                f.canvas.draw()
                f.canvas.flush_events()

            M = self.findMulti(M, t)

    def preprocess(self, M):

        M_new = M

        if self.use_subtract_mean:
            M_new = M_new - np.nanmean(M_new)  # remove the mean
        # np.nan_to_num(M, copy=False)
        # should we also remove cosmic rays/bad pixels at this level?

        M_conv = scipy.signal.convolve2d(M_new, self.psf, mode='same')

        return M_conv

    def scanSections(self):  # to be depricated!

        corners = []
        sections = jigsaw(self.image, self.size_sections, output_corners=corners)

        for i in range(sections.shape[0]):
            this_section = self.preprocess(sections[i])
            self.current_section_corner = corners[i]
            # treat the var map for each section right here!

            self.scanThresholds(this_section)

    ########################## INPUT METHOD #######################################

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
            sections = image[np.newaxis, ...]

        for i in range(sections.shape[0]):

            if not empty(corners):
                self.current_section_corner = corners[i]

            sec = sections[i, :, :]
            # (m,v) = image_stats(sec)
            sec = self.preprocess(sec)
            self.scanThresholds(sec)

        if self.use_show:
            plt.clf()
            h = plt.imshow(self.image)
            h.set_clim(0, 5*np.sqrt(self.var_scalar))
            [streak.plotLines(im_type='full') for streak in self.streaks]
            plt.title("full frame image")
            plt.xlabel(self.filename)
            f = plt.gcf()
            f.canvas.draw()
            f.canvas.flush_events()


if __name__ == "__main__":
    f = Finder()
    f.findSingle(np.random.normal(0, 1, (512, 512)))
