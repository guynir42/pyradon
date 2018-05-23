# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 15:43:28 2017

@author: guyn
"""

from pyradon.streak import Streak
import pyradon.frt
from utils import empty, scalar, compare_size, imsize, crop2size, gaussian2D, fit_gaussian

import matplotlib.pyplot as plt
import scipy.signal
import numpy as np
import math
import sys
import time

class Finder: 
    """
    Streak finding tool. 
    This object can be used in two ways:
    (1) Give it some images, using self.input(images, ...)
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

    (2) Give the finder object as an optional argument to the "frt"
        function, so it does its magic while the FRT is running. 
        In this case you should make sure all parameters of the finder are
        set correctly before starting the FRT on the image. 
        
  SWITCHES AND SETTINGS (see comments in properties block for more details)
    -Image pre-processing: use_subtract_mean, use_conv, use_crop_image, crop_size.
    -Search options: use_short, min_length, use_recursive, recursion_depth. 
    -Post processing: use_exclude, exclude_dx, exclude_dy
    -memory considerations: use_save_images, use_clear_memory.
    -housekeeping: filename, batch_num, frame_num.
    -display defaults: show_bit, display_index, display_which, line_offset,
     rect_size, display_monochrome.
 
  NOTES: -the Radon image that is saved in each streak is already normalized 
          by the variance map. 
         -If you cut the stars out, replace them with NaNs, then subtract 
          the mean of the image, then replace the NaNs with zeros.
 
    """
    
    def __init__(self):
        
        ################# switches #################
        
        # image pre-processing
        self.use_subtract_mean = 1 # subtract any residual bias
        self.use_conv = 1 # match filter with PSF
        self.use_crop = 0 # do we want to crop the input images (e.g., to power of 2)
        self.crop_size = 2048 # does not grow the array
        
        # search options
        self.use_short = 1 # search for short streaks
        self.min_length = 32 # minimal length (along the axis) for short streaks
        self.threshold = 10 # in units of S/N
        self.use_recursive = 0 # find multiple streaks by recursively calling FRT and subtracting the found streak
        self.recursion_depth = 10 # maximum number of recursions for each transposition       
        self.use_only_one = 1 # if for some reason you find a streak in either transpositions, keep only the best (ignored when use_recursion=1)
        self.subtract_psf_widths = 3 # the width to subtract around the found position (when subtracting streaks). Units of PSF sigma
        
        self.use_exclude = 1
        self.exclude_x_pix = [-50, 50]
        self.exclude_y_pix = []
        
        self.use_save_images = 1
        
        self.debug_bit = 1        
        self._version = 1.00

        
        ################## inputs/outputs ######################
        
        self.input_images = [] # images as given to finder
        self.im_size = [] # two-element tuple
        self.radon_image = [] # final FRT result (normalized by the var-map)
        self.radon_image_trans = [] # final FRT of the transposed image
        self.subtracted_image = []# image after PSF filter and after removing any found streaks
        self.snr_values = [] # list of all SNR values found in this run (use "reset()" to clear them)        
        self._im_size_tr = [] # size of image after transposition (if not transposed, equal to im_size)
        self.last_snr = [] # last value of the best S/N for the image scanned
        
        # objects
        self.last_streak= []
        self.streaks = [] # streaks saved from latest call to input()
        self.prev_streaks = [] # a list of Streak objects that passed the threshold,  saved from all scans (use reset() to clear these)        
        
        # housekeeping variables
        self.filename = '' # which file we are currently scanning
        self.batch_num = 0 # which batch in the run
        self.frame_num = 0 # which frame in the batch

        ################ PSF width and image ################ 
        
        self._input_psf = [] # original PSF as given (PSF image or scalar width parameter "sigma")
        self._default_psf_width = 2 # if no PSF is given, assume this as the width of a Gaussian PSF
        
        ################## variance maps and scalars ####################
                
        self._input_var = [] # input variance (can be scalar or 2D array)
        self._var_size = [] # the size of the input variance map
        self._var_was_expanded = [] # did we expand the input variance map
        self._default_var_value = 1 # if no variance is given, this is used as the variance scalar
        self._radon_var_uni = [] # list of partial Radon variances of a scalar        
        self._radon_var_map = [] # list of partial Radon variances from a given var-map

    
    @property
    def noise_var(self):
        if empty(self._input_var):
            return self._default_var_value
        elif scalar(self._input_var):
            return self._input_var
        else:
            return np.median(self._input_var)
    
    @property
    def radon_var_uni(self): # lazy load this list 
        if empty(self.im_size): raise Exception("You are asking for a uniform var-map without giving an image size!")
        
        if empty(self._radon_var_uni) and (empty(self._input_var) or scalar(self._input_var)):
            self._var_size = self.im_size
            self._var_was_expanded = self.useExpand()
            self._radon_var_uni.append(pyradon.frt.frt(np.ones(self.im_size), expand=self.useExpand(), partial=True))

            if self.im_size[0]==self.im_size[1]:
                self._radon_var_uni.append(self._radon_var_uni[0]) # image is square, just make a copy of the var-map 
            else:
                print "making transposed uni var map"
                self._radon_var_uni.append(pyradon.frt.frt(np.ones(self.im_size), expand=self.useExpand(), partial=True, transpose=True))
            
            
        return self._radon_var_uni
    
    @property
    def radon_var_map(self): # lazy load this list
        if empty(self._radon_var_map) and not empty(self._input_var) and not scalar(self._input_var): # no var-map was calculated, but also make sure _input_var is given as a matrix
            self._var_size = imsize(self._input_var)
            self._var_was_expanded = self.useExpand()
            self._radon_var_map.append(pyradon.frt.frt(self._input_var, expand=self.useExpand(), partial=True, transpose=False))
            self._radon_var_map.append(pyradon.frt.frt(self._input_var, expand=self.useExpand(), partial=True, transpose=True))
        
        return self._radon_var_map

    @property
    def psf_sigma(self):
        if empty(self._input_psf):
            return self._default_psf_width
        elif scalar(self._input_psf):
            return self._input_psf
        else:
            a = fit_gaussian(self._input_psf)
            return (a.x[1]+a.x[2])/2 # average the x and y sigma
    
    @property
    def psf(self):
        if empty(self._input_psf):
            return np.empty
        elif scalar(self._input_psf):
            p = gaussian2D(self._input_psf)
            return p/np.sqrt(np.sum(p**2)) # normalized PSF
            
        else:
            p = self._input_psf
            return p/np.sqrt(np.sum(p**2)) # normalized PSF
    
    ################# reset methods ###################
    
    def reset(self):
        
        self.snr_values = []        
        self.prev_streaks = []
        self.total_runtime = 0
        self._input_var = []
        self._input_psf = []
        self.clear()
        
    def clear(self):
        
        self.input_images = []
        self.radon_image = []
        self.radon_image_trans = []
        self.last_streak = []        
        self.streaks = []
    
    ################ getters #####################
    
    def useExpand(self):
        return not self.use_short
    
    @staticmethod
    def psfNorm(psf):
        return np.sum(psf)*np.sqrt(np.sum(psf**2))
    
    def getRadonVariance(self, transpose=0, log_step=None):
        """ Get the correct Radon Variance map for a specific step and transposition. """
        """ If the size of _input_var has changed we will recalculate the maps 
            using the getters defined in __init__.         
        """
        
        if log_step is None: # default variance map is the fully transformed
            if transpose:
                log_step = math.ceil(math.log(self.im_size[1],2)) # transpose==horizontal axis is active
            else:
                log_step = math.ceil(math.log(self.im_size[0],2)) # no transpose==vertical axis is active
        
        if not empty(self._var_size) and (not compare_size(self.im_size, self._var_size) or self.useExpand()!=self._var_was_expanded):
            self._radon_var_uni = [] # clear this to be lazy loaded with the right size
            self._radon_var_map = [] # clear this to be lazy loaded with the right size
        
        log_step = int(log_step)        
        
        if empty(self._input_var) or scalar(self._input_var):
            return self._input_var*self.radon_var_uni[transpose][log_step-1]
        else:
            return self.radon_var_map[transpose][log_step-1]
        
########################## INPUT METHOD #######################################

    def input(self, images, variance=None, psf=None, filename=None, batch_num=None):
        """
        Input images and search for streaks in them. 
        Inputs: -images (expect numpy array, can be 3D)
                -variance (can be scalar or map of noise variance)
                -psf: for the images (2D) or for each image individuall (3D, first dim equal to number of images)
                -filename: for tracking the source of discovered streaks
                -batch_num: if we are running many batches in this run
        """

        if empty(images):
            raise Exception("Cannot do streak finding without images!")
        
        self.clear()
        
        if images.ndim==2:
            images = images[np.newaxis,:,:] # so we can loop over axis0
        
        if self.use_crop:
            images = crop2size(images, self.crop_size)
        
        self.im_size = imsize(images)
        self.input_images = images

        # input the variance, if given!
        if not empty(variance):
            if self.use_crop:
                self._input_var = crop2size(variance, self.crop_size)
            else:
                self._input_var = variance
        
        # input the PSF if given! 
        if not empty(psf):
            self._input_psf = psf
            
        # housekeeping 
        if not empty(filename): self.filename = filename
        if not empty(batch_num): self.batch_num = batch_num

        # need these to normalize each Radon image        
        V = np.transpose(self.getRadonVariance(transpose=False), (0,2,1))[:,:,0]
        VT = np.transpose(self.getRadonVariance(transpose=True), (0,2,1))[:,:,0]
        
        th = np.arctan(np.arange(-self.im_size[0]+1, self.im_size[0])/np.float(self.im_size[0]))
        G = 1.0/np.sqrt(np.maximum(np.cos(th), np.sin(th)))
        thT = np.arctan(np.arange(-self.im_size[1]+1, self.im_size[1])/np.float(self.im_size[1]))
        GT = 1.0/np.sqrt(np.maximum(np.cos(thT), np.sin(thT)))
        
        for i in range(images.shape[0]): # loop over all input images
            
            if self.debug_bit:
                sys.stdout.write("running streak detection on batch %d | frame %d " % (self.batch_num, i))
            
            image_single = images[i,:,:]
            self.frame_num = i
            self.last_snr = []
            self.streaks = [] # make a new, empty list for the transposed FRT
            self.last_streak = []
            
            if self.psf.ndim>2:
                this_psf = self.psf[i,:,:]
            else:
                this_psf = self.psf

            if self.use_subtract_mean:
                image_single = image_single - np.nanmean(image_single) # masked stars don't influence the mean
            
            image_single = np.nan_to_num(image_single) # get rid of nans (used to mask stars...). for some reason copy=False is not working!
            
            if self.use_conv and not empty(this_psf):
                image_single = scipy.signal.fftconvolve(image_single, np.rot90(this_psf, 2), mode='same')

            R = pyradon.frt.frt(image_single, expand=self.useExpand(), finder=self, transpose=False)
            temp_streaks = self.streaks
            self.streaks = [] # make a new, empty list for the transposed FRT
            self.last_streak = []

            if self.use_only_one==0 or self.use_recursive:
                RT = pyradon.frt.frt(self.subtracted_image, expand=self.useExpand(), finder=self, transpose=True)
            else:
                RT = pyradon.frt.frt(image_single, expand=self.useExpand(), finder=self, transpose=True)
            
            self.streaks.extend(temp_streaks) # combine the lists from the regular andf transposed FRT
            
            self.radon_image = R/np.sqrt(V*self.psfNorm(this_psf))*G[:, np.newaxis]
            self.radon_image_trans = RT/np.sqrt(VT*self.psfNorm(this_psf))*GT[:, np.newaxis]
            
            if self.use_only_one and not self.use_recursive:
                self.streaks = [self.best]
               
            if not empty(self.streaks):
                if self.use_save_images:
                    for s in self.streaks:
                        s.input_image = np.array(images[i,:,:])
                        if s.transposed:
                            s.radon_image = np.array(self.radon_image_trans)
                        else:
                            s.radon_image = np.array(self.radon_image)
                        if not empty(s.subframe):
                            s.subframe = np.array(s.subframe)
                else:
                    s.input_image = []
                    s.radon_image = []
                    s.subframe = []
            
            if not empty(self.streaks):
                best_snr = self.best.snr
            else: # no streaks, just take the best S/N in the final Radon images
                best_snr = self.last_snr
            
            if self.debug_bit:
                sys.stdout.write("best S/N found was %f\n" % best_snr)
            
            self.prev_streaks.extend(self.streaks)
            self.snr_values.append(best_snr) 
            
            # end the loop on images

    # end of "input"
            
    ######################## SCAN METHOD ###################################

    def scan(self, subframe, transpose):
        
        if self.debug_bit>1:
            print "scan in progress... transpose=%d subframe.shape= %dx%dx%d" % (transpose, subframe.shape[0], subframe.shape[1], subframe.shape[2])
            
        m = math.log(subframe.shape[0]+1, 2)-1
        
        if not self.use_short and 2**m<self.im_size[int(transpose)]:
            return # short circuit this function if not looking for short streaks
        
        if 2**m < self.min_length/8:
            return # don't bother looking for streaks much shorter than minimal length
        
        V = self.getRadonVariance(transpose, m)
        
        S = (subframe.shape[0]+1)/2
        th = np.arctan(np.arange(-S+1,S)/np.float(S))
        G = 1/np.sqrt(np.maximum(np.cos(th), np.sin(th)))
        G = G[:, np.newaxis, np.newaxis]
        
        SNR = subframe/np.sqrt(V*self.psfNorm(self.psf))*G
        
        SNR_final = SNR 
        
        # add exclusion here
        if self.use_exclude:
            if transpose and not empty(self.exclude_y_pix):
                offset = (subframe.shape[0]+1)/2 #  index of dx=+1, also how many pixels are for angles 0<=th<=45 in this subframe
                scale = self._im_size_tr[1]/offset # scaling factor for smaller subframes
                idx1 = offset + math.ceil(self.exclude_y_pix/scale) - 1
                idx2 = offset + math.floor(self.exclude_y_pix/scale) - 1
                SNR_final[idx1:idx2,:,:] = 0
            elif not empty(self.exclude_x_pix):
                offset = (subframe.shape[0]+1)/2
                scale = self._im_size_tr[0]/offset
                idx1 = int(offset + math.ceil(self.exclude_x_pix[0]/scale))
                idx2 = int(offset + math.floor(self.exclude_x_pix[1]/scale))
                SNR_final[idx1:idx2,:,:] = 0
            
        idx = np.unravel_index(np.nanargmax(SNR_final), SNR_final.shape)
        mx = SNR_final[idx]
        
        if self.debug_bit>1:
            print "SNR found is: "+str(mx)
        
        if empty(self.last_snr) or mx>self.last_snr:
            self.last_snr = mx
        
        if mx>self.threshold and (empty(self.last_streak) or mx>self.last_streak.snr):
            self.last_streak = Streak(finder=self, subframe=SNR, log_step=m, transposed=transpose, count=subframe[idx], index=idx)
                
    # end of "scan"
    
    def finalizeFRT(self, M_in, transpose, radon_image): # this is called at the end of "frt()" if it is given a Finder object
        self.subtracted_image = M_in # even if we don't subtract anything, this must be filled
        
        if not empty(self.last_streak) and (self.last_streak.radon_dy < self.min_length): # if we found a very short streak
            self.last_streak = []
            
        if not empty(self.last_streak):
            
            self.last_streak.input_image = M_in
            self.last_streak.radon_image = radon_image;
            self.streaks.append(self.last_streak)
            self.subtracted_image = self.streaks[-1].subtractStreak(M_in, self.subtract_psf_widths) # first, subtract the found streak
            
            if self.use_recursive and len(self.streaks)<self.recursion_depth:
                
                if self.debug_bit>9:
                    plt.imshow(self.subtracted_image)
                    plt.title("num streaks found %d | transpose: %d" % (len(self.streaks), transpose))
                    time.sleep(1)
                    
                pyradon.frt.frt(self.subtracted_image, transpose=transpose, expand=self.useExpand(), finder=self)
        
        # end of finalizeFRT
                
    ######################## UTILITIES ########################################
        
    @property # to be depricated...
    def psf_norm(self):
        return np.sum(self.input_psf)*np.sqrt(np.sum(self.input_psf**2));
    
    @property
    def best(self):
              
        if empty(self.streaks):
            return []
        else:
            snr = [s.snr for s in self.streaks]
            ind = snr.index(max(snr))
            return self.streaks[ind]
        
    def replaceZeros(array): # I am not sure I need this at all, but I keep it as reference
        array[array==0] = float('nan')

#if __name__ == "__main__":
#    f = Finder()
#    f.input(s.image_noise)