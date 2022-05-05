# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 16:11:12 2017

@author: guyn
"""
import os
import sys
import math
import os
import re
import h5py
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import empty, imsize, model


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
    
     The Streak object can also be used to subtract the streak from the input image. 
     
    """
    
    def __init__(self, snr=None, transpose=False, threshold=None, peak=None, foldings=None, subframe=None, section=None):

        self.image = None  # subtracted image, as given to finder. updated directly from finder
        self.radon_image = None  # full Radon image for the correct transposition. updated directly from finder
        self.subframe = subframe  # subframe where streak is detected. For non-short streaks, equal to "radon_image".
        self.image_section_raw = None  # smaller region of the original image, or just equal to original image (if no sectioning is done)
        self.image_section_proc = section  # same as section, only PSF filtered, and streak subtracted (all processing is applied)
        self.corner_section = (0,0)  # the corner of the section inside the full image
        self.size_section = None  # size of section (2-tuple)
        self.size_section_tr = None  # size of section after transpose (if not transposed, equal to size_section)
        self.image_cutout = None  # small region around streak
        self.cut_size = 128 
        # self.image_cutout_proc = None  # small region around streak with full processing applied
        self.corner_cutout = None  # the corner of the cutout inside the full image
        self.size_cutout = None  # size of the cutout (2-tuple)
        self.psf = None  # the PSF image itself, normalized, that was used to do the filter. updated directly from finder
        
        # these help find the correct image where the streak exists
        self.frame_num = None  # which frame in the batch. updated directly from finder
        self.batch_num = None  # which batch in the run. updated directly from finder
        self.filename = ''  # which file it came from. updated directly from finder        

        self.threshold = threshold  # which threshold was used. updated directly from finder
        self.is_short = None  # if yes, subframe is smaller than the full Radon image. updated in "calculate()"
        
        # how bright the streak was
        self.I = None # intensity, or brightness per unit length. found in "calculate()"
        self.snr = snr # calculated from subframe and index. found in "calculate()"
        self.snr_fwhm = None  # S/N per resolution element
        self.count = None  # how many photons across the whole streak.
        
        # calculated parameters
        self.L = None  # length of streak (pixels). from "calculate()"
        self.th = None  # angle of streak (th=0 is on the y axis). from "calculate()"
        self.x0 = None  # intercept of line with the x axis. from "calculate()"
        self.a = None  # slope parameter (y=ax+b). from "calculate()"
        self.b = None  # intercept parameter (y=ax+b). from "calculate()"
        self.x1 = None  # streak starting point in x. from "calculate()"
        self.x2 = None  # streak end point in x. from "calculate()"
        self.y1 = None  # streak starting point in y. from "calculate()"
        self.y2 = None  # streak end point in y. from "calculate()"
        self.dx = None  # difference of x's
        self.dy = None  # difference of y's

        # raw coordinates from the Radon result / subframe
        self.transposed = transpose  # was the image transposed? from function argument
        self.foldings = foldings  # in what step in the FRT the streak was found. from function argument
        self.radon_max_idx = peak  # 3D index of the maximum of the subframe (streak position). from function argument
        self.radon_x0 = None  # position coordinate in the Radon image. using subframe and index
        self.radon_dx = None  # slope coordinate in the Radon image. using subframe and index
        self.radon_x_var = None  # error estimate on "radon_x"
        self.radon_dx_var = None  # error estimate on radon_dx"
        self.radon_xdx_cov = None  # cross correlation of the slope-position errors
        self.radon_y1 = None  # start of the subframe
        self.radon_y2 = None  # end of the subframe
        
        # switches and parameters for user
        self.noise_var = None  # updated directly from finder
        self.psf_sigma = None  # updated directly from finder
        self.subtract_psf_widths = 3  # how many PSF widths to remove around streak position (overriden by finder)
        
        # internal switches and parameters
        self.was_expanded = False  # check if original image was expanded before FRT. updated manually from finder
        self.was_convolved = False  # check if original image was convolved with PSF. updated manually from finder
        self.num_psfs_peak_region = 5  # rough estimate of the region around the Radon peak we want to cut for error estimates
        self.num_snr_peak_region = 2  # how many S/N units below maximum is still inside the peak region
        self.peak_region = None  # a map of the peak region, with only the part above the cut (peak-num_snr_peak_region) not zeroed out (used for error estimates)

        self._version = 1.03

    @property
    def radon_dy(self):
        return self.radon_y2 - self.radon_y1

    @property
    def x1f(self): return self.x1 + self.corner_section[1]

    @property
    def x2f(self): return self.x2 + self.corner_section[1]

    @property
    def y1f(self): return self.y1 + self.corner_section[0]

    @property
    def y2f(self): return self.y2 + self.corner_section[0]

    @property
    def x1c(self):
        if empty(self.corner_cutout):
            return None
        else:
            return self.x1 + self.corner_section[1] - self.corner_cutout[1]

    @property
    def x2c(self):
        if empty(self.corner_cutout):
            return None
        else:
            return self.x2 + self.corner_section[1] - self.corner_cutout[1]

    @property
    def y1c(self):
        if empty(self.corner_cutout):
            return None
        else:
            return self.y1 + self.corner_section[0] - self.corner_cutout[0]

    @property
    def y2c(self):
        if empty(self.corner_cutout):
            return None
        else:
            return self.y2 + self.corner_section[0] - self.corner_cutout[0]


    @property
    def mid_x(self):
        if empty(self.x1) or empty(self.x2):
            return None
        else:
            return (self.x1+self.x2)/2
    @property
    def mid_y(self):
        if empty(self.y1) or empty(self.y2):
            return None
        else:
            return (self.y1+self.y2)/2
    @property
    def mid_x_full(self): # this gives the middle x position from the full image
        if empty(self.x1f) or empty(self.x2f):
            return None
        else:
            return (self.x1f+self.x2f)/2
    @property
    def mid_y_full(self): # this gives the middle y position from the full image
        if empty(self.y1f) or empty(self.y2f):
            return None
        else:
            return (self.y1f+self.y2f)/2

    def update_from_finder(self, finder):
        if empty(finder): return

        for att in self.__dict__.keys():  # load all attributes in "self" that exist in "finder"
            if hasattr(finder, att) and not callable(getattr(finder,att)):
                setattr(self, att, getattr(finder, att))

        self.noise_var = finder.var_scalar
        self.psf_sigma = finder.sigma_psf

        self.corner_section = finder.current_section_corner

        self.was_convolved = bool(finder.use_conv)
        self.was_expanded = bool(finder.useExpand())
        
    def calculate(self):
        # these are the raw results from this subframe
        self.size_section = self.image_section_proc.shape
        if self.transposed:
            self.size_section_tr = tuple(reversed(self.size_section))
        else:
            self.size_section_tr = self.size_section

        if not empty(self.image):  # part of the orignal image that is defined as the cutout (without processing!)
            self.image_section_raw = self.image[self.corner_section[0]:self.size_section[0], self.corner_section[1]:self.size_section[1]]

        self.radon_y1 = (2**(self.foldings-1)) * self.radon_max_idx[1]  # size of each slice in y, times the slice number (index)
        self.radon_y2 = (2**(self.foldings-1)) * (self.radon_max_idx[1]+1)  # same thing, top index (should we include the last pixel?)
        offset = (self.subframe.shape[2] - self.size_section_tr[1])//2  # this is added on either side if was_expanded
        self.radon_x0 = self.radon_max_idx[2] - offset  # position of x0 in the subframe (removing the expanded pixels)
        self.radon_dx = self.radon_max_idx[0] - self.subframe.shape[0]//2  # the angle is in dim0 needs to offset for negative angles
        
        # signal to noise from the subframe maximum
        self.snr = self.subframe[self.radon_max_idx]
        self.is_short = self.subframe.ndim>2 and self.subframe.shape[1]>1
        
        # assume there is no transpose (then add it if needed)
        self.y1 = self.radon_y1
        self.y2 = self.radon_y2
        self.dy = self.y2-self.y1
        self.dx = self.radon_dx
        if self.radon_dx!=0:
            self.a = self.dy/self.dx
            self.x0 = self.radon_x0 - self.y1/self.a
            self.b = -self.a*self.x0
            self.th = math.degrees(math.atan(self.a))
            self.x1 = (self.y1-self.b)/self.a
            self.x2 = (self.y2-self.b)/self.a
        else:
            self.a = float('NaN')
            self.x0 = self.radon_x0
            self.b = float('NaN')
            self.th = 90
            self.x1 = self.radon_x0
            self.x2 = self.radon_x0
        
        self.L = abs(self.radon_dy/math.sin(math.radians(self.th)))
#        f = math.fabs(math.sin(math.radians(self.th)))
        self.I = self.snr*math.sqrt(self.noise_var*2*math.sqrt(math.pi)*self.psf_sigma/(self.L))
        self.snr_fwhm = self.I*0.81/math.sqrt(self.noise_var)
        
        if self.transposed:
            self.x1, self.y1 = self.y1, self.x1
            self.x2, self.y2 = self.y2, self.x2
            self.a = 1/self.a
            self.b, self.x0 = self.x0, self.b
            self.th = 90 - self.th
            
        # self.x1 = round(self.x1)
        # self.x2 = round(self.x2)
        # self.y1 = round(self.y1)
        # self.y2 = round(self.y2)
        # self.x0 = round(self.x0)

        ############ calculate the errors on the Radon parameters #############

        R = np.array(self.subframe[:,self.radon_max_idx[1],:]) # present this slice as 2D image
        xmax = self.radon_max_idx[2]
        ymax = self.radon_max_idx[0]
        
        S = round(self.psf_sigma*self.num_psfs_peak_region) # size of area of a few PSF widths around the peak
        x1 = xmax - S
        if x1<0: x1 = 0
        x2 = xmax + S
        if x2>=R.shape[1]: x2 = R.shape[1]-1
        y1 = ymax - S
        if y1<0: y1 = 0
        y2 = ymax + S
        if y2>=R.shape[0]: y2 = R.shape[0]-1
        x1 = int(x1)
        x2 = int(x2)
        y1 = int(y1)
        y2 = int(y2)
        C = np.array(R[y1:y2,x1:x2]) # make a copy of the array
        xgrid,ygrid = np.meshgrid(range(C.shape[1]),range(C.shape[0]))

        idx = np.unravel_index(np.nanargmax(C), C.shape)
        mx = C[idx]
        
        C[C<mx-self.num_snr_peak_region] = 0
        
        xgrid = xgrid - idx[1]
        ygrid = ygrid - idx[0]
        
        self.radon_x_var = np.sum(C*xgrid**2)/np.sum(C)
        self.radon_dx_var = np.sum(C*ygrid**2)/np.sum(C) 
        self.radon_xdx_cov = np.sum(C*xgrid*ygrid)/np.sum(C)
        
        self.peak_region = C

        self.makeCutout()

    def makeCutout(self, size=None):
        
        if empty(self.image):
            return; 
        
        if empty(size):
            size = self.cut_size;
        
        d1 = int(np.floor(size/2))
        d2 = int(np.ceil(size/2))
        
        x1 = int(round(self.mid_x_full))-d1
        x2 = int(round(self.mid_x_full))+d2
        y1 = int(round(self.mid_y_full))-d1
        y2 = int(round(self.mid_y_full))+d2
        
        self.image_cutout = self.image[y1:y2,x1:x2]
        
        self.size_cutout = (size, size)
        self.corner_cutout = (y1, x1)
        

    def write_to_disk(self, filename=None):
        if empty(filename):
            if empty(self.filename):
                return 
            else: 
                filename = os.path.splitext(self.filename)[0]+'.h5'
        
        with h5py.File(filename, 'a') as hf:
            numbers=[int(re.search(r'\d+$', k).group()) for k in hf.keys()]
            
            if empty(numbers):
                new_number = 1
            else:
                new_number = max(numbers) + 1
            
            ds = hf.create_dataset('streak_%03d' % (new_number), data=self.image_cutout)
            
            ds.attrs['corner'] = self.corner_cutout
            # add additional metadata later on
        
    def subtractStreak(self, M, replace_value=float('NaN')):

        mask = model(imsize(M), self.x1, self.x2, self.y1, self.y2, self.psf_sigma) > 0

        M[mask] = replace_value

        # # these are really rough estimates. Can improve this by looking at the error ellipse and subtracting y and dy values inside that range only
        # shift_array = np.arange(-self.psf_sigma*width, self.psf_sigma*width+1)
        #
        # M_sub = np.array(M_in)
        #
        # for shift in shift_array:
        #     if self.transposed:
        #         x1 = self.x1
        #         x2 = self.x2
        #         y1 = self.y1 + shift
        #         y2 = self.y2 + shift
        #     else:
        #         x1 = self.x1 + shift
        #         x2 = self.x2 + shift
        #         y1 = self.y1
        #         y2 = self.y2
        #
        # (xlist, ylist, n) = listPixels(x1,x2,y1,y2, M_in.shape)
        # if not empty(xlist):
        #     M_sub[xlist[0], ylist[0]] = 0
        #
        # return M_sub

    def show(self, ax=None):

        if ax is None:
            ax = plt.gca()

        plt.imshow(self.image_section_proc)
        self.plotLines(ax)

    def plotLines(self, ax=None, offset=10, im_type='section', line_format='--m', linewidth=2):

        if ax is None:
            ax = plt.gca()

        if im_type=='section':
            x1 = self.x1
            x2 = self.x2
            y1 = self.y1
            y2 = self.y2
        elif im_type=='full':
            x1 = self.x1f
            x2 = self.x2f
            y1 = self.y1f
            y2 = self.y2f
        elif im_type=='cutout':
            x1 = self.x1c
            x2 = self.x2c
            y1 = self.y1c
            y2 = self.y2c
        else:
            raise RuntimeError('Unknown im_type: %s. Use "section" or "full" or "cutout".' % ('section'))

        if self.transposed:
            ax.plot([x1, x2], [y1 + offset, y2 + offset], line_format, linewidth=linewidth)
            ax.plot([x1, x2], [y1 - offset, y2 - offset], line_format, linewidth=linewidth)
        else:
            ax.plot([x1 + offset, x2 + offset], [y1, y2], line_format, linewidth=linewidth)
            ax.plot([x1 - offset, x2 - offset], [y1, y2], line_format, linewidth=linewidth)

