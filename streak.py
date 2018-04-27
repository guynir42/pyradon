# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 16:11:12 2017

@author: guyn
"""
import math
import numpy as np

from utils import empty, listPixels

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
    
    def __init__(self, finder=None, subframe=[], log_step=[], transposed=False, count=[], index=[]):

        self.im_size = [] # size of original image (after transpose). updated directly from finder

        self.input_image = [] # subtracted image, as given to finder. updated directly from finder
        self.radon_image = [] # full Radon image for the correct transposition. updated directly from finder
        self.subframe = subframe # subframe where streak is detected. For non-short streaks, equal to "radon_+image". from function argument
        self.psf = [] # the PSF image itself, normalized, that was used to do the filter. updated directly from finder 
        
        # these help find the correct image where the streak exists
        self.frame_num = []  # which frame in the batch. updated directly from finder
        self.batch_num = []  # which batch in the run. updated directly from finder
        self.filename = ''  # which file it came from. updated directly from finder        

        self.threshold = [] # which threshold was used. updated directly from finder
        self.is_short = [] # if yes, subframe is smaller than the full Radon image. updated in "calculate()"
        
        # how bright the treak was
        self.I = [] # intensity, or brightness per unit length. found in "calculate()"
        self.snr = [] # calculated from subframe and index. found in "calculate()"
        self.snr_fwhm = [] # S/N per resolution element        
        self.count = count # how many photons across the whole streak. from function argument
        
        # calculated parameters
        self.L = [] # length of streak (pixels). from "calculate()"
        self.th = [] # angle of streak (th=0 is on the y axis). from "calculate()"
        self.x0 = [] # intercept of line with the x axis. from "calculate()"
        self.a = [] # slope parameter (y=ax+b). from "calculate()"
        self.b = [] # intercept parameter (y=ax+b). from "calculate()"
        self.x1 = [] # streak starting point in x. from "calculate()"
        self.x2 = [] # streak end point in x. from "calculate()"
        self.y1 = [] # streak starting point in y. from "calculate()"
        self.y2 = [] # streak end point in y. from "calculate()"
        
        # raw coordinates from the Radon result / subframe
        self.transposed = transposed # was the image transposed? from function argument
        self.radon_step = log_step # in what step in the FRT the streak was found. from function argument
        self.radon_max_idx = index # 3D index of the maximum of the subframe (streak position). from function argument
        self.radon_x0 = [] # position coordinate in the Radon image. using subframe and index
        self.radon_dx = [] # slope coordinate in the Radon image. using subframe and index
        self.radon_x_var = [] # error estimate on "radon_x"
        self.radon_dx_var = [] # error estimate on radon_dx"
        self.radon_xdx_cov = [] # cross correlation of the slope-position errors        
        self.radon_y1 = [] # start of the subframe 
        self.radon_y1 = [] # end of the subframe
        
        # switches and parameters for user
        self.noise_var = []  # updated directly from finder
        self.psf_sigma = [] # updated directly from finder
        self.subtract_psf_widths = 3 # how many PSF widths to remove around streak position (overriden by finder)
        
        # internal switches and parameters
        self.was_expanded = False  # check if original image was expanded before FRT. updated manually from finder
        self.was_convolved = False # check if original image was convolved with PSF. updated manually from finder
        self.num_psfs_peak_region = 5 # rough estimate of the region around the Radon peak we want to cut for error estimates
        self.num_snr_peak_region = 2 # how many S/N units below maximum is still inside the peak region
        self.peak_region = [] # a map of the peak region, with only the part above the cut (peak-num_snr_peak_region) not zeroed out (used for error estimates)
        self._version = 1.01
        
        if not empty(finder):
            self.update_from_finder(finder)
            self.calculate()
            
    def update_from_finder(self, finder):
        
        for att in dir(self): # load all attributes in "self" that exist in "finder"
            if hasattr(finder, att) and not callable(getattr(finder,att)):
                setattr(self, att, getattr(finder, att))
    
#        self.psf = finder.input_psf
        self.im_size = finder._im_size_tr
        self.was_convolved = bool(finder.use_conv)
        self.was_expanded = bool(finder.useExpand())
        
    def calculate(self):
        # these are the raw results from this subframe        
        self.radon_y1 = (2**self.radon_step)*self.radon_max_idx[1] # size of each slice in y, times the slice number (index)
        self.radon_y2 = (2**self.radon_step)*(self.radon_max_idx[1]+1) # same thing, top index (should we include the last pixel?)
        offset = (self.subframe.shape[2]-self.im_size[1])//2 # this is added on either side if was_expanded
        self.radon_x0 = self.radon_max_idx[2] - offset # position of x0 in the subframe (removing the expanded pixels)
        self.radon_dx = self.radon_max_idx[0]-self.subframe.shape[0]//2 # the angle is in dim0 needs to offset for negative angles
        
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
            self.x0 = self.radon_x0 - self.y1/self.a - offset
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
        f = math.fabs(math.sin(math.radians(self.th)))
        self.I = self.snr*math.sqrt(self.noise_var*2*math.sqrt(math.pi)*self.psf_sigma/(self.L*f))
        self.snr_fwhm = self.I*0.81/math.sqrt(self.noise_var)
        
        if self.transposed:
            self.x1,self.y1 = self.y1,self.x1
            self.x2,self.y2 = self.y2,self.x2
            self.a = 1/self.a
            self.b, self.x0 = self.x0, self.b
            self.th = 90 - self.th
            
        self.x1 = round(self.x1)
        self.x2 = round(self.x2)
        self.y1 = round(self.y1)
        self.y2 = round(self.y2)
        self.x0 = round(self.x0)
    
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
    
    def subtractStreak(self, M_in, width=None):
        
        if empty(width):
            width= self.subtract_psf_widths
        
        # these are really rough estimates. Can improve this by looking at the error ellipse and subtracting y and dy values inside that range only 
        shift_array = np.arange(-self.psf_sigma*width, self.psf_sigma*width+1)
        
        M_sub = np.array(M_in)
        
        for shift in shift_array:
            if self.transposed:
                x1 = self.x1
                x2 = self.x2
                y1 = self.y1 + shift
                y2 = self.y2 + shift
            else:
                x1 = self.x1 + shift
                x2 = self.x2 + shift
                y1 = self.y1
                y2 = self.y2
            
        (xlist, ylist, n) = listPixels(x1,x2,y1,y2, M_in.shape)
        if not empty(xlist):
            M_sub[xlist[0], ylist[0]] = 0
        
        return M_sub
                
    @property
    def radon_dy(self):
        return self.radon_y2-self.radon_y1


    