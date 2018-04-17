# -*- coding: utf-8 -*-
"""
Created on Mon Dec 25 17:40:22 2017

The Simulator class creates images with "realistic" streaks in them. 
This happens in three stages:
(1) choose pixels along a line and add the same value to each pixel on the line. 
(2) convolve the line with a Gaussian PSF. 
(3) add Gaussian white noise. 

The key parameters to input before making a streak are:
-x1,x2,y1,y2: the streak start and end points (normalized units!) e.g, x1=0.1, x2=0.2, y1=0, y2=1
-im_size: the image size as a scalar (makes square images only, for now). Use powers of 2. Default is 512.
-bg_noise_var: noise variance (counts^2 per pixel). Default is 1.
-psf_sigma: width parameter of the PSF. default is 2. 
-intensity: counts/unit length (diagonal lines have slightly larger value per pixel)

You can also turn on/off the source noise (use_source_noise)
and make lines not convolved with a PSF (use_conv) if you like. 

The streaks are automatically input to the Finder object, that returns (hopefully) 
with the right streak parameters. 


@author: guy.nir@weizmann.ac.il
"""

import scipy.signal
import numpy as np
import math

from utils import empty, scalar, imsize, gaussian2D, listPixels
import pyradon.finder

class Simulator:
    def __init__(self):
        
        # objects
        self.finder = pyradon.finder.Finder()
        
        # outputs
        self.image_line = []
        self.image_conv = []
        self.image_final = []
        self.psf = []
        
        self.x_list = []
        self.y_list = []
        self.num_pixels = 0
        
        # switches
        self.im_size = 512 # assume we want to make square images
        self.intensity = 10
        self.bg_noise_var = 1
        self.use_source_noise = True
        self.use_conv = True
        self.psf_sigma = 2
        
        # these parameters are in units of image size
        self.x1 = 0
        self.x2 = 1
        self.y1 = 0
        self.y2 = 1
    
        self.debug_bit = 1
        
    @property
    def L(self):
        return math.sqrt((self.x2-self.x1)**2+(self.y2-self.y1)**2)
    
    @property
    def th(self):
        return math.degrees(math.atan2(self.y2-self.y1, self.x2-self.x1))
    
    @property
    def a(self):
        if self.x1==self.x2:
            return float('NaN')
        else:
            return (self.y2-self.y1)/(self.x2-self.x1)
    
    @property
    def b(self):
        return self.y2-self.a*self.x2
    
    @property
    def x0(self):
        if self.x1==self.x2:
            return self.x1
        elif self.a==0:
            return float('NaN')            
        else:
            return -self.b/self.a 
        
    @property
    def midpoint_x(self):
        return (self.x2+self.x1)/2
     
    @property
    def midpoint_y(self):
        return (self.y2+self.y1)/2
    
    @property
    def trig_factor(self):
        return max(math.fabs(math.cos(math.radians(self.th))), math.fabs(math.sin(math.radians(self.th))))
    
    def clear(self):
        self.image_line = []
        self.image_conv = []
        self.image_final = []
        self.psf = []
        
        self.x_list = []
        self.y_list = []
        self.num_pixels = 0
        
    def isVertical(self):
        
        val1 = self.th>=45 and self.th<=135
        val2 = self.th>=-135 and self.th<=-45
        
        return val1 or val2
    
    def calcSNR(self):
        
        f = self.trig_factor
        L = self.num_pixels/f
        
        snr = self.intensity*math.sqrt(L*f/self.bg_noise_var)
        
        if self.use_conv:
            snr = snr/math.sqrt(2*math.sqrt(math.pi)*self.psf_sigma)
    
        return abs(snr)
    
    def makeImage(self):
       
       if self.debug_bit>1:
           print "makeImage()"
           
       self.clear()
       self.drawLine()
       self.convolve()
       self.addNoise()
       
    def listPixels(self):
        
        if empty(self.im_size):
            raise Exception("Cannot listPixels without an image size!")
        if empty(self.x1) or empty(self.x2) or empty(self.y1) or empty(self.y2):
            raise Exception("Cannot listPixels without x1,x2,y1,y2!")

        S = self.im_size
        if scalar(S):
            S = (S,S)

        x1_pix = S[0]*self.x1
        x2_pix = S[0]*self.x2
          
        y1_pix = S[1]*self.y1
        y2_pix = S[1]*self.y2

        x,y,N = listPixels(x1_pix, x2_pix, y1_pix, y2_pix, self.im_size)
        
        # these are lists because we may later add support for multiple lines
        self.x_list = x[0];
        self.y_list = y[0];
        self.num_pixels = N[0]
        
        
    def drawLine(self):
        
        if self.debug_bit>2:
           print "drawLine()"
        self.listPixels()
        self.image_line = np.zeros(imsize(self.im_size))
        
        self.image_line[self.y_list, self.x_list] = self.intensity/self.trig_factor
    
    def convolve(self):

        if self.debug_bit>2:
           print "convolve()"        
        
        self.psf = gaussian2D(self.psf_sigma)
        self.image_conv = scipy.signal.fftconvolve( self.image_line, np.rot90(self.psf, 2), mode='same')
    
    def addNoise(self):
        
        if self.debug_bit>2:
           print "addNoise()"
           
        bg = np.ones(imsize(self.im_size))*self.bg_noise_var        
        
        if self.use_source_noise:
            var = bg + np.abs(self.image_conv)
        else:
            var = bg
        
        self.image_final = np.random.normal(self.image_conv, np.sqrt(var))

    def find(self):
        
        if self.debug_bit>1:
           print "find()"
#        self.image_final = np.concatenate((self.image_final ,np.zeros((self.image_final.shape[0],300))),1) # this is just for testing non-square input images...
        self.finder.input(self.image_final, variance=self.bg_noise_var, psf=self.psf_sigma)
        
    def run(self):
        
        if self.debug_bit>1:
           print "run()"
        self.clear()
        self.makeImage()
        self.find()
        
        if empty(self.finder.streaks):
            print "No streaks found. Maximal S/N= %f" % max(np.max(self.finder.radon_image), np.max(self.finder.radon_image_trans))
        else:

            s = self.finder.streaks[0]
            
            if self.debug_bit:            
                print "SIMULATED : S/N= %4.2f | I= %5.2f | L= %6.1f | th= %4.2f | x0= %4.1f" % (self.calcSNR(), self.intensity, self.L*self.im_size, self.th, self.x0*self.im_size)
                print "CALCULATED: S/N= %4.2f | I= %5.2f | L= %6.1f | th= %4.2f | x0= %4.1f" % (s.snr, s.I, s.L, s.th, s.x0)
    
            if self.debug_bit>1:
    
                input_xy = (self.x1, self.x2, self.y1, self.y2)
                input_xy = tuple((int(round(x*self.im_size)) for x in input_xy))
                print "INPUT: x1= % 4d | x2= % 04d | y1= % 4d | y2= % 4d" % input_xy
                
                print "FOUND: x1= % 4d | x2= % 04d | y1= % 4d | y2= % 4d" % (s.x1, s.x2, s.y1, s.y2)
                
                
# test (reload object)
if __name__ == "__main__":
    
    print "this is a test for Simulator and Finder..."
    
    if 's' not in locals() or not isinstance(s, Simulator):
        s = Simulator()
        s.debug_bit=1
    
    s.x1 = 3.0/8
    s.x2 = 0.5
    s.y1 = 1/3
    s.y2 = 1.0/2
    
    s.run()
    