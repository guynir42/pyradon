# -*- coding: utf-8 -*-
"""
Created on Mon Dec 25 17:40:22 2017

The Simulator class creates images with "realistic" streaks in them: 
streaks have width according to the PSF size, and white noise background added. 

The key parameters to input before making a streak are:
-x1,x2,y1,y2: the streak start and end points (normalized units!) e.g, x1=0.1, x2=0.2, y1=0, y2=1
-im_size: the image size as a scalar (makes square images only, for now). Use powers of 2. Default is 512.
-bg_noise_var: noise variance (counts^2 per pixel). Default is 1.
-psf_sigma: width parameter of the PSF. default is 2. 
-intensity: counts/unit length (diagonal lines have slightly larger value per pixel)

You can also turn on/off the source noise (use_source_noise) or background by 
setting bg_noise_var to zero. 

The streaks are automatically input to the Finder object, that returns (hopefully) 
with the right streak parameters. 


@author: guy.nir@weizmann.ac.il
"""

import scipy.signal
import numpy as np
import math

from utils import empty, scalar, imsize, gaussian2D, model
from finder import Finder


class Simulator:
    def __init__(self):

        # objects
        self.finder = Finder()

        # outputs
        self.image_clean = []
        self.image = []
        self.psf = []

        # switches
        self.im_size = 512  # assume we want to make square images
        self.intensity = 10
        self.bg_noise_var = 1
        self.use_source_noise = True
        self.psf_sigma = 2

        # these parameters are in units of image size
        self.x1 = 0
        self.x2 = 1
        self.y1 = 0
        self.y2 = 1

        self.debug_bit = 1

    @property
    def L(self):
        # return math.sqrt((self.x2-self.x1)**2+(self.y2-self.y1)**2)
        x1 = min(max(self.x1, 0), 1)
        x2 = min(max(self.x2, 0), 1)
        y1 = min(max(self.y1, 0), 1)
        y2 = min(max(self.y2, 0), 1)

        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    @property
    def th(self):
        return math.degrees(math.atan2(self.y2-self.y1, self.x2-self.x1))

    @property
    def a(self):
        if self.x1 == self.x2:
            return float('NaN')
        else:
            return (self.y2-self.y1)/(self.x2-self.x1)

    @property
    def b(self):
        return self.y2-self.a*self.x2

    @property
    def x0(self):
        if self.x1 == self.x2:
            return self.x1
        elif self.a == 0:
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
        self.image_clean = []
        self.image = []
        self.psf = []

    def isVertical(self):

        val1 = self.th >= 45 and self.th <= 135
        val2 = self.th >= -135 and self.th <= -45

        return val1 or val2

    def calcSNR(self):

        snr = self.intensity*math.sqrt(self.L*self.im_size/self.bg_noise_var)

        snr = snr/math.sqrt(2*math.sqrt(math.pi)*self.psf_sigma)

        return abs(snr)

    def makeImage(self):

        if self.debug_bit > 1:
            print("makeImage()")

        self.clear()
        self.makeClean()
        self.addNoise()

    def makeClean(self):
        self.image_clean = self.intensity*model(self.im_size, self.x1*self.im_size,
                                                self.x2*self.im_size, self.y1*self.im_size,
                                                self.y2*self.im_size, self.psf_sigma)

    def addNoise(self):

        if self.debug_bit > 2:
            print("addNoise()")

        bg = np.ones(imsize(self.im_size), dtype="float32")*self.bg_noise_var

        if self.use_source_noise:
            var = bg + np.abs(self.image_clean)
        else:
            var = bg

        self.image = np.random.normal(self.image_clean, np.sqrt(var)).astype('float32')

    def find(self):

        if self.debug_bit > 1:
            print("find()")
#        self.image_final = np.concatenate((self.image_final ,np.zeros((self.image_final.shape[0],300))),1) # this is just for testing non-square input images...
        #self.finder.input_psf = self.psf_sigma
        #self.finder.input_var = self.bg_noise_var
        # self.finder.findSingle(self.image)
        self.finder.input(self.image, psf=self.psf_sigma, variance=self.bg_noise_var)

    def run(self):

        if self.debug_bit > 1:
            print("run()")
        self.clear()
        self.makeImage()
        self.find()

        if empty(self.finder.streaks):
            # print("No streaks found. Maximal S/N= %f" % max(np.max(self.finder.radon_image), np.max(self.finder.radon_image_trans)))
            print("No streaks found. Maximal S/N= %f" % self.finder.best_SNR)
        else:

            s = self.finder.streaks[0]

            if self.debug_bit:
                print("SIMULATED : S/N= %4.2f | I= %5.2f | L= %6.1f | th= %4.2f | x0= %4.1f" %
                      (self.calcSNR(), self.intensity, self.L*self.im_size, self.th, self.x0*self.im_size))
                print("CALCULATED: S/N= %4.2f | I= %5.2f | L= %6.1f | th= %4.2f | x0= %4.1f" %
                      (s.snr, s.I, s.L, s.th, s.x0))

            if self.debug_bit > 1:

                input_xy = (self.x1, self.x2, self.y1, self.y2)
                input_xy = tuple((int(round(x*self.im_size)) for x in input_xy))
                print("INPUT: x1= % 4d | x2= % 04d | y1= % 4d | y2= % 4d" % input_xy)

                print("FOUND: x1= % 4d | x2= % 04d | y1= % 4d | y2= % 4d" % (s.x1, s.x2, s.y1, s.y2))


# test (reload object)
if __name__ == "__main__":

    import matplotlib.pyplot as plt

    print("this is a test for Simulator and Finder...")

    if 's' not in locals() or not isinstance(s, Simulator):
        s = Simulator()
        s.debug_bit = 1

    s.x1 = 3.0/8
    s.x2 = 0.5
    s.y1 = 1/3
    s.y2 = 1.0/2

    s.x1 = 0.2
    s.y1 = 0.01
    s.x2 = 1
    s.y2 = 1.5

    s.finder.use_subtract_mean = 0
    s.run()

    fig, ax = plt.subplots()
    ax.imshow(s.image)
    if not empty(s.finder.streaks):
        for st in s.finder.streaks:
            st.plotLines(ax)
