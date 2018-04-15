# -*- coding: utf-8 -*-
"""
Created on Sun April 8 15:54:00 2018

@author: guynir@weizmann.ac.il
"""
        
from __future__ import division   
import numpy as np        
import math

def empty(array):
    if array is not None and np.asarray(array).size>0:
        return False
    else:
        return True 
def scalar(array):
    return (not hasattr(array, '__len__')) and (not isinstance(array, str))

def imsize(array):
    if empty(array):
        return 0
    if np.size(array)==1:
        size = np.asscalar(np.asarray(array))
        return (size, size)
    if np.size(array)==2:
        return tuple(array)
    if np.ndim(array)>1 and hasattr(array, 'shape'):
        return (array.shape[-2], array.shape[-1])
        
    raise Exception("input to IMSIZE is not an image or a scalar or a two-element vector")

def compare_size(first, second):
    return np.all(np.array(first)==np.array(second))

def crop2size(image, im_size):

    if np.asarray(im_size).size==1:
        im_size = np.ones((2,))*np.asscalar(im_size)

    if image.dims==2:
        h,w = image.shape
        gap_h = np.maximum(0, (h-im_size[0])//2)
        gap_w = np.maximum(0, (w-im_size[1])//2)
    
        return image[gap_h:-gap_h, gap_w:-gap_w]
        
    else:
        (p,h,w) = image.shape
        gap_h = np.maximum(0, (h-im_size[0])//2)
        gap_w = np.maximum(0, (w-im_size[1])//2)
        return image[:, gap_h:-gap_h, gap_w:-gap_w]
    
def gaussian2D(sigma_x=2, sigma_y=None, rotation_radians=0, offset_x=0, offset_y=0, size=None, norm=1):
        
        if empty(sigma_y):
            sigma_y = sigma_x
        
        if empty(size):
            size = np.asscalar(np.maximum(sigma_x, sigma_y)*20)
        
        if np.size(size)==1:
            size = np.asscalar(np.asarray(size))
            size = (size, size)
                
        x = np.linspace(-(size[1]-1)/2,(size[1]-1)/2,size[1])
        y = np.linspace(-(size[0]-1)/2,(size[0]-1)/2,size[0])
        
        xgrid,ygrid = np.meshgrid(x,y)
        
        x = xgrid*math.cos(rotation_radians) + ygrid*math.sin(rotation_radians) - offset_x
        y = -xgrid*math.sin(rotation_radians) + ygrid*math.cos(rotation_radians) - offset_y
        
        G = np.exp(-0.5*(x**2/sigma_x**2+y**2/sigma_y**2))
        if norm==0:
            pass
        elif norm==1:
            G = G/np.sum(G)
        elif norm==2:
            G = G/math.sqrt(np.sum(G**2))
        
        return G
        
def fit_gaussian(data):

    from scipy.optimize import minimize    
    
    # initial guess (can improve this later)
    peak = np.max(data)
    width = 2
    rotation = 0
    offset_x = 0
    offset_y = 0
    
    x0 = np.array([peak, width, width, rotation, offset_x, offset_y])    
    
    def min_func(x):
        g = x[0]*gaussian2D(x[1], x[2], x[3], x[4], x[5], size=data.shape)
        return np.sum((g-data)**2)
    
    return minimize(min_func, x0, options={'disp':False})
    
def listPixels(x1,x2,y1,y2,size):
    
    if scalar(x1): x1 = [x1]        
    if scalar(x2): x2 = [x2]        
    if scalar(y1): y1 = [y1]
    if scalar(y2): y2 = [y2]
    if scalar(size): size = [size, size]
        
    N = max([len(x1), len(x2), len(y1), len(y2)])
    if len(x1)<N: x1.extend(x1[-1]*(N-len(x1)))
    if len(x2)<N: x2.extend(x2[-1]*(N-len(x2)))
    if len(y1)<N: y1.extend(y1[-1]*(N-len(y1)))
    if len(y2)<N: y2.extend(y2[-1]*(N-len(y2)))
    
    xlists = []
    ylists = []
    num_pixels = []
    
    for i in range(N):
        
        if x1[i]<0 and x2[i]<0: continue
        if x1[i]>=size[1] and x2[i]>=size[1]: continue
    
        if y1[i]<0 and y2[i]<0: continue
        if y1[i]>=size[0] and y2[i]>=size[0]: continue
    
        if x1[i]==x2[i]:
            a = float('nan')
        else:
            a = (y2[i]-y1[i])/(x2[i]-x1[i])
        
        
        if math.isnan(a) or math.fabs(a)>=1: # vertical (or closer to vertical) lines
            
            if y1[i]<y2[i]:
                y = np.arange(y1[i], y2[i])
            else:
                y = np.arange(y2[i], y1[i])

            if math.isnan(a): # if x1[i]==x2[i]
                x = np.ones(len(y))*x1[i]
            else:
                if y1[i]<y2[i]:
                    x = x1[i] + (y-y1[i])/a # these values are not rounded!
                else:
                    x = x2[i] + (y-y2[i])/a # these values are not rounded!

        else: # horizontal (or closer to horizontal) lines
            
            if x1[i]<x2[i]:
                x = np.arange(x1[i], x2[i])
                y = y1[i] + (x - x1[i])*a
            else:
                x = np.arange(x2[i], x1[i])
                y = y2[i] + (x - x2[i])*a
            
        # clip xy values where y is outside the frame...
        if y[0]<y[-1]: # ascending order
            ind_low = np.searchsorted(y, -0.5, side='right')
            ind_high = np.searchsorted(y, size[0]-0.5, side='left')            
        elif y[0]>y[-1]: # descending order
            ind_high = len(y) - np.searchsorted(y[::-1], -0.5, side='left')
            ind_low = len(y) - np.searchsorted(y[::-1], size[0]-0.5, side='right')
        else:
            ind_low = 0;
            ind_high = len(y)

        y = y[ind_low:ind_high]
        x = x[ind_low:ind_high]

        # clip xy values where x is outside the frame...
        if x[0]<x[-1]:        
            ind_low = np.searchsorted(x, -0.5, side='right')
            ind_high = np.searchsorted(x, size[1]-0.5, side='left')            
        elif x[0]>x[-1]:
            ind_high = len(x) - np.searchsorted(x[::-1], -0.5, side='left')
            ind_low = len(x) - np.searchsorted(x[::-1], size[1]-0.5, side='right')
        else:
            ind_low = 0
            ind_high = len(x)
        
        y = y[ind_low:ind_high]
        x = x[ind_low:ind_high]
    
        xlists.append(np.round(x).astype(int))
        ylists.append(np.round(y).astype(int))
        
        num_pixels.append(len(x))
        
    # end of loop on i
    
    return xlists, ylists, num_pixels
    
if __name__=='__main__':
    
    import matplotlib.pyplot as plt
    
    print "this is a test for listPixels..."
    S = 512
    x,y,n = listPixels(0, 512, 100, 300, S)
    print str(x)
    print str(y)
    
    z = np.zeros((S,S))
    z[y,x] = 1
    
    plt.imshow(z)