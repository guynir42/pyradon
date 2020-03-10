# -*- coding: utf-8 -*-
"""
Created on Sun April 8 15:54:00 2018

@author: guynir@weizmann.ac.il
"""
        
from __future__ import division   
import numpy as np        
import math
import matplotlib.pyplot as plt
import warnings

from numba import njit

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
            size = (np.maximum(sigma_x, sigma_y)*20).item(0)
        
        if np.size(size)==1:
            size = (np.asarray(size)).item(0)
            size = (size, size)
                
        # x = np.linspace(-(size[1]-1)/2,(size[1]-1)/2,size[1])
        # y = np.linspace(-(size[0]-1)/2,(size[0]-1)/2,size[0])
        # xgrid,ygrid = np.meshgrid(x,y)

        size = (round(size[0]), round(size[1]))

        (y0,x0) = np.indices(size, dtype='float32')
        
        x0-=size[1]/2
        y0-=size[0]/2
        
        x = x0*math.cos(rotation_radians) + y0*math.sin(rotation_radians) - offset_x
        y = -x0*math.sin(rotation_radians) + y0*math.cos(rotation_radians) - offset_y
        
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
    
def model(im_size, x1, x2, y1, y2, sigma, replace_value=0, threshold=1e-10, oversample=4):

    if empty(im_size):
        raise('Must supply a valid size as first input to model')
    else:
        im_size = imsize(im_size)
    
    if oversample:
        im_size = tuple(s*oversample for s in im_size)
        x1 = (x1-0.5)*oversample + 0.5
        x2 = (x2-0.5)*oversample + 0.5
        y1 = (y1-0.5)*oversample + 0.5
        y2 = (y2-0.5)*oversample + 0.5
        sigma = sigma*oversample
        
#    (x,y) = np.meshgrid(range(im_size[1]), range(im_size[0]), indexing='xy')
    (y,x) = np.indices(im_size, dtype='float32')

    if x1==x2:
        a = float('Inf')  # do we need this?
        b = float('NaN')  # do we need this?
        d = np.abs(x - x1)  # distance from vertical line
    else:
        a = (y2-y1)/(x2-x1) # slope parameter
        b = (y1*x2 - y2*x1)/(x2-x1) # impact parameter
        d = np.abs(a * x - y + b) / np.sqrt(1 + a ** 2)  # distance from line

    M0 = (1/np.sqrt(2.*np.pi)/sigma)*np.exp(-0.5*d**2/sigma**2) # infinite line
    
    # must clip this line! 
    if x1==x2 and y1==y2: # this is extremely unlikely to happen...
        M0 = np.zeros(M0.shape)
    elif x1==x2: # vertical line (a is infinite)
        if y1>y2:
            M0[y>y1] = 0
            M0[y<y2] = 0
        else:
            M0[y<y1] = 0
            M0[y>y2] = 0
        
    elif y1==y2: # horizontal line
        if x1>x2:
            M0[x>x1] = 0
            M0[x<x2] = 0
        else:
            M0[x<x1] = 0
            M0[x>x2] = 0
        
    elif y1<y2:
        M0[y<(-1/a*x+y1+1/a*x1)] = 0
        M0[y>(-1/a*x+y2+1/a*x2)] = 0
    else:
        M0[y>(-1/a*x+y1+1/a*x1)] = 0
        M0[y<(-1/a*x+y2+1/a*x2)] = 0
    
    
    M1 = (1/np.sqrt(2*np.pi)/sigma)*np.exp(-0.5*((x-x1)**2+(y-y1)**2)/sigma**2);
    M2 = (1/np.sqrt(2*np.pi)/sigma)*np.exp(-0.5*((x-x2)**2+(y-y2)**2)/sigma**2);
    
    #print('M0: '+str(M0.shape)+' M1: '+str(M1.shape)+' M2: '+str(M2.shape))    
    
    M = np.fmax(M0,np.fmax(M1,M2))
    
    if oversample>1:
        M = downsample(M, oversample)/oversample
    
#    print(str(M.shape))    
    
    M[M<threshold] = replace_value
    
    return M

def downsample(I, factor=2, normalization='sum'):
    
    import scipy.signal
        
    factor = int(round(factor))
    
    if factor==1:
        return I

    k = np.ones((factor, factor), dtype=I.dtype)
    if normalization=='mean':
        k = k/np.sum(k)
    
#    print('I: '+str(I.dtype)+' '+str(I.shape)+' k: '+str(k.dtype)+' '+str(k.shape))
    
    I_conv = scipy.signal.convolve2d(I,k,mode='same')

#    print(str(I_conv.shape))

    return I_conv[factor-1::factor, factor-1::factor]

def jigsaw(M, cut_size, pad_value=float('NaN'), output_corners=None):  # cut an image into small cutouts and return them in a 3D array

    S = imsize(M)  # size of the input
    C = imsize(cut_size)  # size of the cutouts

    corner_x = list(range(0, S[1], C[1]))  # x position of the corner of each cutout in the input image
    corner_y = list(range(0, S[0], C[0]))  # y position of the corner of each cutout in the input image

    N = len(corner_x)*len(corner_y)  # number of cutouts

    if np.isnan(pad_value):
        M_out = np.empty((N, C[0], C[1]), dtype=M.dtype)
        M_out[:] = np.nan
    elif pad_value==0:
        M_out = np.zeros((N, C[0], C[1]), dtype=M.dtype)
    else:
        M_out = np.ones((N, C[0], C[1]), dtype=M.dtype) * pad_value

    counter = 0

    if output_corners is not None:
        output_corners.clear()

    for cy in corner_y:
        for cx in corner_x:

            c2x = min(cx+C[1], S[1])
            c2y = min(cy+C[0], S[0])
            # print("cx= %d | c2x= %d | cy= %d | c2y= %d" % (cx,c2x,cy,c2y))
            M_out[counter,0:c2y-cy,0:c2x-cx] = M[cy:c2y,cx:c2x]

            if output_corners is not None:
                output_corners.append((cy,cx))

            counter+=1

    return M_out

def image_stats(M, cut_size=32):

    cutouts = jigsaw(M, cut_size)
    # print(cutouts.shape)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = np.nanmedian(np.nanmean(cutouts, (1,2)))
        v = np.nanmedian(np.nanvar(cutouts, (1,2)))

    return m,v

if __name__=='__main__':
    
    print("this is a test for model...")
    
    M = model((500, 400), 350, 200, 300, 400, 3)
    
    plt.imshow(M)
