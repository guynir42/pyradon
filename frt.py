# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 13:56:51 2017

@author: guyn
"""

import numpy as np
import math
from pyradon.utils import empty
import pyradon.finder

def frt(M_in, transpose=False, expand=False, padding=True, partial=False, finder=None, output=None):
    """ Fast Radon Transform (FRT) of the input matrix M_in (must be 2D numpy array)
    Additional arguments: 
     -transpose (False): transpose M_in (replace x with y) to check all the other angles. 
     -expand (False): adds zero padding to the sides of the passive axis to allow for corner-crossing streaks
     -padding (True): adds zero padding to the active axis to fill up powers of 2. 
     -partial (False): use this to save second output, a list of Radon partial images (useful for calculating variance at different length scales)
     -finder (None): give a "finder" object that is used to scan for streaks. Must have a "Scan" method. 
     -output (None): give the right size array for FRT to put the return value into it.
    """
#    print "running FRT with: transpose= "+str(transpose)+", expand= "+str(expand)+", padding= "+str(padding)+", partial= "+str(partial)+", finder= "+str(finder)

    ############### CHECK INPUTS AND DEFAULTS #################################

    if empty(M_in):
        return

    if M_in.ndim>2:
        raise Exception("FRT cannot handle more dimensions than 2D")

    if not empty(finder):
        if not isinstance(finder, pyradon.finder.Finder):
            raise Exception("must input a pyradon.finder.Finder object as finder...")

        scan_method = getattr(finder, 'scan', None)
        if not scan_method or not callable(scan_method):
            raise Exception("finder given to FRT doesn't have a scan method")
    
        finder.last_streak = []
        
    ############## PREPARE THE MATRIX #########################################
    
    M = np.array(M_in) # keep a copy of M_in to give to finalizeFRT
    
    if transpose:
        M = M.T
    
    if padding: 
        M = padMatrix(M)
        
    if expand: 
        M = expandMatrix(M)
    
    M_partial = []
    
    if not empty(finder):
        finder._im_size_tr = M.shape
        finder.im_size = M_in.shape
    
    (Nrows, Ncols) = M.shape
    dx = np.array([0])
        
    M = M[np.newaxis,:,:]
    
    for m in range(1, int(math.log(Nrows,2))+1): # loop over logarithmic steps
        
        M_prev = M
        dx_prev = dx
        
        Nrows = M.shape[1] # number of rows in M_prev! 
        
        max_dx = 2**(m)-1
        dx = range(-max_dx, max_dx+1)
        M = np.zeros((len(dx), Nrows/2, Ncols), dtype=M.dtype)
        
        counter = 0;
        
        for i in range(Nrows/2): # loop over pairs of rows  (number of rows in new M)
            
            for j in range(len(dx)): # loop over different shifts
                
                # find the value and index of the previous shift
                dx_in_prev = int(float(dx[j])/2)
                j_in_prev = dx_in_prev + int(len(dx_prev)/2)
                # print "dx[%d]= %d | dx_prev[%d]= %d | dx_in_prev= %d" % (j, dx[j], j_in_prev, dx_prev[j_in_prev], dx_in_prev)
                gap_x = dx[j] - dx_in_prev # additional shift needed
                
                M1 = M_prev[j_in_prev, counter, :]
                M2 = M_prev[j_in_prev, counter+1,:]
                
                M[j,i,:] = shift_add(M1, M2, -gap_x)
                
            counter+=2 
            
        if finder:
            finder.scan(M, transpose)
        if partial:
            M_partial.append(M)
    
    # end of loop on m
    
    M_out = np.transpose(M, (0,2,1))[:,:,0] # lose the empty dimension
    
    if not empty(finder):
        finder.finalizeFRT(M_in, transpose, M_out)
    
    if partial:
        if not empty(output):
            np.copyto(output, M_partial)
        return M_partial
            
    else:
        if not empty(output):
            np.copyto(output, M_out)
        return M_out
                
def padMatrix(M):
    N = M.shape[0]
    dN = int(2**math.ceil(math.log(N,2))-N)
#    print "must add "+str(dN)+" lines..."
    M = np.vstack((M, np.zeros((dN, M.shape[1]))))
    return M 
    
def expandMatrix(M):
    Z = np.zeros((M.shape[0],M.shape[0]))
    M = np.hstack((Z,M,Z))
    return M 
    
def shift_add(M1,M2, gap):
    
    output = np.zeros_like(M2)
    
    if gap>0:
        output[:gap] = M1[:gap]
        output[gap:] = M1[gap:]+M2[:-gap]
    elif gap<0:
        output[gap:] = M1[gap:]
        output[:gap] = M1[:gap]+M2[-gap:]
    else:
        output = M1+M2
    
    return output

if __name__=="__main__":
    print "this is a test for radon.frt"

    import time
    
    print "image of ones((2048,2048)) into FRT"
    t = time.time()
    
    M = np.ones((2048,2048))
    R = frt(M)
     
    print "Elapsed time: "+str(time.time()- t)
