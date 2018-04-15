Streak detection Python package, based on the Fast Radon Transform (FRT)

Written by Guy Nir (guy.nir@weizmann.ac.il)

ORIENTATION:
-------------
This package includes three classes and a function (and some utilities). 
The function is frt(), which perfoms a Fast Radon Transform (FRT) on some data. 
This function can be used as-is to do transformations, for streak detection or other uses. 

To do more advanced searches more code is needed:
The three classes included in this package are:
(1) Finder: this object contains useful information such as a Radon variance map, 
    needed for normalizing the Radon images and checking Radon peaks against a S/N threshold. 
    Also the finder can be setup to find short streaks and multiple streaks (that the frt() function 
    cannot accomplish alone). 
    The Finder can also convolve the images with a given PSF;
    It can keep track of the best S/N results and find a automatic threshold.
    The results of all streaks that are found when using the Finder are saved 
    as a list of Streak objects inside Finder. 
(2) Streak: this object keeps track of the raw Radon coordinates where the streak
    was found, and also calculates the streak parameters in the input image. 
    Each Streak object keeps a single streak data, and can be used to display
    the streak in the original image, to display some statistics on it, and 
    to subtract itself from the image. 
(3) Simulator: this object can generate streaks, with a given PSF width and noise, 
    and can automatically feed the resulting images into the Finder it contains. 
    The simulator can make multiple streaks of different intensities and coordinates, 
    and can simulate random streaks with parameters chosen uniformly in a user-defined range. 


NOTEs: *this package is based on the MATLAB package (that has some additional functionality)
        Available on github: https://github.com/guynir42/radon
       *use the jupyter notebook "demo.ipynb" to see some examples.


