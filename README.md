Streak detection Python package, based on the Fast Radon Transform (FRT)

Written by Guy Nir (guy.nir@weizmann.ac.il)
Based on the methods described in Nir et al. 2018, Astronomical Journal, vol 156, number 5,
(https://ui.adsabs.harvard.edu/abs/2018AJ....156..229N/abstract)

# ORIENTATION:

This package includes three classes and a function (and some utilities).
The function is `frt()`, which performs a Fast Radon Transform (FRT) on some data.
This function can be used as-is to do transformations, for streak detection or other uses.

To do more advanced searches more code is needed:
The three classes included in this package are:

1. `Finder`: this object contains useful information such as a Radon variance map,
   needed for normalizing the Radon images and checking Radon peaks against a S/N threshold.
   Also, the finder can be setup to find short streaks and multiple streaks
   (that the frt() function cannot accomplish alone).
   The finder can also convolve the images with a given PSF;
   it can scan multiple thresholds, effectively removing all point sources;
   it can section large images into smaller ones to save time and memory;
   and it can store the resulting streaks and the S/N values for all images
   which is useful for finding the best threshold and estimating the false alarm rates.
   The results of all streaks that are found when using the Finder are saved
   as a list of `Streak` objects inside `Finder`.
2. `Streak`: this object keeps track of the raw Radon coordinates where the streak
   was found, and also calculates the streak coordinates in the input image,
   along with other parameters of the streak.
   Each `Streak` object keeps a single streak's data, and can be used to display
   the streak in the original image, to display some statistics on it, and
   to subtract itself from the image.
   `Streak` objects are only created by the `Finder` object.
3. `Simulator`: this object can generate images with streaks in them,
   with a given PSF width and noise,
   and can automatically feed the resulting images into the Finder it contains.
   The simulator can make multiple streaks of different intensities and coordinates,
   and can simulate random streaks with parameters chosen uniformly in a user-defined range.

### Using the Finder

To use the full power of the `Finder` object,
use the `input()` function, giving it three parameters:

```
f = Finder()
f.input(im=image_array, psf=psf_array_or_scalar, variance=var_array_or_scalar)
```

The image should be a `numpy` array with two dimensions.

The PSF is given either as a scalar, representing the "sigma"
coefficient of a Gaussian.
This is a useful approximation in many cases.
If the FWHM of the seeing is known, it should be divided by 2.355.
If the actual PSF is known, it should be input as a small 2D `numpy` array.

The variance can be given as a scalar which represents the average
variance of the image given to the finder,
or it can be given as a 2D map of the same size as the image.
If the mean noise standard deviation is known (or background noise image)
the input to the finder should be the square of these values.
The variance can be different for each image, or constant for
a series of images given in turn to the finder, that will
just re-use the existing variance values if none are given.
The default variance values is 1, which assumes the image is noramlized.

The variance map, even if it were given as a mean value which is
replicated into a uniform map of the same size as the image,
is necessary to normalize streaks of different length.
For example, even if the variance is set to the default scalar value,
the signal for longer streaks will be divided by a larger factor
than the signal from shorter streaks.
This is achieved by saving a Radon image of the variance map,
which is divided by the Radon image of the streak images.
These variance Radon maps are cached by the Finder to be re-used
with multiple images if needed,
and are refreshed when new variance values are given.

### Preprocessing input images

### Using parts of the finding algorithm

### Fast Radon Transform function details

### Reading the results from Streak objects

### Cleaning streaks off images

### Using the Simulator

### NOTES

- this package is based on the MATLAB package
- (that has some additional GUI functionality but is otherwise less updated)
  Available on github: https://github.com/guynir42/radon
- Please make sure to cite Nir et al. 2018 (https://ui.adsabs.harvard.edu/abs/2018AJ....156..229N/abstract)
  if making use of this software.
- use the jupyter notebook "demo.ipynb" to see some examples.
