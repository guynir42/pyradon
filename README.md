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
from src.finder import Finder
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
The default variance values is 1, which assumes the image is normalized.

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

The Radon method is very sensitive to any positive bias in the image.
Diffuse sources, such as galaxies or even ghosts or reflections
can be mis-identified as streaks.
Stars, if too many of them are present in the image,
may be detected as streaks if they are aligned in a row.
Sometimes even two bright stars will trigger the algorithm.
On the other hand, any negative bias can reduce the detection efficiency.
Since the algorithm sums together many pixels into streaks,
even a small bias value can substantially affect the results.

So, an effort should be made to remove stars,
and any residual bias from the image.
If possible, finding and removing point sources
is a useful first step.
Measuring the background in a robust way
and reducing it is also a good idea
(e.g., using sigma clipping, measuring the b/g
on different parts of the image individually, etc.).
It is up to the user to provide images as clean as possible,
before using the tools in this package.

The `Finder` object does have some ability
to remove bias and unwanted sources.
First, it runs iteratively, and in each iteration
it can remove the bias by subtracting the image mean
before finding each streak.
Streaks that are found are removed from the image
and replaced with NaNs, so they do not count when
calculating the mean.

A second mitigation strategy to remove point sources
uses multiple iterations of the algorithm with different thresholds,
starting with a very high threshold and dividing by two each time.
For each threshold the `Finder` reduces any pixels above the threshold
(clips them to the threshold), and then finds streaks.
Since point sources have only a few pixels, their signal is never high
enough to trigger once they have been clipped.
On the other hand, streaks have multiple pixels,
each below the threshold, but when summed
have a high enough signal to trigger.
This can be useful for removing small artefacts
and individual stars, but will not guarantee that
multiple stars do not trigger together.

If other artefacts and extended sources still remain,
they will generally be identified as streaks,
but will usually not stop the iterative process from
finding actual streaks (from satellites and asteroids).
Therefore, some additional work may be needed to
classify the different objects identified by the algorithm.

### Clipping and sectioning

One option of the `Finder` is to clip and section
the input image before looking for streaks.
The clipping option is useful if the image edges
contain many artefacts or bias rows.
Another reason is to clip a small part of the image
so it fits into an integer power of 2.
For example if the image is 1050x1050 pixels,
the algorithm must zero-pad it to 2048 pixels on one axis,
to be able to run the FRT.
To save calculating all those empty pixels,
clipping the image to 1024x1024 will save calculation time,
while only forfeiting a small part of the image.

Sectioning can be used to carve the image into small portions.
While the runtime of the algorithm is not very different
when running on many subsections instead of once on a large image
(the algorithm scales as N\*log(N) where N is the total number of pixels),
there are some benefits to this mode:
(a) very large images can exhaust the system memory, causing significant slowdowns;
(b) if one section contains many streaks/artefacts, only that section will require
many iterations and many threshold levels;
(c) preprocessing can be more accurate for smaller images,
e.g., the mean of each section can be measured more robustly.
The main downside to sectioning is that very long streaks
are either not detected (since not all pixels can be summed at once)
and the ones that are found would be split into smaller streak objects.
If the section size is larger than the streak the user is interested in,
then sectioning can be safely used.
For example asteroid streaks tend to be few-pixels long,
while LEO satellites can cross the entire image.
If looking only for asteroids, sectioning could be useful.

The `Finder` parameters to control clipping and sectioning are:

- `use_sections`: enable or disable all clipping and sectioning
- `size_sections`: the size of the sections to be used
- `offset_sections`: the offset between lower-left corner of first section
  and the lower-left corner of the image.
- pad_sections: If True, will add zero padding
  to sections that extend beyond the image.
  If False, will clip parts not included in the sections,
  and drop the sections that are not completely ovelapping
  with the image.

To generate an image with a single section,
but slightly smaller than the original image (i.e., clipping)
set a small positive offset and a sections size slightly
smaller than the image size:

```
# assume f is a Finder with a 1048x1048 image
f.use_sections = True
f.size_sections = 1024
f.offset_sections = (12, 12)
f.pad_sections = False

```

In this case the section starts 12 pixels from the corner,
and ends 12 pixels from the other corner,
clipping the image to 1024x1024 pixels.
If using `size_sections=512`,
the clipping would be the same,
but there would be four sections,
each with a size of 512x512 pixels.

If the image is slightly smaller than a power of two,
padding can be used to pad the sections before
starting the analysis:

```
# assume f is a Finder with a 1000x1000 image
f.use_sections = True
f.size_sections = 1024
f.offset_sections = None
f.pad_sections = True
```

In this case the image is padded to 1024x1024 pixels,
with a single section.
If `size_sections=512`,
the padding would be the same,
but there would be four sections,
each with a size of 512x512 pixels.
If `pad_sections=False`,
three of the four sections would be clipped,
which would throw away almost 3/4 of the image.
Note that even if the sections are not powers of two,
they will be padded by the `FRT()` function.

### Using parts of the finding algorithm

The full `Finder` algorithm requires multiple iterations
and can be quite slow, especially if many streaks (or artefacts) are found.
To speed things up, in particular if the images are very clean,
one can use only a subset of the algorithm.

The full algorithm can be summarized:

1. Cut or clip the image into sections (this is done in the main function `input()`).
2. For each section, subtract the mean and filter with the image PSF (using `preprocess()`)
3. Choose the highest threshold level for each section,
   iterate over each threshold until reaching the minimal threshold,
   while clipping pixels above the threshold (this is done using `scan_thresholds()`)
4. For each threshold, detect multiple streaks on that section and the transpose of that section
   (this is done using `find_multi()`).
5. Each time a streak is found using a single call to the `FRT()` function,
   on a section or its transpose (this is done using `find_single()`).
6. Each streak found after the call to `find_single()` is removed from the image section
   and replaced with NaNs. If no streaks are found, the iteration stops and the next threshold is scanned.
7. All thresholds must be scanned, with a minimum of two calls to the `FRT()` function
   (one for each transpose). If streaks or artefacts are found, the `FRT()` may be called more times.
   Use `self.data._num_frt_calls` to keep track of the number of calls.
   This counter is reset when calling `self.clear()`.

In some cases, it is faster to skip the scanning of thresholds,
particularly if the images do not have any point sources.
In that case just running `preprocess()` and then `find_multi()` is sufficient.
This function will make sure to use the image and its transpose when searching,
which is important for finding streaks at all possible angles
(a single FRT can only find streaks in the -45 to 45 degree range).

If the image is known to contain only a single streak at most,
then running `preprocess()` and then `find_single()` twice on the image,
once with `transpose=False` and once with `transpose=True`.
The `Finder` will find the brightest streak in the image
and in the transposed image,
by comparing the Radon image to the Radon variance map.

In all the above cases, the `preprocess()` function
requires that the `Finder` is supplied an estimate of the PSF.
This can be given as an array or a scalar width parameter.
Either one can be given to `finder.data.psf`.
The `Finder` also needs an estimate of the variance
in the image to properly estimate the S/N of the streaks.
Supply the scalar variance value or a variance map of the
same size as the image to `finder.data.variance`.

# examples:

Find the brightest streak in an image:

```
f = Finder()
f.data.psf = psf_width_or_small_array
f.data.variance = variance_mean_or_array_of_same_size_as_image
f.data.image = image
f.preprocess()
s1 = f.find_single(transpose=False)
s2 = f.find_single(transpose=True)

# choose the streak with the best S/N
if s1 is not None and s2 is not None:
   s = s1 if s1.snr > s2.snr else s2
else:
    s = s1 or s2

s.print()  # print the streak coordinates and properties
```

Find up to 10 streaks in the image
and up to 10 streaks in the image transposed:

```
# assume f already has a preprocessed image, a PSF and a variance:
f.pars.num_iterations = 10  # or give num_iter directly
f.find_multi(num_iter=10)  # if num_iter=None, will use the default given above

for s in f.streaks:  # can be up to 20 streaks
    s.print()  # print the streak details for each streak

```

For scanning multiple thresholds,
and finding multiple streaks at each threshold:

```
f.pars.threshold = 7.5  # or give threshold directly as "min_threshold"
f.scan_thresholds(min_threshold=None)
# maximum threshold is 7.5 times 2^N,
# where N is big enough to not clip any pixels

# a list of streaks accumulated
# over all thresholds/iterations/transposes:
f.streaks

```

If there is no need for streak detection,
calculation of streak coordinates and S/N,
then a single call to `FRT()` may be sufficient.
In this case, the `Finder` will not be used at all.
The output of the function is the raw Radon image,
that can still be used to find streaks by
searching for hotspots in the transformed image.
However, caution should be used as some
areas of the raw Radon image have higher signal
just because they represent longer streaks,
with more pixels and more accumulated noise.
To normalize this result,
the Radon variance must be used.

### Fast Radon Transform function details

### Reading the results from Streak objects

### Cleaning streaks from images

### Using the Simulator

### NOTES

- this package is based on the MATLAB package
  (that has some additional GUI functionality but is otherwise less updated)
  Available on github: https://github.com/guynir42/radon
- Please make sure to cite Nir et al. 2018
  (https://ui.adsabs.harvard.edu/abs/2018AJ....156..229N/abstract)
  if making use of this software.
- use the jupyter notebook "demo.ipynb" to see some examples.
