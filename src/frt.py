import numpy as np
import math


def FRT(im, transpose=False, expand=False, padding=True, partial=False, output=None):
    """
    Fast Radon Transform (FRT) of the input matrix im

    Parameters
    ----------
    transpose: scalar boolean
        Choose if the function should transpose im
        (replace x with y) to check angles above 45 degrees.
        Default is False.
    expand: scalar boolean
        Choose if to add zero padding to the sides
        of the passive axis to allow for corner-crossing streaks.
        This should not be used if searching for short streaks
        (when using the partial results).
        Default is False.
    padding: scalar boolean
        Choose if to add zero padding to the active axis to fill up powers of 2.
        This is really important, since the algorithm expects to be able to fold
        the data into powers of two. Default is True.
    partial: scalar boolean
        Choose of to output a list of partial Radon images
        (useful for calculating variance at different length scales)
        Default is False.
    output: None or np.ndarray
        Give an array with the right size,
        for FRT to put the return value into it.
        Note that if partial=True then output must
        be a list of arrays with the right dimensions.
    """

    # CHECK INPUTS AND DEFAULTS
    if im.ndim > 2:
        raise RuntimeError("FRT cannot handle more dimensions than 2D")

    # PREPARE THE MATRIX
    im = np.nan_to_num(im, copy=True)  # get rid of NaNs (replace with zeros)

    if transpose:
        im = im.T

    if padding:
        im = pad_matrix(im)

    if expand:
        im = expand_matrix(im)

    num_folds = get_num_log_foldings(im)
    (num_rows, num_cols) = im.shape

    im_out = []

    if output is not None:  # output will have the partial transforms list
        if partial:
            for m in range(2, num_folds + 1):
                if output[m - 1].shape != get_partial_dims(im, m):
                    raise RuntimeError(
                        f"Wrong dimensions of output array[{m - 1}]: {output[m - 1].shape},"
                        f" should be {get_partial_dims(im, m)}"
                    )

            im_out = output

        else:
            if output.shape[0] != 2 * im.shape[0] - 1:
                raise RuntimeError(
                    f"Y dimension of output ({output.shape[0]})"
                    f" is inconsistent with (padded and doubled) input ({im.shape[0] * 2 - 1})"
                )
            if output.shape[1] != im.shape[1]:
                raise RuntimeError(
                    f"X dimension of output ({output.shape[1]})"
                    f" is inconsistent with (expanded?) input ({im.shape[1]})"
                )

    dx = np.array([0])

    im = im[np.newaxis, :, :]

    for m in range(1, num_folds + 1):  # loop over logarithmic steps

        im_prev = im
        dx_prev = dx

        num_rows = im_prev.shape[1]

        max_dx = 2 ** (m) - 1
        dx = range(-max_dx, max_dx + 1)
        if partial and output is not None:
            # we already have memory allocated for this result
            im = im_out[m - 1]
        else:
            # make a new array each time
            im = np.zeros((len(dx), num_rows // 2, num_cols), dtype=im.dtype)

        counter = 0

        for i in range(
            num_rows // 2
        ):  # loop over pairs of rows (number of rows in new M)

            for j in range(len(dx)):  # loop over different shifts

                # find the value and index of the previous shift
                dx_in_prev = int(float(dx[j]) / 2)
                j_in_prev = dx_in_prev + int(len(dx_prev) / 2)
                gap_x = dx[j] - dx_in_prev  # additional shift needed

                im1 = im_prev[j_in_prev, counter, :]
                im2 = im_prev[j_in_prev, counter + 1, :]

                im[j, i, :] = shift_add(im1, im2, -gap_x)

            counter += 2

        # only append to the list if it hasn't been given from the start using "output"
        if partial and output is None:
            im_out.append(im)

    # end of loop on m

    # we don't care about partial transforms, we were not given an array to fill
    if not partial:
        im_out = im[:, 0, :]  # lose the empty dim

        if output is not None:  # do us a favor and also copy it into the array
            np.copyto(output, im_out)
            # this can be made more efficient if we use the "output" array as
            # target for assignment at the last iteration on m.
            # this will save an allocation and a copy of the array.
            # however, this is probably not very expensive and not worth
            # the added complexity of the code

    return im_out


# end of FRT algorithm, helper functions:


def pad_matrix(im):
    size = im.shape[0]
    size_diff = int(2 ** math.ceil(math.log(size, 2)) - size)
    # im = np.vstack((im, np.zeros((size_diff, im.shape[1]), dtype=im.dtype)))
    im = np.pad(im, [(0, size_diff), (0, 0)])
    return im


def expand_matrix(im):
    # Z = np.zeros((M.shape[0], M.shape[0]), dtype=M.dtype)
    # M = np.hstack((Z, M, Z))
    im = np.pad(im, [(0, 0), (im.shape[0], im.shape[0])])
    return im


def shift_add(im1, im2, gap):

    output = np.zeros_like(im2)

    if gap > 0:
        output[:gap] = im1[:gap]
        output[gap:] = im1[gap:] + im2[:-gap]
    elif gap < 0:
        output[gap:] = im1[gap:]
        output[:gap] = im1[:gap] + im2[-gap:]
    else:
        output = im1 + im2

    return output


def get_partial_dims(im, log_level):
    x = im.shape[1]
    y = im.shape[0]

    y = int(y / 2**log_level)
    z = int(2 ** (log_level + 1) - 1)

    return z, y, x


def get_num_log_foldings(im):
    return int(np.ceil(np.log2(im.shape[0])))


def get_empty_partial_array_list(im):
    return [
        np.zeros(get_partial_dims(im, m), dtype=im.dtype)
        for m in range(1, int(get_num_log_foldings(im) + 1))
    ]


####################### MAIN ######################################

if __name__ == "__main__":

    import time
    import matplotlib.pyplot as plt

    print("this is a test for pyradon.frt")

    t = time.time()

    im = np.random.normal(0, 1, (2048, 2048)).astype(np.float32)

    print(f"random image of size {im.shape} sent to FRT")

    use_partial = True

    if use_partial:
        output = get_empty_partial_array_list(im)
    else:
        output = np.zeros((im.shape[0] * 2 - 1, im.shape[1]), dtype=im.dtype)

    radon = FRT(im, partial=use_partial, output=output)

    if use_partial:
        plt.imshow(radon[-1][:, 0, :])
    else:
        plt.imshow(radon)

    print(f"Elapsed time: {time.time() - t:.2f}")
