import numpy as np
import math

from utils import empty


def FRT(M_in, transpose=False, expand=False, padding=True, partial=False, output=None):
    """Fast Radon Transform (FRT) of the input matrix M_in (must be 2D numpy array)
    Additional arguments:
     -transpose (False): transpose M_in (replace x with y) to check all the other angles.
     -expand (False): adds zero padding to the sides of the passive axis to allow for corner-crossing streaks
     -padding (True): adds zero padding to the active axis to fill up powers of 2.
     -partial (False): use this to save second output, a list of Radon partial images (useful for calculating variance at different length scales)
     -output (None): give the a pointer to an array with the right size, for FRT to put the return value into it.
                     Note that if partial=True then output must be a list of arrays with the right dimensions.
    """
    #    print("running FRT with: transpose= "+str(transpose)+", expand= "+str(expand)+", padding= "+str(padding)+", partial= "+str(partial)+", finder= "+str(finder))

    ############### CHECK INPUTS AND DEFAULTS #################################

    if empty(M_in):
        return

    if M_in.ndim > 2:
        raise Exception("FRT cannot handle more dimensions than 2D")

    ############## PREPARE THE MATRIX #########################################

    M = np.array(M_in)  # keep a copy of M_in to give to finalizeFRT

    np.nan_to_num(M, copy=False)  # get rid of NaNs (replace with zeros)

    if transpose:
        M = M.T

    if padding:
        M = padMatrix(M)

    if expand:
        M = expandMatrix(M)

    ############## PREPARE THE MATRIX #########################################

    Nfolds = getNumLogFoldings(M)
    (Nrows, Ncols) = M.shape

    M_out = []

    if not empty(output):  # will return the entire partial transform list
        if partial:
            for m in range(2, Nfolds + 1):
                if output[m - 1].shape != getPartialDims(M, m):
                    raise RuntimeError(
                        "Wrong dimensions of output array["
                        + str(m - 1)
                        + "]: "
                        + str(output[m - 1].shape)
                        + ", should be "
                        + str(getPartialDims(M, m))
                    )

            M_out = output

        else:
            if output.shape[0] != 2 * M.shape[0] - 1:
                raise RuntimeError(
                    "Y dimension of output ("
                    + str(output.shape[0])
                    + ") is inconsistent with (padded and doubled) input ("
                    + str(M.shape[0] * 2 - 1)
                    + ")"
                )
            if output.shape[1] != M.shape[1]:
                raise RuntimeError(
                    "X dimension of output ("
                    + str(output.shape[1])
                    + ") is inconsistent with (expanded?) input ("
                    + str(M.shape[1])
                    + ")"
                )

    dx = np.array([0], dtype="int64")

    M = M[np.newaxis, :, :]

    for m in range(1, Nfolds + 1):  # loop over logarithmic steps

        M_prev = M
        dx_prev = dx

        Nrows = M_prev.shape[1]

        max_dx = 2 ** (m) - 1
        dx = range(-max_dx, max_dx + 1)
        if partial and not empty(output):
            M = M_out[m - 1]  # we already have memory allocated for this result
        else:
            M = np.zeros(
                (len(dx), Nrows // 2, Ncols), dtype=M.dtype
            )  # make a new array each time

        counter = 0

        for i in range(Nrows // 2):  # loop over pairs of rows (number of rows in new M)

            for j in range(len(dx)):  # loop over different shifts

                # find the value and index of the previous shift
                dx_in_prev = int(float(dx[j]) / 2)
                j_in_prev = dx_in_prev + int(len(dx_prev) / 2)
                # print "dx[%d]= %d | dx_prev[%d]= %d | dx_in_prev= %d" % (j, dx[j], j_in_prev, dx_prev[j_in_prev], dx_in_prev)
                gap_x = dx[j] - dx_in_prev  # additional shift needed

                M1 = M_prev[j_in_prev, counter, :]
                M2 = M_prev[j_in_prev, counter + 1, :]

                M[j, i, :] = shift_add(M1, M2, -gap_x)

            counter += 2

        if partial and empty(
            output
        ):  # only append to the list if it hasn't been given from the start using "output"
            M_out.append(M)

    #     end of loop on m

    if (
        not partial
    ):  # we don't care about partial transforms, we were not given an array to fill
        #        M_out = np.transpose(M, (0,2,1))[:,:,0] # lose the empty dimension
        M_out = M[:, 0, :]  # lose the empty dim

        if not empty(output):  # do us a favor and also copy it into the array
            np.copyto(output, M_out)
            # this can be made more efficient if we use the "output" array as
            # target for assignment at the last iteration on m.
            # this will save an allocation and a copy of the array.
            # however, this is probably not very expensive and not worth
            # the added complexity of the code

    return M_out


############# end of FRT algorithm, helper functions: ########################


def padMatrix(M):
    N = M.shape[0]
    dN = int(2 ** math.ceil(math.log(N, 2)) - N)
    #    print "must add "+str(dN)+" lines..."
    M = np.vstack((M, np.zeros((dN, M.shape[1]), dtype=M.dtype)))
    return M


def expandMatrix(M):
    Z = np.zeros((M.shape[0], M.shape[0]), dtype=M.dtype)
    M = np.hstack((Z, M, Z))
    return M


def shift_add(M1, M2, gap):

    output = np.zeros_like(M2)

    if gap > 0:
        output[:gap] = M1[:gap]
        output[gap:] = M1[gap:] + M2[:-gap]
    elif gap < 0:
        output[gap:] = M1[gap:]
        output[:gap] = M1[:gap] + M2[-gap:]
    else:
        output = M1 + M2

    return output


def getPartialDims(M, log_level):
    x = M.shape[1]
    y = M.shape[0]

    y = int(y / 2**log_level)
    z = int(2 ** (log_level + 1) - 1)

    return (z, y, x)


def getNumLogFoldings(M):
    return int(np.ceil(np.log2(M.shape[0])))


def getEmptyPartialArrayList(M):
    return [
        np.zeros(getPartialDims(M, m), dtype=M.dtype)
        for m in range(1, int(getNumLogFoldings(M) + 1))
    ]


####################### MAIN ######################################

if __name__ == "__main__":

    import time
    import matplotlib.pyplot as plt

    print("this is a test for pyradon.frt")

    t = time.time()

    M = np.random.normal(0, 1, (2048, 2048)).astype("float32")

    print("random image of size " + str(M.shape) + " sent to FRT")

    use_partial = 1

    if use_partial:
        Rout = getEmptyPartialArrayList(M)
    else:
        Rout = np.zeros((M.shape[0] * 2 - 1, M.shape[1]), dtype=M.dtype)

    R = FRT(M, partial=use_partial, output=Rout)

    if use_partial:
        plt.imshow(R[-1][:, 0, :])
    else:
        plt.imshow(R)

    print("Elapsed time: " + str(time.time() - t))
