import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer
import pytest

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import finder


def test_finder_data_variance():
    pars = finder.Finder.Pars()
    data = finder.Finder.Data()
    data._pars = pars

    # make sure the default variance is used
    assert data.variance == 1

    # after changing the parameters this should be reflected in the default value
    pars.default_var_scalar = 2.5
    assert data.variance == 2.5

    # assign a scalar value to the variance
    data.variance = 3.6
    assert data.variance == 3.6
    assert data.var_scalar == 3.6
    assert data._var_image is None

    # assign a var map now
    v_map = np.random.rand(256, 512)
    v_med = np.median(v_map)
    data.variance = v_map
    assert data.var_scalar == v_med
    assert np.array_equal(v_map, data.variance)
    assert np.array_equal(v_map, data._var_image)

    # now again assign a scalar variance
    data.variance = 4.5
    assert data.variance == 4.5
    assert data.var_scalar == 4.5
    assert data._var_image is None

    # reset the variance and make sure defaults are returned
    data.variance = None  # clear the variance inputs
    assert data.variance == 2.5  # the pars default
    assert data.var_scalar == 2.5
    assert data._var_scalar is None
    assert data._var_image is None
    # should still be able to produce a default var image
    assert data.get_var_image((0, 0), (512, 512))[0, 0] == 2.5


def test_finder_data_radon_var_maps():
    pass  # TODO: continue this!


def test_finder_data_psf():
    pars = finder.Finder.Pars()
    data = finder.Finder.Data()
    data._pars = pars

    # make sure the default PSF width is used
    assert data.psf_sigma == 1

    # assign a new default sigma
    pars.default_psf_sigma = 2.5
    assert data.psf_sigma == 2.5

    # assign a scalar psf sigma
    data.psf = 1.32
    assert data.psf_sigma == 1.32
    assert isinstance(data.psf, np.ndarray)
    assert abs(np.sum(data.psf**2) - 1) < 0.01

    # now feed back the same psf with only scaled up
    psf_map = data.psf
    psf_map *= 3
    psf_map = np.pad(psf_map, 10)
    data.psf = psf_map
    assert data.psf.shape == psf_map.shape
    # image of the PSF should still be normalized
    assert abs(np.sum(psf_map**2) - 1) < 0.01
    # should be able to recover the gaussian width
    assert abs(data.psf_sigma - 1.32) < 0.01

    # clearing the PSF should return things to defaults
    data.clear_psf()
    assert data.psf_sigma == 2.5
    # should generate a new PSF with sigma=2.5
    assert psf_map.shape != data.psf.shape


def test_find_single(f, sim):
    sim.x1 = 3.0 / 8.0
    sim.x2 = 0.5
    sim.y1 = 0.25
    sim.y2 = 0.50
    sim.make_image()
    f.data.image = sim.image
    f.data.psf = sim.psf_sigma
    f.data.variance = sim.bg_noise_var
    f.find_single(f.preprocess(sim.image))

    try:
        assert f.streaks
        assert (
            abs(f.streaks[0].snr - sim.calc_snr()) / sim.calc_snr() < 0.1
        )  # 10% differences
        assert (
            abs(f.streaks[0].I - sim.intensity) / sim.intensity < 0.1
        )  # 10% differences
        assert abs(f.streaks[0].th - sim.th) / sim.th < 0.1  # 10% differences
        assert (
            abs(f.streaks[0].x0 - sim.x0 * sim.im_size) / (sim.x0 * sim.im_size) < 0.1
        )  # 10% differences
        assert (
            abs(f.streaks[0].L - sim.L * sim.im_size) / (sim.L * sim.im_size) < 0.1
        )  # 10% differences
    except Exception as e:
        sim.print()
        f.streaks[0].print()
        f.streaks[0].show()
        plt.show(block=True)
        raise e
