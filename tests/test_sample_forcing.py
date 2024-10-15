"""Module for testing the sampling strategies for the forcing."""

import fractions

import numpy as np
import xarray as xr

import gpu_deconv.utils


def test_cumsum() -> None:
    """Minimal test for the cumsum down- and upsampling strategy for forcings."""
    # fmt: off
    #              1  2  3  4  5  6  7  8  9 10  1  2  3  4  5  6  7  8  9 10  1  2  3  4  5  6  7  8  9 10  1  2  3  4  5  6  7  8  9 10  1  2  3  4  5  6
    input_array = [0, 0, 0, 4, 0, 0, 0, 0, 0, 2, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 3, 0, 0, 2, 0, 0, 0, 0, 5, 0, 0, 6, 0, 0, 0, 0, 0]
    a1 =          [0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 0, 3, 0, 2, 0, 0, 0, 0, 5, 0, 0, 6, 0, 0, 0, 0, 0]
    a2 =          [0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 0, 3, 0, 2, 0, 0, 0, 0, 5, 0, 0, 6, 0, 0, 0, 0, 0]
    a3 =          [0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 2, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 3, 0, 2, 0, 0, 0, 0, 0, 5, 0, 6, 0, 0, 0, 0, 0]
    a4 =          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0,11, 0, 0, 0, 0, 0, 0, 0, 0, 0,13, 0, 0, 0, 0, 0]
    # fmt: on
    timer = np.arange(len(input_array))
    arr = xr.DataArray(input_array, coords=[("time", timer)])
    sampler = gpu_deconv.utils.SampleStrategyForcing(arr)
    for n, m, a_ in [(9, 10, a1), (4, 5, a2), (1, 2, a3), (1, 10, a4)]:
        a = np.asarray(a_)
        _, out_ = sampler.keep_every_ratio_cumsum(fractions.Fraction(n, m))
        out = out_.isel(sample_ratio=-1)
        assert np.array_equal(out, a)  # noqa: S101
