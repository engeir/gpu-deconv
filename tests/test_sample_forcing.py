"""Module for testing the sampling strategies for the forcing."""

import fractions

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import gpu_deconv.utils


def view() -> None:
    """Plot a simple view of sampling strategies."""
    random_state = np.random.default_rng(125)
    timer = np.arange(1e3)
    amps = random_state.exponential(size=100)
    arrivals = random_state.uniform(low=0, high=1e3, size=100).astype(int)
    forcing = np.zeros_like(timer)
    assert len(np.unique(arrivals)) == len(arrivals)  # noqa: S101
    for i, idx in enumerate(arrivals):
        forcing[idx] = amps[i]
    # forcing = np.ones_like(forcing)
    # forcing[::4] = 0
    forcing_array = xr.DataArray(forcing, coords=[("time", timer)])
    sampler = gpu_deconv.utils.SampleStrategyForcing(forcing_array)
    # ratio = fractions.Fraction(3, 5)  # doesn't work with the original cumsum method
    ratio = fractions.Fraction(1, 6)
    down_blind, up_blind = sampler.sample_lossy_nth(ratio)
    down_index = 1
    down_cumsum_start, up_cumsum_start = sampler.sample_lossless_nth(
        ratio, down_index=down_index, up_index=0
    )
    down_cumsum_1, up_cumsum_1 = sampler.sample_lossless_nth(
        ratio, down_index=down_index, up_index=1
    )
    down_cumsum_end, up_cumsum_end = sampler.sample_lossless_nth(
        ratio, down_index=down_index, up_index=-1
    )
    down_repeat, up_repeat = sampler.sample_lossless_repeat(ratio)
    down_repeat2, up_repeat2 = sampler.sample_lossless_repeat(ratio, down_index=2)
    down_repeat4, up_repeat4 = sampler.sample_lossless_repeat(ratio, down_index=4)
    down_fourier, up_fourier = sampler.fourier_transform(ratio)
    down_lowpass, up_lowpass = sampler.lowpass(ratio)
    # fmt: off
    plt.plot(forcing_array.time.data, forcing_array.data, label="Orig")
    # plt.plot(getattr(down_blind, f"time_sparse_{ratio.numerator}_{ratio.denominator}").data, down_blind.data, label="Blind down")
    # plt.plot(up_blind.time.data, up_blind.isel(sample_ratio=-1).data, label="Blind up")

    plt.plot(getattr(down_cumsum_end, f"time_sparse_{ratio.numerator}_{ratio.denominator}").data, down_cumsum_end.data, c="r", label="cumsum_end down", alpha=0.3)
    plt.plot(up_cumsum_end.time.data, up_cumsum_end.isel(sample_ratio=-1).data, c="r", label="cumsum_end up")
    plt.plot(getattr(down_cumsum_start, f"time_sparse_{ratio.numerator}_{ratio.denominator}").data, down_cumsum_start.data, c="b", label="cumsum_start down", alpha=0.3)
    plt.plot(up_cumsum_start.time.data, up_cumsum_start.isel(sample_ratio=-1).data, c="b", label="cumsum_start up")
    plt.plot(getattr(down_cumsum_1, f"time_sparse_{ratio.numerator}_{ratio.denominator}").data, down_cumsum_1.data, c="g", label="cumsum_1 down", alpha=0.3)
    plt.plot(up_cumsum_1.time.data, up_cumsum_1.isel(sample_ratio=-1).data, c="g", label="cumsum_1 up")
    plt.plot(getattr(down_repeat, f"time_sparse_{ratio.numerator}_{ratio.denominator}").data, down_repeat.data, c="m", label="cumsum_repeat down", alpha=0.3)
    plt.plot(up_repeat.time.data, up_repeat.isel(sample_ratio=-1).data, c="m", label="cumsum_repeat up")
    plt.plot(getattr(down_repeat2, f"time_sparse_{ratio.numerator}_{ratio.denominator}").data, down_repeat2.data, c="k", label="cumsum_repeat2 down", alpha=0.3)
    plt.plot(up_repeat2.time.data, up_repeat2.isel(sample_ratio=-1).data, c="k", label="cumsum_repeat2 up")
    plt.plot(getattr(down_repeat4, f"time_sparse_{ratio.numerator}_{ratio.denominator}").data, down_repeat4.data, c="c", label="cumsum_repeat4 down", alpha=0.3)
    plt.plot(up_repeat4.time.data, up_repeat4.isel(sample_ratio=-1).data, c="c", label="cumsum_repeat4 up")

    # My Fourier methods are crap

    # plt.plot(getattr(down_fourier, f"time_sparse_{ratio.numerator}_{ratio.denominator}").data, down_fourier.data, ls="--" ,label="fourier down")
    # plt.plot(up_fourier.time.data, up_fourier.isel(sample_ratio=-1).data, label="fourier up")
    # plt.plot(getattr(down_lowpass, f"time_sparse_{ratio.numerator}_{ratio.denominator}").data, down_lowpass.data, ls=":", label="lowpass down")
    # plt.plot(up_lowpass.time.data, up_lowpass.isel(sample_ratio=-1).data, label="lowpass up")

    # fmt: on
    plt.legend()
    plt.show()


if __name__ == "__main__":
    view()
