"""Downsample (then upsample) forcing, no noise."""

# IMPORTANT: This script has been verified against Sajidah's output.
import fractions

import cupy as cp
import fppanalysis.deconvolution_methods as dec
import numpy as np
import xarray as xr

from gpu_deconv import utils

# gamma_list = [0.9]
# iterlist = [int(1e2), int(1e3), int(1e4), int(1e5)]
iterlist = [int(1e2), int(1e3)]

seed_tw = 10
seed_a = 2

dt = 0.01

gamma_list = [0.1, 1]

total_samples = 1e6 + 1

savedata = utils.ASSETS / "no_noise_downsampled_forcing"
savedata.mkdir(parents=False, exist_ok=True)


class SampleStrategy:
    """Container class specifying different ways of down- and up-sampling an array."""

    def __init__(self, array: xr.DataArray) -> None:
        self.arr = array.copy()
        self._arr = array.copy()

    def check_da_dims(
        self, arr: xr.DataArray, new_dim: str, new_coord: float
    ) -> xr.DataArray:
        if new_dim not in self.arr.dims:
            self.arr = self.arr.expand_dims(dim={new_dim: [0]})
        return arr.expand_dims(dim={new_dim: [new_coord]})

    def keep_every_ratio(
        self, ratio: fractions.Fraction
    ) -> tuple[xr.DataArray, xr.DataArray]:
        """Keep only every `ratio`-th value in the forcing."""
        if ratio >= 1:
            raise ValueError
        _a = self._arr.data
        _t = self._arr.time.data
        # First, reshape so that the length of a row is `ratio.denominator`
        rows = int(np.ceil(len(_a) / ratio.denominator))
        reshaped = np.pad(
            _a.astype(float),
            (0, ratio.denominator * rows - _a.size),
            mode="constant",
            constant_values=np.nan,
        ).reshape(rows, ratio.denominator)
        reshaped_time = np.pad(
            _t.astype(float),
            (0, ratio.denominator * rows - _t.size),
            mode="constant",
            constant_values=np.nan,
        ).reshape(rows, ratio.denominator)
        frc = reshaped[:, : ratio.numerator].flatten()
        frc_time = reshaped_time[:, : ratio.numerator].flatten()
        if not len(frc) % 2:
            frc = frc[:-1]
            frc_time = frc_time[:-1]
        utils.verify_odd_length(frc)
        reshaped[:, ratio.numerator :] = 0
        full = reshaped.flatten()[: len(_a)]
        if full.shape != self._arr.shape:
            raise utils.UnequalArrayLengthError
        down = utils.Wardrobe.dress_a_downsampled(
            "Forcing",
            frc,
            frc_time,
            ratio,
            desc='Raw "choose every n" downsampling',
        )
        da = utils.Wardrobe.dress_an_upsampled(
            "Forcing",
            full,
            _t,
            ratio,
            desc="Forcing with the same length, but downsampled to keep every `n` sample.",
        )
        new_dim = "sample_ratio"
        da = self.check_da_dims(da, new_dim, float(ratio))
        self.arr = xr.concat((self.arr, da), dim=new_dim)
        return down, self.arr

    def down_keep_every_ratio_cumsum(
        self, ratio: fractions.Fraction = fractions.Fraction(1, 10)
    ) -> xr.DataArray:
        """Keep only every `numerator/denominator` value in the forcing without loss."""
        if ratio >= 1:
            raise ValueError
        _a = np.cumsum(self._arr.data)
        _t = self._arr.time.data
        # First, reshape so that the length of a row is `ratio.denominator`
        rows = int(np.ceil(len(_a) / ratio.denominator))
        reshaped = np.pad(
            _a.astype(float),
            (0, ratio.denominator * rows - _a.size),
            mode="constant",
            constant_values=np.nan,
        ).reshape(rows, ratio.denominator)
        reduced_frc = reshaped[:, : ratio.numerator].flatten()
        # Calculate the number of original samples per new sample
        keep_every = int(ratio.denominator / ratio.numerator)
        frc = _a[::keep_every]
        frc = np.diff(np.concatenate(([0], frc)))
        if not len(frc) % 2:
            frc = frc[:-1]
        utils.verify_odd_length(frc)
        return utils.Wardrobe.dress_a_downsampled(
            "Forcing", frc, _t, ratio, desc="Cumsum downsampling"
        )


for gamma in gamma_list:
    print(f"generate signal for {gamma = }")

    total_pulses = int(total_samples * gamma * dt)
    ds = utils.new_time_series(total_pulses=total_pulses, gamma=gamma, dt=dt)
    time_array = ds["time"]
    signal = ds["signal"]
    forcing_original = ds["forcing"]

    res, err = dec.RL_gauss_deconvolve(
        signal=cp.asarray(signal),
        kern=cp.asarray(forcing_original),
        iteration_list=iterlist,
        gpu=True,
    )
    res_da = utils.Wardrobe.dress_response_pulse(res.get(), time_array.data, iterlist)
    res_da = res_da.expand_dims(dim={"sample_ratio": [0]})
    err_da = utils.Wardrobe.dress_response_pulse_error(err.get())
    err_da = err_da.expand_dims(dim={"sample_ratio": [0]})
    sampler = SampleStrategy(forcing_original)
    strategy = ["", "_cumsum"]
    ratios = ["99/100", "49/50", "9/10", "1/2"]
    for ratio in ratios:
        ratio_ = fractions.Fraction(ratio)
        fd_name = "forcing_downsampled"
        fd_name = f"{fd_name}_{ratio_.numerator}_{ratio_.denominator}"
        fd_, fdu_ = sampler.keep_every_ratio(ratio=ratio_)
        # forcing_original.plot()
        # fd_.plot()
        # fdu_.sel(sample_ratio=ratio_.denominator).plot()
        # plt.show()
        ds[fd_name] = fd_
        # fdu_ = sampler.up_zero_pad(fd_, ratio=ratio)
        fdu = cp.asarray(fdu_.sel(sample_ratio=float(ratio_)))
        signal = cp.asarray(signal)
        res, err = dec.RL_gauss_deconvolve(
            signal=signal,
            kern=fdu,
            iteration_list=iterlist,
            gpu=True,
        )
        res_ = utils.Wardrobe.dress_response_pulse(res.get(), time_array.data, iterlist)
        res_ = res_.expand_dims(dim={"sample_ratio": [ratio]})
        res_da = xr.concat((res_da, res_), "sample_ratio")
        err_ = utils.Wardrobe.dress_response_pulse_error(err.get())
        err_ = err_.expand_dims(dim={"sample_ratio": [ratio]})
        err_da = xr.concat((err_da, err_), "sample_ratio")
    ds = ds.assign(forcing=fdu_)
    ds = ds.assign(response_pulse=res_da)
    ds = ds.assign(response_pulse_err=err_da)

    print(f"deconv done for gamma={gamma}")

    fname = f"gamma_{gamma}"

    ds.to_netcdf(f"{savedata / fname}.nc", format="NETCDF4")
