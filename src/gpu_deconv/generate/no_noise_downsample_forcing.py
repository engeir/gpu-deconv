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


class SampleStrategyForcing:
    """Container class specifying different ways of down- and up-sampling an array."""

    def __init__(self, array: xr.DataArray) -> None:
        self.arr = array.copy()
        self._arr = array.copy()

    def _check_da_dims(
        self, arr: xr.DataArray, new_dim: str, new_coord: float
    ) -> xr.DataArray:
        if new_dim not in self.arr.dims:
            self.arr = self.arr.expand_dims(dim={new_dim: [0]})
        return arr.expand_dims(dim={new_dim: [new_coord]})

    def _create_corresponding_upsampled(
        self, arr: np.ndarray, times: np.ndarray, ratio: fractions.Fraction
    ) -> None:
        da = utils.Wardrobe.dress_an_upsampled(
            "Forcing",
            arr,
            times,
            ratio,
            desc="Forcing with the same length, but downsampled to keep every `n` sample.",
        )
        new_dim = "sample_ratio"
        da = self._check_da_dims(da, new_dim, float(ratio))
        self.arr = xr.concat((self.arr, da), dim=new_dim)

    @staticmethod
    def _reshape(arr: np.ndarray, ratio: fractions.Fraction) -> np.ndarray:
        rows = int(np.ceil(len(arr) / ratio.denominator))
        return np.pad(
            arr.astype(float),
            (0, ratio.denominator * rows - arr.size),
            mode="constant",
            constant_values=np.nan,
        ).reshape(rows, ratio.denominator)

    def _sample_at_ratio(
        self, orig_arr: np.ndarray, orig_time: np.ndarray, ratio: fractions.Fraction
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample an array at a given ratio.

        Parameters
        ----------
        orig_arr : np.ndarray
            The input array that should be downsampled.
        orig_time : np.ndarray
            The time axis of the input array that should be downsampled.
        ratio : fractions.Fraction
            The ratio of the downsampling to original array.

        Returns
        -------
        np.ndarray
            The downsampled input array.
        np.ndarray
            The time axis of the downsampled input array.
        np.ndarray
            The upsampling of the input array with equal size to the input array.
        """
        # Re-shape the array with column lengths equal to `denominator`, and pad with
        # NaN if it does not fit.
        # Keep only the first `numerator` number of columns in each row.
        reshaped = self._reshape(orig_arr, ratio)
        frc = reshaped[:, : ratio.numerator].flatten()
        frc_time = self._reshape(orig_time, ratio)[:, : ratio.numerator].flatten()
        if not len(frc) % 2:
            frc = frc[:-1]
            frc_time = frc_time[:-1]
        utils.verify_odd_length(frc)
        return frc, frc_time, reshaped

    def keep_every_ratio_blind(
        self, ratio: fractions.Fraction
    ) -> tuple[xr.DataArray, xr.DataArray]:
        """Keep only every `ratio`-th value in the forcing."""
        if ratio >= 1:
            raise ValueError
        _a = self._arr.data
        _t = self._arr.time.data
        # First, reshape so that the length of a row is `ratio.denominator`
        frc, frc_time, reshaped = self._sample_at_ratio(_a, _t, ratio)
        # The upsampled version is assumed to be filled with zeros where the
        # downsampling occurred.
        reshaped[:, ratio.numerator :] = 0
        full = reshaped.flatten()[: len(_a)]
        utils.verify_equal_length(full, self._arr)
        down = utils.Wardrobe.dress_a_downsampled(
            "Forcing",
            frc,
            frc_time,
            ratio,
            desc='Raw "choose every n" downsampling',
        )
        self._create_corresponding_upsampled(full, _t, ratio)
        return down, self.arr

    def keep_every_ratio_cumsum(
        self, ratio: fractions.Fraction = fractions.Fraction(1, 10)
    ) -> tuple[xr.DataArray, xr.DataArray]:
        """Keep only every `numerator/denominator` value in the forcing without loss."""
        if ratio >= 1:
            raise ValueError
        _a = np.cumsum(self._arr.data)
        _t = self._arr.time.data
        frc, frc_time, reshaped = self._sample_at_ratio(_a, _t, ratio)
        frc = np.diff(np.concatenate(([0], frc)))
        # The upsampled version is assumed to be filled with zeros where the
        # downsampling occurred.
        width = 2 * ratio.numerator - reshaped.shape[1]
        reshaped[:, ratio.numerator :] = reshaped[:, width : ratio.numerator]
        full = reshaped.flatten()[: len(_a)]
        full = np.diff(np.concatenate(([0], full)))
        utils.verify_equal_length(full, self._arr)
        down = utils.Wardrobe.dress_a_downsampled(
            "Forcing",
            frc,
            frc_time,
            ratio,
            desc='Raw "choose every n" downsampling',
        )
        self._create_corresponding_upsampled(full, _t, ratio)
        return down, self.arr


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
    sampler = SampleStrategyForcing(forcing_original)
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
