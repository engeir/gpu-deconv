"""Downsample (then upsample) forcing, no noise."""

# IMPORTANT: This script has been verified against Sajidah's output.
import cupy as cp
import fppanalysis.deconvolution_methods as dec
import numpy as np
import xarray as xr

from gpu_deconv import utils

# gamma_list = [0.9]
iterlist = [int(1e2), int(1e3), int(1e4), int(1e5)]

seed_tw = 10
seed_a = 2

dt = 0.01
dt_forcing = 0.1

gamma_list = [0.1, 1, 10]

total_samples = 1e5

savedata = utils.ASSETS / "no_noise_downsampled_forcing"
savedata.mkdir(parents=False, exist_ok=True)


class SampleStrategy:
    """Container class specifying different ways of down- and up-sampling an array."""

    def __init__(self, array: xr.DataArray) -> None:
        self.arr = array

    def down_keep_every_ratio(
        self, numerator: int = 1, denominator: int = 10
    ) -> xr.DataArray:
        """Keep only every `numerator/denominator` value in the forcing."""
        if numerator != 1:
            raise NotADirectoryError
        _a = self.arr.data
        _t = self.arr.time.data
        # frc = np.zeros_like(_f)
        # Calculate the number of original samples per new sample
        keep_every = int(denominator / numerator)
        # # Perform the downsampling by copying values and zero-padding
        # for i in range(0, _t.size, keep_every):
        #     if i < _t.size:
        #         frc[i] = forcing_original[i]
        frc = _a[::keep_every]
        if not len(frc) % 2:
            frc = frc[:-1]
        utils.verify_odd_length(frc)
        return utils.Wardrobe.dress_a_downsampled("Forcing", frc, _t, denominator)

    def up_zero_pad(
        self, arr: xr.DataArray, before: int = 1, after: int = 10
    ) -> xr.DataArray:
        """Pad the array with zeros, such that for each `before` there are now `after` elements."""
        if before != 1:
            raise NotADirectoryError
        length_goal = len(self.arr)
        _a = arr.data
        reshape = len(_a) // before
        reshaped = _a[: int(reshape * before)].reshape(reshape, before)
        reshaped = np.pad(
            reshaped, ((0, 0), (0, after - before)), "constant", constant_values=0
        ).flatten()
        rest = _a[int(reshape * before) :]
        final = np.concatenate((reshaped, rest))
        if (diff := len(final)) < length_goal:
            final = np.pad(
                final, (0, length_goal - diff), "constant", constant_values=0
            )
        elif diff > length_goal:
            final = final[:length_goal]
        return utils.Wardrobe.dress_an_upsampled(
            "Forcing",
            final,
            self.arr.time,
            after,
            desc="Forcing with the same length, but downsampled to keep every `n` sample.",
        )


for gamma in gamma_list:
    total_pulses = int(total_samples * gamma * dt)

    print(f"generate signal for gamma={gamma}")

    ds = utils.new_time_series(total_pulses=total_pulses, gamma=gamma, dt=dt)
    time_array = ds["time"]
    signal = ds["signal"]
    forcing_original = ds["forcing"]
    amplitude = ds["events"]
    arrival_times = ds["events"].arrival_time

    sampler = SampleStrategy(forcing_original)
    forcing_downsampled = sampler.down_keep_every_ratio()
    ds["forcing_downsampled"] = forcing_downsampled
    ds["forcing_downsampled_upsampled"] = sampler.up_zero_pad(forcing_downsampled)
    forcing_downsampled = cp.asarray(forcing_downsampled)

    signal = cp.array(signal)

    tau = time_array.data
    res, err = dec.RL_gauss_deconvolve(
        signal=signal, kern=forcing_downsampled, iteration_list=iterlist, gpu=True
    )
    ds["response_pulse"] = utils.Wardrobe.dress_response_pulse(
        res.get(), time_array.data, iterlist
    )
    ds["response_pulse_err"] = utils.Wardrobe.dress_response_pulse_error(err.get())

    print(f"deconv done for gamma={gamma}")

    fname = f"gamma_{gamma}"

    ds.to_netcdf(f"{savedata / fname}.nc", format="NETCDF4")
