"""Downsample (then upsample) forcing, no noise."""

# IMPORTANT: This script has been verified against Sajidah's output.
import fractions

import cupy as cp
import fppanalysis.deconvolution_methods as dec
import numpy as np
import xarray as xr
from icecream import ic

from gpu_deconv import utils

# gamma_list = [0.9]
# iterlist = [int(1e2), int(1e3), int(1e4), int(1e5)]
iterlist = [int(1e2), int(1e3)]

seed = 125
seed_tw = 10
seed_a = 2

dt = 0.1
gamma_list = [0.01, 0.1]
total_samples = 1e4 + 1

savedata = utils.ASSETS / "no_noise_downsampled_forcing"
savedata.mkdir(parents=False, exist_ok=True)


def main() -> None:
    """Run the main program."""
    for strategy in ["lossy_nth", "lossy_repeat", "lossless_nth", "lossless_repeat"]:
        for gamma in gamma_list:
            print(f"Generate signal for {gamma = }")

            total_pulses = int(total_samples * gamma * dt)
            ic(total_pulses)
            # ds = utils.new_time_series(total_pulses=total_pulses, gamma=gamma, dt=dt)
            ds = utils.TimeSeriesModel(seed)(
                total_pulses=total_pulses, gamma=gamma, dt=dt
            )
            time_array = ds["time"]
            signal = ds["signal"]
            forcing_original = ds["forcing"]
            pulse = ds["pulse_shape"]

            # initial_guess = np.heaviside(pulse.tau.data, 1)
            initial_guess = np.ones_like(pulse.tau.data)
            res, err = dec.RL_gauss_deconvolve(
                signal=cp.asarray(signal),
                kern=cp.asarray(forcing_original),
                iteration_list=iterlist,
                initial_guess=cp.asarray(initial_guess),
                gpu=True,
            )
            res_da = utils.Wardrobe.dress_response_pulse(
                res.get(), time_array.data, iterlist
            )
            res_da = res_da.expand_dims(dim={"sample_ratio": [0]})
            err_da = utils.Wardrobe.dress_response_pulse_error(err.get())
            err_da = err_da.expand_dims(dim={"sample_ratio": [0]})
            sampler = utils.SampleStrategyForcing(forcing_original)
            sample_func = getattr(sampler, f"sample_{strategy}")
            ratios = ["1/2", "1/5", "1/10", "1/24"]
            for ratio in ratios:
                ratio_ = fractions.Fraction(ratio)
                fd_name = "forcing_downsampled"
                fd_name = f"{fd_name}_{ratio_.numerator}_{ratio_.denominator}"
                fd_, fdu_ = sample_func(ratio=ratio_)
                ds[fd_name] = fd_
                fdu = cp.asarray(fdu_.sel(sample_ratio=float(ratio_)))
                signal = cp.asarray(signal)
                res, err = dec.RL_gauss_deconvolve(
                    signal=signal,
                    kern=fdu,
                    iteration_list=iterlist,
                    initial_guess=cp.asarray(initial_guess),
                    gpu=True,
                )
                res_ = utils.Wardrobe.dress_response_pulse(
                    res.get(), time_array.data, iterlist
                )
                res_ = res_.expand_dims(dim={"sample_ratio": [float(ratio_)]})
                res_da = xr.concat((res_da, res_), dim="sample_ratio")
                err_ = utils.Wardrobe.dress_response_pulse_error(err.get())
                err_ = err_.expand_dims(dim={"sample_ratio": [float(ratio_)]})
                err_da = xr.concat((err_da, err_), dim="sample_ratio")
            ds = ds.assign(forcing=fdu_)
            ds = ds.assign(response_pulse=res_da)
            ds = ds.assign(response_pulse_err=err_da)
            ds = ds.assign_attrs(sampling_strategy=strategy)

            print(f"Deconv done for {gamma = }")

            fname = f"{strategy}_gamma_{gamma}"

            ds.to_netcdf(f"{savedata / fname}.nc", format="NETCDF4")


if __name__ == "__main__":
    main()
