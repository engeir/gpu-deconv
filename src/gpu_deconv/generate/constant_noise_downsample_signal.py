"""Downsample (then upsample) the signal, with constant noise.

The upsampling is done by repetition, downsampled data is by choosing every `n`.
"""

# IMPORTANT: This script has been verified against Sajidah's output.

import fractions

import cupy as cp
import fppanalysis.deconvolution_methods as dec
import numpy as np
import xarray as xr

from gpu_deconv import utils

gamma_list = [0.9]
eps_list = [0.01, 0.1, 1]
iterlist = [int(1e2), int(1e3), int(1e4), int(1e5)]

seed_tw = 10
seed_a = 2

dt_signal = 0.01
dt_forcing = 0.01

downsample_factor = 10

total_pulses = 71
gamma = 0.9


savedata = utils.ASSETS / "constant_noise_downsampled_signal"
savedata.mkdir(parents=False, exist_ok=True)

for eps in eps_list:
    for gamma in gamma_list:
        ds: xr.Dataset = utils.new_time_series(
            total_pulses,
            gamma,
            dt_signal,
            epsilon_additive=eps,
            epsilon_dynamic=eps,
        )
        signal = ds["signal"]
        time_array = ds["signal"].time
        signal_additive = ds["signal_additive"]
        signal_dynamic = ds["signal_dynamic"]
        amplitude = np.array(ds["events"].data)
        arrival_times = np.array(ds["events"].arrival_time.data)
        forcing_original = ds["forcing"]

        # Downsample noise data
        signal_add_downsampled = signal_additive[::downsample_factor]
        signal_dyn_downsampled = signal_dynamic[::downsample_factor]
        ds["signal_additive_downsampled"] = utils.Wardrobe.dress_a_downsampled(
            "Signal",
            signal_add_downsampled.data,
            time_array.data[::downsample_factor],
            ratio=fractions.Fraction(1, downsample_factor),
        )
        ds["signal_dynamic_downsampled"] = utils.Wardrobe.dress_a_downsampled(
            "Signal",
            signal_dyn_downsampled.data,
            time_array.data[::downsample_factor],
            ratio=fractions.Fraction(1, downsample_factor),
        )

        # Apply constant method - repeat values
        # np.repeat does this: To upsample back to the original length using constant values, you use np.repeat to duplicate
        # each point in the downsampled series 10 times. This method assumes that the value of each point remains constant
        # until the next point in the downsampled series.
        signal_add_downsampled_upsampled = np.repeat(
            signal_add_downsampled, downsample_factor
        )[: len(time_array)]
        signal_dyn_downsampled_upsampled = np.repeat(
            signal_dyn_downsampled, downsample_factor
        )[: len(time_array)]

        utils.verify_odd_length(signal_add_downsampled_upsampled)
        utils.verify_odd_length(signal_dyn_downsampled_upsampled)
        ds["signal_additive_downsampled_upsampled"] = utils.Wardrobe.dress_an_upsampled(
            "Signal",
            signal_add_downsampled_upsampled,
            time_array,
            ratio=fractions.Fraction(1, downsample_factor),
        )
        ds["signal_dynamic_downsampled_upsampled"] = utils.Wardrobe.dress_an_upsampled(
            "Signal",
            signal_dyn_downsampled_upsampled,
            time_array,
            ratio=fractions.Fraction(1, downsample_factor),
        )

        signal_add_downsampled_upsampled = cp.array(signal_add_downsampled_upsampled)
        signal_dyn_downsampled_upsampled = cp.array(signal_dyn_downsampled_upsampled)
        forcing_original = cp.array(forcing_original)

        res_a, err_a = dec.RL_gauss_deconvolve(
            signal=signal_add_downsampled_upsampled,
            kern=forcing_original,
            iteration_list=iterlist,
            gpu=True,
        )
        res_d, err_d = dec.RL_gauss_deconvolve(
            signal=signal_dyn_downsampled_upsampled,
            kern=forcing_original,
            iteration_list=iterlist,
            gpu=True,
        )
        mid = len(time_array.data) // 2
        tau = time_array.data - time_array.data[mid]
        ds["response_pulse_additive"] = utils.Wardrobe.dress_response_pulse(
            res_a.get(), tau, iterlist
        )
        ds["response_pulse_dynaimc"] = utils.Wardrobe.dress_response_pulse(
            res_d.get(), tau, iterlist
        )
        ds["response_pulse_additive_err"] = utils.Wardrobe.dress_response_pulse_error(
            err_a.get()
        )
        ds["response_pulse_dynamic_err"] = utils.Wardrobe.dress_response_pulse_error(
            err_d.get()
        )

        print(f"deconv done for eps={eps}")
        fname = f"eps_{eps}"
        ds.to_netcdf(f"{savedata / fname}.nc", format="NETCDF4")
