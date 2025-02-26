"""Downsample (then upsample) the signal, with constant noise.

The upsampling is done in the Fourier domain, downsampled data is by choosing every `n`.
"""

import cupy as cp
import fppanalysis.deconvolution_methods as dec
import numpy as np
import xarray as xr

from gpu_deconv import utils

# Copy this script over to you directory.
# Make sure you have the relevant imports.

# Specifying the parameters I want to test out.
# Feel free to adjust as you like.
gamma_list = [0.09, 0.9]
epsilon_list = [0.01, 0.1, 0.5, 1]

# Iteration list for the deconvolution.
# You may not be interested in what the result looks like at
# certain points of the iteration, so this may not be necessary.
iterlist = [int(1e2), int(1e4)]

# This was needed to get the same time series for different noise levels.
seed_tw = 10
seed_a = 2

# Number of events.
total_pulses = 71

# Sampling time(s)
sampling_time_signal = 0.01

downsample_factor = 10

savedata = utils.ASSETS / "fourier_noise_downsampled_signal"
savedata.mkdir(parents=False, exist_ok=True)


# Investigating different noise levels and gamma on upsampled downsampled signals
for epsilon in epsilon_list:
    for gamma in gamma_list:
        ds: xr.Dataset = utils.new_time_series(
            total_pulses,
            gamma,
            sampling_time_signal,
            epsilon_additive=epsilon,
            epsilon_dynamic=epsilon,
        )
        time_array = ds["signal"].time
        signal_add = ds["signal_additive"]
        signal_dynamic = ds["signal_dynamic"]
        # Here, the forcing will have the same sampling time as the pre-downsampled signal.
        # I have simply called it 'forcing' because we're not downsampling it later.
        amplitude = np.array(ds["events"].data)
        arrival_times = np.array(ds["events"].arrival_time.data)
        forcing = ds["forcing"]

        # Downsample noise data
        signal_add_downsampled = signal_add[::downsample_factor]
        signal_dyn_downsampled = signal_dynamic[::downsample_factor]
        ds["signal_additive_downsampled"] = xr.DataArray(
            signal_add_downsampled,
            dims=["time_sparse"],
            coords={"time_sparse": time_array.data[::downsample_factor]},
        )
        ds["signal_dynamic_downsampled"] = xr.DataArray(
            signal_dyn_downsampled,
            dims=["time_sparse"],
            coords={
                "time_sparse": (
                    "time_sparse",
                    time_array.data[::downsample_factor],
                    {"long_name": "Time", "units": r"$\tau_d$"},
                )
            },
        )

        # Apply Fourier method
        # np.fft.rfft computes the one-dimensional discrete Fourier Transform for real input.
        signal_add_fft_coeffs = np.fft.rfft(signal_add_downsampled)
        signal_dyn_fft_coeffs = np.fft.rfft(signal_dyn_downsampled)

        # np.fft.irfft computes the inverse of the one-dimensional discrete
        # Fourier Transform for real input, as computed by rfft.
        # Ensures same length as the time array that hasn't been altered.
        signal_add_downsampled_upsampled = np.fft.irfft(
            signal_add_fft_coeffs, n=len(time_array)
        )
        signal_dyn_downsampled_upsampled = np.fft.irfft(
            signal_dyn_fft_coeffs, n=len(time_array)
        )

        # Check if the upsampled signal has an even number of data points.
        # Also make sure that the time array still matches the length of the
        # upsampled downsamped signal after doing the following conditional.
        # This was done by using the function adjust_lengths
        utils.verify_odd_length(time_array)
        utils.verify_odd_length(signal_add_downsampled_upsampled)
        utils.verify_odd_length(signal_dyn_downsampled_upsampled)
        utils.verify_odd_length(forcing)
        utils.verify_equal_length(
            time_array,
            signal_add_downsampled_upsampled,
            signal_dyn_downsampled_upsampled,
            forcing,
        )
        ds["signal_additive_downsampled_upsampled"] = xr.DataArray(
            signal_add_downsampled_upsampled,
            dims=["time"],
            coords={"time": time_array},
        )
        ds["signal_dynamic_downsampled_upsampled"] = xr.DataArray(
            signal_dyn_downsampled_upsampled,
            dims=["time"],
            coords={"time": time_array},
        )

        # Convert from numpy to cupy arrays if using the GPU.
        # The updated deconvolution function on fpp_analysis should be able to do this already.
        # So you may not need the next three lines.
        signal_add_downsampled_upsampled = cp.array(signal_add_downsampled_upsampled)
        signal_dyn_downsampled_upsampled = cp.array(signal_dyn_downsampled_upsampled)
        forcing = cp.array(forcing)

        res_a, err_a = dec.RL_gauss_deconvolve(
            signal=signal_add_downsampled_upsampled,
            kern=forcing,
            iteration_list=iterlist,
            gpu=True,
        )
        res_d, err_d = dec.RL_gauss_deconvolve(
            signal=signal_dyn_downsampled_upsampled,
            kern=forcing,
            iteration_list=iterlist,
            gpu=True,
        )
        ds["response_pulse_additive"] = xr.DataArray(
            res_a.get(),
            dims=["time", "iterlist"],
            coords={"time": time_array, "iterlist": iterlist},
            attrs={"description": "Result from deconvolution"},
        )
        ds["response_pulse_dynaimc"] = xr.DataArray(
            res_d.get(),
            dims=["time", "iterlist"],
            coords={"time": time_array, "iterlist": iterlist},
            attrs={"description": "Result from deconvolution"},
        )
        ds["response_pulse_additive_err"] = xr.DataArray(
            err_a.get(),
            dims=["iterations"],
            coords={"iterations": np.arange(len(err_a))},
            attrs={"description": "Errors after result from deconvolution"},
        )
        ds["response_pulse_dynaimc_err"] = xr.DataArray(
            err_d.get(),
            dims=["iterations"],
            coords={"iterations": np.arange(len(err_d))},
            attrs={"description": "Errors after result from deconvolution"},
        )

        fname = f"epsilon_{epsilon}"

        ds.to_netcdf(f"{savedata / fname}.nc", format="NETCDF4")

        print(f"Deconvolution done for epsilon = {epsilon} for gamma = {gamma}")
