import pathlib

import cupy as cp
import fppanalysis.deconvolution_methods as dec
import numpy as np
import superposedpulses.point_model as pm

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
seedTW = 10
seedA = 2

# Number of events.
K = 71

# Sampling time(s)
sampling_time_signal = 0.01

downsample_factor = 10

savedata = pathlib.Path().cwd() / "assets" / "fourier_noise_downsampled_signal"
savedata.mkdir(parents=False, exist_ok=True)
print(savedata.resolve())
exit()


# A function to adjust the lengths of the signal and time array so they
# are the same length and have an odd number of data points.
def adjust_lengths(S, T):
    """
    This function ensures that the signal and time array are the same lengths and
    have an odd number of data points for the deconvolutions.
    S: the signal array (it can have noise or whatever).
    T: the time array.
    """
    # Check if S has an odd number of elements
    if len(S) % 2 != 0:
        print("The signal has an odd number of data points.")
    else:
        S = S[:-1]  # Remove the last element
        print("The signal has an even number of data points.")

    # Ensure S and T have the same length
    if len(S) != len(T):
        print("The signal and time array are not the same length. Adjusting...")
        # Truncate the longer array or extend the shorter array
        min_length = min(len(S), len(T))
        S = S[:min_length]
        T = T[:min_length]
        print("The signal and time array are adjusted to the same length.")
    else:
        print("The signal and time array are already the same length.")

    return S, T


# Investigating different noise levels and gamma on upsampled downsampled signals
for epsilon in epsilon_list:
    for gamma in gamma_list:
        # S, S_d, and S_a and the signals without noise, with dynamic noise and additive noise.
        model = pm.PointModel(
            waiting_time=1 / gamma,
            total_duration=K / gamma,
            dt=sampling_time_signal,
        )
        time_array, S = model.make_realization()
        forcing = model.get_last_used_forcing()
        amplitude = forcing.amplitudes
        arrival_time = forcing.arrival_times
        model.add_noise(epsilon, noise_type="additive")
        _, S_a = model.make_realization()
        model.add_noise(epsilon, noise_type="dynamic")
        _, S_d = model.make_realization()

        # Here, the forcing will have the same sampling time as the pre-downsampled signal.
        # I have simply called it 'forcing' because we're not downsampling it later.
        amplitude = np.array(amplitude)
        arrival_time = np.array(arrival_time)
        arrival_time_index = np.ceil(arrival_time / sampling_time_signal).astype(int)
        forcing = np.zeros(time_array.size)
        for i in range(arrival_time_index.size):
            forcing[arrival_time_index[i]] += amplitude[i]

        # Downsample noise data
        S_add_downsampled = S_a[::downsample_factor]
        S_dyn_downsampled = S_d[::downsample_factor]

        # Apply Fourier method
        # np.fft.rfft computes the one-dimensional discrete Fourier Transform for real input.
        S_add_fft_coeffs = np.fft.rfft(S_add_downsampled)
        S_dyn_fft_coeffs = np.fft.rfft(S_dyn_downsampled)

        # np.fft.irfft computes the inverse of the one-dimensional discrete
        # Fourier Transform for real input, as computed by rfft.
        # Ensures same length as the time array that hasn't been altered.
        upsampled_S_add_downsampled = np.fft.irfft(S_add_fft_coeffs, n=len(time_array))
        upsampled_S_dyn_downsampled = np.fft.irfft(S_dyn_fft_coeffs, n=len(time_array))

        # Check if the upsampled signal has an even number of data points.
        # Also make sure that the time array still matches the length of the
        # upsampled downsamped signal after doing the following conditional.
        # This was done by using the function adjust_lengths
        upsampled_S_add_downsampled, time_array = adjust_lengths(
            upsampled_S_add_downsampled, time_array
        )
        upsampled_S_dyn_downsampled, _ = adjust_lengths(
            upsampled_S_dyn_downsampled, time_array
        )

        # The forcing to be made needs to have an odd number of data points so check this.
        if len(forcing) % 2 == 0:
            forcing = forcing[:-1]
            print("forcing array has an odd number of data points")

        # Convert from numpy to cupy arrays if using the GPU.
        # The updated deconvolution function on fpp_analysis should be able to do this already.
        # So you may not need the next three lines.
        upsampled_S_add_downsampled = cp.array(upsampled_S_add_downsampled)
        upsampled_S_dyn_downsampled = cp.array(upsampled_S_dyn_downsampled)
        forcing = cp.array(forcing)

        # Old version of deconvolution function used.
        # Estimating the pulse shape here using the upsampled downsampled signal
        # and the original forcing.
        res_a, err_a = dec.RL_gauss_deconvolve(
            signal=upsampled_S_add_downsampled,
            kern=forcing,
            iteration_list=iterlist,
            gpu=True,
        )
        res_d, err_d = dec.RL_gauss_deconvolve(
            signal=upsampled_S_dyn_downsampled,
            kern=forcing,
            iteration_list=iterlist,
            gpu=True,
        )

        fname = f"epsilon_{epsilon}.npz"

        np.savez(
            savedata / fname,
            time=time_array,
            signal_dyn=S_d,
            signal_add=S_a,
            signal_add_downsampled=S_add_downsampled,
            signal_dyn_downsampled=S_dyn_downsampled,
            upsampled_signal_add=upsampled_S_add_downsampled,
            upsampled_signal_dyn=upsampled_S_dyn_downsampled,
            amplitude=amplitude,
            arrival_time=arrival_time,
            forcing=forcing,
            result_add=res_a,
            error_add=err_a,
            result_dyn=res_d,
            error_dyn=err_d,
        )

        print(f"Deconvolution done for epsilon = {epsilon} for gamma = {gamma}")
