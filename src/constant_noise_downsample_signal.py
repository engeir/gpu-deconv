import pathlib

import cupy as cp
import fppanalysis.deconvolution_methods as dec
import numpy as np
import superposedpulses.point_model as pm

gamma_list = [0.9]
eps_list = [0.01, 0.1, 1]
iterlist = [int(1e2), int(1e3), int(1e4), int(1e5)]

seedTW = 10
seedA = 2

dt_signal = 0.01
dt_forcing = 0.01

downsample_factor = 10

K = 71
gamma = 0.9
N = int(K / (gamma * dt_signal))


savedata = pathlib.Path().cwd() / "assets" / "constant_noise_downsample_signal"
savedata.mkdir(parents=False, exist_ok=True)

for eps in eps_list:
    for gamma in gamma_list:
        model = pm.PointModel(
            waiting_time=1 / gamma,
            total_duration=K / gamma,
            dt=dt_signal,
        )
        T, S = model.make_realization()
        forcing = model.get_last_used_forcing()
        A = forcing.amplitudes
        ta = forcing.arrival_times
        model.add_noise(eps, noise_type="additive")
        _, S_a = model.make_realization()
        model.add_noise(eps, noise_type="dynamic")
        _, S_d = model.make_realization()

        if len(S) % 2 == 0:
            S = S[:-1]
            T = T[:-1]
            print("S and T is now odd")

        if len(A) % 2 == 0:
            A = A[:-1]
            ta = ta[:-1]
            print("A and ta data is now odd")

        # Here, the forcing has a higher sampling than the signal.
        A = np.array(A)
        ta = np.array(ta)
        ta_index = np.ceil(ta / dt_signal).astype(int)
        forcing_original = np.zeros(T.size)
        for i in range(ta_index.size):
            forcing_original[ta_index[i]] += A[i]

        # Downsample noise data
        S_add_downsampled = S_a[::downsample_factor]
        S_dyn_downsampled = S_d[::downsample_factor]

        # Apply constant method - repeat values
        # np.repeat does this: To upsample back to the original length using constant values, you use np.repeat to duplicate
        # each point in the downsampled series 10 times. This method assumes that the value of each point remains constant
        # until the next point in the downsampled series.

        upsampled_S_add_downsampled = np.repeat(S_add_downsampled, downsample_factor)[
            : len(T)
        ]
        upsampled_S_dyn_downsampled = np.repeat(S_add_downsampled, downsample_factor)[
            : len(T)
        ]

        if len(S_add_downsampled) % 2 == 0:
            upsampled_S_add_downsampled = upsampled_S_add_downsampled[:-1]
            upsampled_S_dyn_downsampled = upsampled_S_dyn_downsampled[:-1]
            print("noise data now odd")

        upsampled_S_add_downsampled = cp.array(upsampled_S_add_downsampled)
        upsampled_S_dyn_downsampled = cp.array(upsampled_S_dyn_downsampled)
        forcing_original = cp.array(forcing_original)

        res_a, err_a = dec.RL_gauss_deconvolve(
            signal=upsampled_S_add_downsampled,
            kern=forcing_original,
            iteration_list=iterlist,
            gpu=True,
        )
        res_d, err_d = dec.RL_gauss_deconvolve(
            signal=upsampled_S_dyn_downsampled,
            kern=forcing_original,
            iteration_list=iterlist,
            gpu=True,
        )

        print(f"deconv done for eps={eps}")

        fname = f"eps_{eps}.npz"

        np.savez(
            savedata / fname,
            time=T,
            signal_dyn=S_d,
            signal_add=S_a,
            signal_add_downsampled=S_add_downsampled,
            signal_dyn_downsampled=S_dyn_downsampled,
            upsampled_signal_add=upsampled_S_add_downsampled,
            upsampled_signal_dyn=upsampled_S_dyn_downsampled,
            amplitude=A,
            arrival_time=ta,
            forcing_original=forcing_original,
            result_add=res_a,
            error_add=err_a,
            result_dyn=res_d,
            error_dyn=err_d,
        )
