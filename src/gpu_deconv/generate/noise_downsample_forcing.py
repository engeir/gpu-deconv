
import cupy as cp
import fppanalysis.deconvolution_methods as dec
import numpy as np
import superposedpulses.point_model as pm

from gpu_deconv import utils

gamma_list = [0.9]
eps_list = [0.01, 0.1, 0.5, 1, 1.2, 1.5]
iterlist = [int(1e2), int(1e3), int(1e4), int(1e5)]

seedTW = 10
seedA = 2

dt_signal = 0.01
dt_forcing = 0.1

K = 71
gamma = 0.9
N = int(K / (gamma * dt_signal))

# K = int(N*gamma*dt_signal)

savedata = utils.ASSETS / "noise_downsampled_forcing"
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

        if len(S_a) % 2 == 0:
            S_a = S_a[:-1]
            S_d = S_d[:-1]
            print("noise data now odd")

        # Calculate forcing with the same sampling as signal

        A = np.array(A)
        ta = np.array(ta)
        ta_index = np.ceil(ta / dt_signal).astype(int)
        forcing_original = np.zeros(T.size)
        for i in range(ta_index.size):
            forcing_original[ta_index[i]] += A[i]

        forcing_downsampled = np.zeros(T.size)

        # Calculate the number of original samples per new sample
        samples_per_new_interval = int(dt_forcing / dt_signal)

        # Perform the downsampling by copying values and zero-padding
        for i in range(0, T.size, samples_per_new_interval):
            if i < T.size:
                forcing_downsampled[i] = forcing_original[i]

        forcing_downsampled = cp.asarray(forcing_downsampled)

        S_a = cp.array(S_a)
        S_d = cp.array(S_d)

        res_a, err_a = dec.RL_gauss_deconvolve(
            signal=S_a, kern=forcing_downsampled, iteration_list=iterlist, gpu=True
        )
        res_d, err_d = dec.RL_gauss_deconvolve(
            signal=S_d, kern=forcing_downsampled, iteration_list=iterlist, gpu=True
        )

        print(f"deconv done for eps={eps}")

        fname = f"eps_{eps}"

        np.savez(
            savedata / fname,
            time=T,
            signal_dyn=S_d,
            signal_add=S_a,
            amplitude=A,
            arrival_time=ta,
            forcing_original=forcing_original,
            forcing_downsampled=forcing_downsampled,
            result_add=res_a,
            error_add=err_a,
            result_dyn=res_d,
            error_dyn=err_d,
        )
