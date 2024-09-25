import pathlib

import cupy as cp
import fppanalysis.deconvolution_methods as dec
import numpy as np
import superposedpulses.point_model as pm

# gamma_list = [0.9]
iterlist = [int(1e2), int(1e3), int(1e4), int(1e5)]

seedTW = 10
seedA = 2

dt_signal = 0.01
dt_forcing = 0.1


gamma_list = [0.1, 1, 10]

N = 1e5

savedata = pathlib.Path().cwd() / "assets" / "no_noise_downsampled_forcing"
savedata.mkdir(parents=False, exist_ok=True)

for gamma in gamma_list:
    K = int(N * gamma * dt_signal)

    print(f"generate signal for gamma={gamma}")

    model = pm.PointModel(
        waiting_time=1 / gamma,
        total_duration=K / gamma,
        dt=dt_signal,
    )
    T, S = model.make_realization()
    forcing = model.get_last_used_forcing()
    A = forcing.amplitudes
    ta = forcing.arrival_times
    _, S_a = model.make_realization()
    _, S_d = model.make_realization()

    if len(S) % 2 == 0:
        S = S[:-1]
        T = T[:-1]
        print("S and T is now odd")

    if len(A) % 2 == 0:
        A = A[:-1]
        ta = ta[:-1]
        print("A and ta data is now odd")

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

    S = cp.array(S)

    res, err = dec.RL_gauss_deconvolve(
        signal=S, kern=forcing_downsampled, iteration_list=iterlist, gpu=True
    )

    print(f"deconv done for gamma={gamma}")

    fname = f"gamma_{gamma}.npz"

    np.savez(
        savedata / fname,
        time=T,
        signal=S,
        amplitude=A,
        arrival_time=ta,
        forcing_original=forcing_original,
        forcing_downsampled=forcing_downsampled,
        result=res,
        error=err,
    )
