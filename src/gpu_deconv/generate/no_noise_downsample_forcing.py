"""Downsample (then upsample) forcing, no noise."""

# IMPORTANT: This script has been verified against Sajidah's output.
import fractions
import pathlib

import cupy as cp
import fppanalysis.deconvolution_methods as dec
import matplotlib.pyplot as plt
import numpy as np
import plastik
import superposedpulses.pulse_shape as ps_store
import xarray as xr
from icecream import ic

from gpu_deconv import utils
from gpu_deconv.plotting.analyse_forcing_sampling import PlotPulseShapes

plt.style.use(
    [
        "https://raw.githubusercontent.com/uit-cosmo/cosmoplots/main/cosmoplots/default.mplstyle",
        "gpu_deconv.extra",
        "gpu_deconv.jgr",
    ],
)


# gamma_list = [0.9]
# iterlist = [int(1e2), int(1e3), int(1e4), int(1e5)]
iterlist = [int(1e2), int(1e3)]

seed = 125
seed_tw = 10
seed_a = 2

dt = 0.1
gamma_list = [0.01, 0.1, 1]
total_samples = 1e4 + 1

savedata = utils.ASSETS / "no_noise_downsampled_forcing"
savedata.mkdir(parents=False, exist_ok=True)


def _find_pulse_shape(ps: str) -> ps_store.PulseGenerator | None:
    match ps:
        case "exp":
            return None
        case "lomax":
            return utils.lomax_pulse_generator
        case "gamma":
            return utils.gamma_pulse_generator
    raise ValueError


def _not_noisy_experiments(
    strategy: str, /, *, gamma: float, guess: str = "heaviside", ps: str = "exp"
) -> pathlib.Path:
    print(f"Generate signal for {gamma = }")

    total_pulses = int(total_samples * gamma * dt)
    ic(total_pulses)
    # ds = utils.new_time_series(total_pulses=total_pulses, gamma=gamma, dt=dt)
    ds = utils.TimeSeriesModel(seed)(
        total_pulses=total_pulses, gamma=gamma, dt=dt, ps=_find_pulse_shape(ps)
    )
    time_array = ds["time"]
    signal = ds["signal"]
    forcing_original = ds["forcing"]
    pulse = ds["pulse_shape"]

    initial_guess = np.heaviside(pulse.tau.data, 1)
    if guess == "boxcar":
        box_width = 20
        initial_guess[np.argwhere(pulse.tau.data > box_width)] = 0
    if guess == "ones":
        initial_guess = np.ones_like(pulse.tau.data)
    res, err = dec.RL_gauss_deconvolve(
        signal=cp.asarray(signal),
        kern=cp.asarray(forcing_original),
        iteration_list=iterlist,
        initial_guess=cp.asarray(initial_guess),
        gpu=True,
    )
    res_da = utils.Wardrobe.dress_response_pulse(res.get(), time_array.data, iterlist)
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
        res_ = utils.Wardrobe.dress_response_pulse(res.get(), time_array.data, iterlist)
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

    fname = f"{strategy}-gamma_{str(gamma).replace('.', '')}-{guess}"

    ds.to_netcdf(f"{savedata / ps / fname}.nc", format="NETCDF4")
    fname_ext = fname + ".nc"
    return savedata / ps / fname_ext


def plot_wrap(
    p1: pathlib.Path, p2: pathlib.Path, p3: pathlib.Path, p4: pathlib.Path
) -> None:
    """Plot four experiments in a single figure to compare down- and upsampling."""
    fig, axs = plastik.figure_grid(rows=2, columns=2)
    errors = []
    for i, (ax, p_) in enumerate(zip(axs, (p1, p2, p3, p4), strict=True)):
        p = PlotPulseShapes(p_)
        errors.append(p.plot(ax))
        ax.legend()
        if i != 1:
            ax.get_legend().remove()
        match i:
            case 0:
                text = "w/ loss\n zeros"
            case 1:
                text = "w/o loss\n zeros"
            case 2:
                text = "w/ loss\n repeat"
            case 3:
                text = "w/o loss\n repeat"
        ax.text(2, 1, text, va="top")
        ax.set_xlabel(r"$\tau/\tau_{d}$")
        ax.set_ylabel(r"$\varphi/\max\varphi$")
        ax.set_title("")
        ax.set_xlim((-1.5, 15))
    fname = "-".join(p1.stem.split("-")[1:])
    fig.savefig(savedata / "plt" / p1.parent.parts[-1] / fname)
    utils.print_table(
        savedata / "plt" / p1.parent.parts[-1] / fname,
        "r",
        [r"\(1\)", r"\(1/2\)", r"\(1/5\)", r"\(1/10\)", r"\(1/24\)"],
        *errors,
    )
    # plt.show()
    plt.close("all")


def _make_plot(gamma: float, guess: str, ps: str) -> None:
    plot_wrap(
        *[
            _not_noisy_experiments("lossy_nth", gamma=gamma, guess=guess, ps=ps),
            _not_noisy_experiments("lossless_nth", gamma=gamma, guess=guess, ps=ps),
            _not_noisy_experiments("lossy_repeat", gamma=gamma, guess=guess, ps=ps),
            _not_noisy_experiments("lossless_repeat", gamma=gamma, guess=guess, ps=ps),
        ]
    )


def main() -> None:
    """Run the main program."""
    # Exp
    _make_plot(gamma=1e-1, guess="heaviside", ps="exp")
    _make_plot(gamma=1, guess="heaviside", ps="exp")
    _make_plot(gamma=10, guess="heaviside", ps="exp")
    _make_plot(gamma=1e-1, guess="boxcar", ps="exp")
    _make_plot(gamma=1, guess="boxcar", ps="exp")
    _make_plot(gamma=10, guess="boxcar", ps="exp")
    # Lomax
    _make_plot(gamma=1e-1, guess="heaviside", ps="lomax")
    _make_plot(gamma=1, guess="heaviside", ps="lomax")
    _make_plot(gamma=10, guess="heaviside", ps="lomax")
    _make_plot(gamma=1e-1, guess="boxcar", ps="lomax")
    _make_plot(gamma=1, guess="boxcar", ps="lomax")
    _make_plot(gamma=10, guess="boxcar", ps="lomax")
    # Gamma
    _make_plot(gamma=1e-1, guess="heaviside", ps="gamma")
    _make_plot(gamma=1, guess="heaviside", ps="gamma")
    _make_plot(gamma=10, guess="heaviside", ps="gamma")
    _make_plot(gamma=1e-1, guess="boxcar", ps="gamma")
    _make_plot(gamma=1, guess="boxcar", ps="gamma")
    _make_plot(gamma=10, guess="boxcar", ps="gamma")


if __name__ == "__main__":
    main()
