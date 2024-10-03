"""Plot all time series and compare with Sajidah's output."""

import pathlib
from collections.abc import Callable, Generator

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from gpu_deconv import utils

plt.style.use(
    [
        "https://raw.githubusercontent.com/uit-cosmo/cosmoplots/main/cosmoplots/default.mplstyle",
        "gpu_deconv.extra",
        "gpu_deconv.jgr",
    ],
)
savedata = utils.ASSETS

__LOOKUP_TABLE__ = {
    "signal": "None",
    "forcing": ("time", "forcing_original"),
    "events": ("arrival_time", "amplitude"),
    "signal_dynamic": ("time", "signal_dyn"),
    "signal_additive": ("time", "signal_add"),
    "signal_additive_downsampled": ("signal_add_downsampled",),
    "signal_dynamic_downsampled": ("signal_add_downsampled",),
    "signal_additive_downsampled_upsampled": ("time", "upsampled_signal_add"),
    "signal_dynamic_downsampled_upsampled": ("time", "upsampled_signal_dyn"),
    "forcing_downsampled": "None",
    "forcing_downsampled_upsampled": ("time", "forcing_downsampled"),
    "response_pulse": ("result",),
    "response_pulse_additive": ("result_add",),
    "response_pulse_dynaimc": ("result_dyn",),
    "response_pulse_err": ("error",),
    "response_pulse_additive_err": ("error_add",),
    "response_pulse_dynaimc_err": ("error_dyn",),
}


def all_files(directory: str) -> Generator[pathlib.Path]:
    """List all files in the given directory."""
    return (savedata / directory).glob("*nc")


def run_cmd_on_files(func: Callable, files: Generator[pathlib.Path]) -> None:
    """Run all files as arguments to the same command."""
    for f in files:
        func(f)


def plot_comparison(file: pathlib.Path) -> None:
    with xr.open_dataset(file) as f:
        for var in f.data_vars:
            print(f"Looking at {var}")
            out = __LOOKUP_TABLE__[var]
            with np.load(
                file.parent / "saj" / f"{file.name.removesuffix('.nc')}.npz"
            ) as npz:
                if out == "None":
                    continue
                if "err" in var:
                    y = npz[out[0]]
                    x = np.arange(len(y))
                    plt.semilogy(x, y, label="Saj")
                    f[var].plot(label="Mine")
                elif len(out) == 2:
                    x = npz[out[0]]
                    y = npz[out[1]]
                    x = x[: len(y)]
                    y = y[: len(x)]
                    plt.plot(x, y, label="Saj")
                    f[var].plot(label="Mine")
                elif len(out) == 1 and f[var].ndim == 1:
                    x = npz["time"]
                    y = npz[out[0]]
                    nth = len(x) // len(y) + 1
                    x = x[::nth]
                    plt.plot(x, y, label="Saj")
                    f[var].plot(label="Mine")
                elif len(out) == 1 and f[var].ndim == 2:
                    data_ = f[var].isel(iterlist=-1)
                    x = npz["time"]
                    y = npz[out[0]][:, -1]
                    x = x[: len(y)]
                    y = y[: len(x)]
                    mid = len(x) // 2
                    x = x - x[mid]
                    plt.plot(x, y, label="Saj")
                    data_ = f[var].isel(iterlist=-1)
                    x = data_.tau
                    y = data_.data
                    plt.plot(x, y, label="Mine")
            plt.legend()
            plt.show()


if __name__ == "__main__":
    # run_cmd_on_files(plot_comparison, all_files("constant_noise_downsampled_signal"))
    run_cmd_on_files(plot_comparison, all_files("no_noise_downsampled_forcing"))
    # run_cmd_on_files(plot_comparison, all_files("fourier_noise_downsampled_signal"))
    # run_cmd_on_files(plot_comparison, all_files("noise_downsampled_forcing"))
    # run_cmd_on_files(plot_comparison, all_files("noise_downsampled_signal"))
    plt.show()
