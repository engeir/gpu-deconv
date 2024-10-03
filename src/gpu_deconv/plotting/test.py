"""Plot the output of the scripts."""

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


def all_files(directory: str) -> Generator[pathlib.Path]:
    """List all `npz` files in the given directory."""
    return (savedata / directory).glob("*nc")


def run_cmd_on_files(func: Callable, files: Generator[pathlib.Path]) -> None:
    """Run all files as arguments to the same command."""
    for f in files:
        func(f)


def plot_key(key: str) -> str:
    """Plot a key if it is found."""
    match key:
        case "signal_dyn":
            return "signal_dyn"
        case "signal_dynamic":
            return "signal_dynamic"
        case "signal_add":
            return "signal_add"
        case "signal_additive":
            return "signal_additive"
        case "signal_dynamic_downsampled_upsampled":
            return "signal_dynamic_downsampled_upsampled"
        case "signal_additive_downsampled_upsampled":
            return "signal_additive_downsampled_upsampled"
        case "signal":
            return "signal"
        case _:
            return "Not found"


def plot_from_npz(file: pathlib.Path) -> None:
    """Plot from a npz archive file."""
    with np.load(file) as f:
        time = f["time"]
        for key in f:
            key_ = plot_key(key)
            if key_ == "Not found":
                continue
            arr = f[key_]
            plt.plot(time, arr, label=key_)


def plot_from_nc(file: pathlib.Path) -> None:
    """Plot from a netCDF4 archive file."""
    with xr.open_dataset(file) as f:
        for key in f:
            key_ = plot_key(str(key))
            if key_ == "Not found":
                continue
            arr = f[key_]
            arr.plot(label=key_)


def plot_all_data(file: pathlib.Path) -> None:
    """Plot all data in the saved data file."""
    print(f"Looking at {file.name} inside {file.parent.name}")
    plt.figure()
    if file.suffix == ".npz":
        plot_from_npz(file)
    elif file.suffix == ".nc":
        plot_from_nc(file)
    plt.legend(title=file.name)


if __name__ == "__main__":
    # run_cmd_on_files(plot_all_data, all_files("constant_noise_downsampled_signal"))
    # run_cmd_on_files(plot_all_data, all_files("fourier_noise_downsampled_signal"))
    run_cmd_on_files(plot_all_data, all_files("no_noise_downsampled_forcing"))
    # run_cmd_on_files(plot_all_data, all_files("noise_downsampled_forcing"))
    # run_cmd_on_files(plot_all_data, all_files("noise_downsampled_signal"))
    plt.show()
