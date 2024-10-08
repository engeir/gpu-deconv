"""Run analysis procedures investigating different forcing sampling strategies."""

import fractions

import matplotlib.pyplot as plt
import plastik
import xarray as xr

from gpu_deconv import utils

plt.style.use(
    [
        "https://raw.githubusercontent.com/uit-cosmo/cosmoplots/main/cosmoplots/default.mplstyle",
        "gpu_deconv.extra",
        "gpu_deconv.jgr",
    ],
)


class Plotter:
    """Plotting functions showing how forcing sampling affect the deconvolution."""

    def __init__(self) -> None:
        self.all_ds = []
        for f in utils.all_files("no_noise_downsampled_forcing"):
            with xr.open_dataset(f) as f_:
                self.all_ds.append(f_)

    @staticmethod
    def _to_str_fraction(ratio: float) -> str:
        return str(fractions.Fraction(ratio))

    @staticmethod
    def _plot_forcing(da: xr.DataArray) -> None:
        for val in da.sample_ratio.data:
            arr = da.sel(sample_ratio=val)
            arr.plot(label=val)

    @staticmethod
    def _plot_response(da: xr.DataArray) -> None:
        for val in da.sample_ratio.data:
            if "iterlist" in da.dims:
                arr = da.sel(sample_ratio=val).isel(iterlist=-1)
                arr.plot(label=val, ls="--")
            else:
                da.sel(sample_ratio=val).plot(label=val, ls="--")

    @staticmethod
    def _plot_err(da: xr.DataArray) -> None:
        for val in da.sample_ratio.data:
            arr = da.sel(sample_ratio=val)
            arr.plot(label=val)

    def plot_dataset(self, ds: xr.Dataset) -> None:
        fig, ax = plastik.figure_grid(3, 2)
        for var_ in ds.data_vars:
            var = str(var_)
            if "forcing" in var:
                if "downsampled" in var:
                    plt.sca(ax[1])
                    # ds["forcing"].sel(sample_ratio=0).plot(label="Original")
                else:
                    plt.sca(ax[0])
                    # for ratio in ds[var].sample_ratio.data:
                    self._plot_forcing(ds[var])
                    continue
                arr = ds[var]
            elif "signal" in var:
                plt.sca(ax[2])
                arr = ds[var]
            elif "err" in var:
                plt.sca(ax[5])
                plt.semilogy()
                self._plot_response(ds[var])
                continue
            elif "pulse_shape" in var:
                plt.sca(ax[4])
                arr = ds[var]
            elif "response" in var:
                plt.sca(ax[4])
                plt.xlim((-2, 20))
                self._plot_response(ds[var])
                continue
                # if "iterlist" in ds[var].dims:
                #     arr = ds[var].isel(iterlist=-1)
                # else:
                #     arr = ds[var]
            elif "event" in var:
                plt.sca(ax[3])
                arr = ds[var]
            else:
                continue
            if any(res := [val for val in arr.attrs if "factor" in val]):
                label = arr.attrs[res[0]]
            else:
                label = var.split("_")[-1]
            if label == var:
                label = "Original"
            ls = "-" if label == "Original" else "--"
            arr.plot(label=label, ls=ls)
        [a.legend(loc="upper right") for a in ax]
        [a.set_title("") for a in ax]


def main() -> None:
    """Run the main program."""
    p = Plotter()
    p.plot_dataset(p.all_ds[0])
    p.plot_dataset(p.all_ds[1])
    plt.show()


if __name__ == "__main__":
    main()
