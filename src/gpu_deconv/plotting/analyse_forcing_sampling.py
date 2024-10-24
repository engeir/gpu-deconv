"""Run analysis procedures investigating different forcing sampling strategies."""

import fractions

import cosmoplots
import matplotlib as mpl
import matplotlib.pyplot as plt
import plastik
import returns.maybe
import returns.result
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
            label = str(fractions.Fraction(val).limit_denominator(1000))
            arr = da.sel(sample_ratio=val)
            ls = "-" if val == 0 else "--"
            attr = {"color": "k", "label": "Original"} if val == 0 else {"label": label}
            arr.plot(ls=ls, **attr)  # type: ignore[arg-type]

    @staticmethod
    def _plot_response(da: xr.DataArray) -> None:
        if not hasattr(da, "sample_ratio"):
            da.plot(label="Original", c="k")  # type: ignore[call-arg]
            return
        for val in da.sample_ratio.data:
            label = fractions.Fraction(val).limit_denominator(1000)
            attr = {"color": "grey"} if val == 0 else {}
            if "iterlist" in da.dims:
                arr = da.sel(sample_ratio=val).isel(iterlist=-1)
                arr.plot(label=label, ls="--", **attr)  # type: ignore[arg-type]
            else:
                da.sel(sample_ratio=val).plot(label=label, ls="--", **attr)  # type: ignore[arg-type]

    @staticmethod
    def _plot_err(da: xr.DataArray) -> None:
        for val in da.sample_ratio.data:
            label = fractions.Fraction(val).limit_denominator(1000)
            arr = da.sel(sample_ratio=val)
            arr.plot(label=label)  # type: ignore[call-arg]

    @staticmethod
    @returns.result.safe
    def _find_attr(ds: xr.Dataset | xr.DataArray, name: str) -> str:
        return str(getattr(ds, name))

    @staticmethod
    def _get_title(ds: xr.DataArray | xr.Dataset) -> returns.maybe.Maybe[str]:
        title = ""
        match Plotter._find_attr(ds, "sampling_strategy"):
            case returns.result.Success(value):
                title += f"{value} : "
        attrs = [("g", "gamma"), ("dt", "time_step"), ("K", "total_pulses")]
        attrs_out = []
        for v, a in attrs:
            match Plotter._find_attr(ds, a):
                case returns.result.Success(value):
                    attrs_out.append(f"{v} = {value}")
        title += ", ".join(attrs_out)
        if title == "":
            return returns.maybe.Nothing
        return returns.maybe.Some(title)

    @staticmethod
    def _create_fig_ax(ds: xr.Dataset) -> tuple[mpl.figure.Figure, list[mpl.axes.Axes]]:
        match Plotter._get_title(ds):
            case returns.maybe.Some(title):
                fig, ax = plastik.figure_grid(3, 2, using={"expand_top": 1.05})
                fig.suptitle(title)
            case _:
                fig, ax = plastik.figure_grid(3, 2)
        return fig, ax

    def plot_dataset(self, ds: xr.Dataset) -> None:
        """Plot a single dataset created from the `TimeSeriesModel` class."""
        fig, ax = self._create_fig_ax(ds)
        for var_ in ds.data_vars:
            var = str(var_)
            if "forcing" in var:
                if "downsampled" in var:
                    plt.sca(ax[3])
                else:
                    plt.sca(ax[2])
                    self._plot_forcing(ds[var])
                    continue
                arr = ds[var]
            elif "signal" in var:
                plt.sca(ax[0])
                arr = ds[var]
            elif "err" in var:
                plt.sca(ax[5])
                plt.semilogy()
                self._plot_response(ds[var])
                continue
            elif any(v in var for v in ["response", "pulse_shape"]):
                plt.sca(ax[4])
                plt.xlim((-3, 10))
                self._plot_response(ds[var])
                continue
            elif "event" in var:
                plt.sca(ax[1])
                min_, max_ = ds["signal"].time.min(), ds["signal"].time.max()
                plt.xlim((min_ - 0.05 * (max_ - min_), max_ + 0.05 * (max_ - min_)))
                arr = ds[var]
            else:
                continue
            if any(res := [val for val in arr.attrs if "factor" in val]):
                label = arr.attrs[res[0]]
            else:
                label = var.split("_")[-1]
            if label == var:
                label = "Original"
            attr = {"color": "k"} if label == "Original" else {}
            ls = "-" if label == "Original" else "--"
            arr.plot(label=label, ls=ls, **attr)  # type: ignore[arg-type]
        [a.legend(loc="upper right") for a in ax]
        [cosmoplots.change_log_axis_base(a) for a in ax]
        [a.set_title("") for a in ax]

    def plot_files(self) -> None:
        """Plot all found datasets."""
        for f in self.all_ds:
            self.plot_dataset(f)


def main() -> None:
    """Run the main program."""
    p = Plotter()
    p.plot_files()
    plt.show()


if __name__ == "__main__":
    main()
