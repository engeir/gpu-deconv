"""Run analysis procedures investigating different forcing sampling strategies."""

import fractions
import itertools
import pathlib

import cosmoplots
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
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


def err_as_latex(num: float, *, precision: int = 0, inline_dollar: bool = True) -> str:
    """Print the error using `tex` styling."""
    number = f"{num:.{precision}e}"
    first = number.split("e")[0]
    last = number.split("e")[1]
    if last == "+00":
        out = first
    elif last == "+01":
        out = first + r"\cdot 10"
    else:
        out = first + r"\cdot 10^{" + last.replace("0", "") + r"}"
    if inline_dollar:
        return r"$e\approx" + out + "$"
    return r"\(e\approx" + out + r"\)"


class Plotter:
    """Plotting functions showing how forcing sampling affect the deconvolution."""

    def __init__(self, file: pathlib.Path | None = None) -> None:
        self.all_ds: list[tuple[pathlib.Path, xr.Dataset]] = []
        if file is not None:
            with xr.open_dataset(file) as f_:
                self.all_ds.append((file, f_))
            return
        for f in utils.all_files("no_noise_downsampled_forcing"):
            with xr.open_dataset(f) as f_:
                self.all_ds.append((f, f_))

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
    def _plot_response(
        da: xr.DataArray,
        true_pulse: xr.DataArray | None = None,
        noisy: str = "",
    ) -> None:
        if not hasattr(da, "sample_ratio"):
            da.plot(label="Original", c="k")  # type: ignore[call-arg]
            return
        for val in da.sample_ratio.data:
            label = str(fractions.Fraction(val).limit_denominator(1000))
            attr = {"color": "grey"} if val == 0 else {}
            ls = "--"
            if "dynamic" in noisy:
                ls = ":"
                label = f"D, {label}"
            elif "additive" in noisy:
                label = f"A, {label}"
            if "iterlist" in da.dims:
                arr = da.sel(sample_ratio=val).isel(iterlist=-1)
                if true_pulse is not None:
                    err = np.sum((arr.data - true_pulse.data) ** 2) / len(
                        true_pulse.data
                    )
                    label += f", {err_as_latex(err)}"
                arr.plot(label=label, ls=ls, **attr)  # type: ignore[arg-type]
            else:
                da.sel(sample_ratio=val).plot(label=label, ls=ls, **attr)  # type: ignore[arg-type]

    @staticmethod
    def _plot_err(da: xr.DataArray, noisy: str = "") -> None:
        for val in da.sample_ratio.data:
            attr = {"color": "grey"} if val == 0 else {}
            label = str(fractions.Fraction(val).limit_denominator(1000))
            ls = "--"
            if "dynamic" in noisy:
                ls = ":"
                label = f"D, {label}"
            elif "additive" in noisy:
                label = f"A, {label}"
            arr = da.sel(sample_ratio=val)
            arr.plot(label=label, ls=ls, **attr)  # type: ignore[arg-type]

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
        attrs = [
            ("g", "gamma"),
            ("dt", "time_step"),
            ("K", "total_pulses"),
            ("e", "epsilon"),
        ]
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

    def plot_dataset(self, ds: xr.Dataset) -> None:  # noqa: PLR0912, C901
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
                plt.sca(ax[1])
                plt.semilogy()
                self._plot_err(ds[var], noisy=var)
                continue
            elif any(v in var for v in ["response", "pulse_shape"]):
                plt.sca(ax[4])
                plt.xlim((-3, 10))
                if "pulse_shape" in ds.data_vars and var != "pulse_shape":
                    extra = ds.pulse_shape
                else:
                    extra = None
                self._plot_response(ds[var], true_pulse=extra, noisy=var)
                plt.sca(ax[5])
                self._plot_response(ds[var], noisy=var)
                continue
            elif "event" in var:
                continue
                # plt.sca(ax[1])
                # min_, max_ = ds["signal"].time.min(), ds["signal"].time.max()
                # plt.xlim((min_ - 0.05 * (max_ - min_), max_ + 0.05 * (max_ - min_)))
                # arr = ds[var]
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
        for f, f_ in self.all_ds:
            self.plot_dataset(f_)
            (img_dir := f.parent / "plt").mkdir(exist_ok=True)
            plt.savefig(img_dir / f"{f.stem}.png")
            # plt.close("all")

    def plot_all_responses(self) -> None:
        """Plot the response pulse function from all found datasets."""
        for f, f_ in self.all_ds:
            self.plot_dataset(f_)
            (img_dir := f.parent / "plt").mkdir(exist_ok=True)
            plt.savefig(img_dir / f"{f.stem}-pulse-shape.png")
            plt.close("all")


class PlotPulseShapes:
    """Plot only the pulse shapes from experiments."""

    def __init__(self, exp: pathlib.Path) -> None:
        self.path = exp
        with xr.open_dataset(exp) as f_:
            self.exp = f_
        self.original: xr.DataArray | None = None

    def set_original(self, original: xr.DataArray) -> None:
        """Define an array as the original pulse shape."""
        self.original = original

    def _get_scaling(self, arr: xr.DataArray, scale: bool | int) -> float | int:
        match scale:
            case False:
                return 1
            case True:
                return float(arr.data.max())
            case int(s) if self.original is not None:
                midpoint = len(arr.data) // 2
                return float(arr.data[midpoint + s] / self.original.data[midpoint + s])
            case _:
                raise ValueError

    def plot(self, *ax: mpl.axes.Axes, scale: bool | int = True) -> list[float]:  # noqa: C901
        """Plot the pulse shape from all experiments that were provided.

        The method assumes all experiments to be similar, thus including all pulse
        shapes on the same axis. An original pulse shape will be plotted if it is set by
        the `set_original` method.
        """
        if any("additive" in v for v in self.exp.data_vars) and any(  # type: ignore[operator]
            "dynamic" in v  # type: ignore[operator]
            for v in self.exp.data_vars
        ):
            mult = 2
        else:
            mult = 1
        c_list = plastik.colors.create_colorlist(
            "cmc.batlow", self.exp.sizes["sample_ratio"] * mult
        )
        c_cyc = itertools.cycle(c_list)
        lss = plastik.lines.get_linestyle_cycle()
        plt.sca(ax[0])

        errors = []
        for var_ in self.exp.data_vars:
            var = str(var_)
            if "err" in var or not any(v in var for v in ["response", "pulse_shape"]):
                continue
            da = self.exp[var]
            if "pulse_shape" in self.exp.data_vars and var != "pulse_shape":
                true_pulse = self.exp.pulse_shape
            if not hasattr(da, "sample_ratio"):
                (da / self._get_scaling(da, scale)).plot(label="Original", c="k")  # type: ignore[call-arg]
                continue
            for val in da.sample_ratio.data:
                frac = fractions.Fraction(val).limit_denominator(1000)
                label = f"$r={str(1) if frac == 0 else frac}$"
                c = next(c_cyc)
                attr = {"color": "grey"} if val == 0 else {"color": c}
                ls = next(lss)
                if "dynamic" in var:
                    label = f"D, {label}"
                elif "additive" in var:
                    label = f"A, {label}"
                if "iterlist" in da.dims:
                    arr = da.sel(sample_ratio=val).isel(iterlist=-1)
                    (arr / self._get_scaling(arr, scale)).plot(
                        label=label,
                        ls=ls,
                        **attr,  # type: ignore[arg-type]
                    )
                else:
                    arr = da.sel(sample_ratio=val)
                    (arr / self._get_scaling(arr, scale)).plot(
                        label=label,
                        ls=ls,
                        **attr,  # type: ignore[arg-type]
                    )
                if true_pulse is not None:
                    errors.append(
                        np.sum((arr.data - true_pulse.data) ** 2) / len(true_pulse.data)
                    )
                    # label += f", {err_as_latex(err)}"
        return errors


def main() -> None:
    """Run the main program."""
    p = Plotter(
        pathlib.Path(
            "/media/fusion/fusion-files/een023/projects/gpu-deconv/assets/noise_downsampled_forcing/lossless_repeat-gamma_10-epsilon_2-boxcar-dynamic.nc"
        )
    )
    p.plot_files()
    plt.show()


if __name__ == "__main__":
    main()
