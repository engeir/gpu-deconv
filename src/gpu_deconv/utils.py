"""Generic functions used across the project."""

import fractions
import pathlib
from collections.abc import Generator
from typing import Literal

import numpy as np
import superposedpulses.point_model as pm
import xarray as xr


def find_repo_root(path: pathlib.Path | str) -> pathlib.Path:
    """Find repository root from the path's parents."""
    for path_ in pathlib.Path(path).parents:
        # Check whether "path/.git" exists and is a directory
        git_dir = path_ / ".git"
        if git_dir.is_dir():
            return path_
    raise FileNotFoundError


ASSETS = find_repo_root(__file__) / "assets"


def all_files(directory: str) -> Generator[pathlib.Path]:
    """List all asset files in the given directory."""
    return (ASSETS / directory).glob("*nc")


class EvenLengthError(Exception):
    """Exception raised for even length arrays."""

    def __init__(self, message: str = "The arrays must have an odd length.") -> None:
        self.message = message
        super().__init__(self.message)


class TauError(EvenLengthError):
    """Exception raised for bad tau arrays."""

    def __init__(
        self,
        message: str = "The arrays must have an odd length.",
        arr: np.ndarray | None = None,
    ) -> None:
        self.message = message
        if arr is not None:
            self.message += f" Tau starts and ends with {arr[0]} and {arr[-1]}."
        super().__init__(self.message)


class UnequalArrayLengthError(Exception):
    """Exception raised for uneven array lengths."""

    def __init__(self, message: str = "The arrays must have the same length.") -> None:
        self.message = message
        super().__init__(self.message)


def verify_odd_length(signal: np.ndarray | xr.DataArray) -> None:
    """Raise an error if the signal does not have an odd number length.

    signal : np.ndarray
        The signal array.
    """
    if len(signal) % 2 == 0:
        raise EvenLengthError


def verify_equal_length(*signals: np.ndarray | xr.DataArray) -> None:
    """Raise an error if the signals have unequal length.

    *signals : np.ndarray
        The signals to compare
    """
    first = signals[0]
    if any(x.size != first.size for x in signals[1:]):
        raise UnequalArrayLengthError


class TimeSeriesModel:
    """Generate a signal without noise, and with additive and dynamical noise."""

    def __init__(
        self,
        total_pulses: float | None = None,
        gamma: float | None = None,
        dt: float | None = None,
    ) -> None:
        self._total_pulses = 1e3 if total_pulses is None else total_pulses
        self._gamma = 1e-1 if gamma is None else gamma
        self._dt = 1e-2 if dt is None else dt

    def __call__(
        self,
        total_pulses: float,
        gamma: float,
        dt: float,
        *,
        epsilon_dynamic: float | None = None,
        epsilon_additive: float | None = None,
    ) -> xr.Dataset:
        """Create a dataset containing all deconvolution relevant time series."""
        self._total_pulses, self._gamma, self._dt = total_pulses, gamma, dt
        self._create_model()
        _signal = len(self.ds["signal"].data)
        if epsilon_dynamic is not None:
            signal = self._add_noise(epsilon_dynamic, "dynamic")
            if len(signal) != _signal:
                signal = signal[:_signal]
            self.ds["signal_dynamic"] = xr.DataArray(
                signal,
                dims=["time"],
                attrs={
                    "description": "Signal with dynamic noise",
                    "long_name": "Signal",
                    "epsilon": epsilon_additive,
                },
            )
        if epsilon_additive is not None:
            signal = self._add_noise(epsilon_additive, "additive")
            if len(signal) != _signal:
                signal = signal[:_signal]
            self.ds["signal_additive"] = xr.DataArray(
                signal,
                dims=["time"],
                attrs={
                    "description": "Signal with additive noise",
                    "long_name": "Signal",
                    "epsilon": epsilon_dynamic,
                },
            )
        return self.ds

    def _create_model(self) -> None:
        self._model = pm.PointModel(
            waiting_time=1 / self._gamma,
            total_duration=self._total_pulses / self._gamma,
            dt=self._dt,
        )
        if not len(self._model._times) % 2:  # noqa: SLF001
            self._model._times = self._model._times[:-1]  # noqa: SLF001
        time_array, signal = self._model.make_realization()
        self._forcing_model = self._model.get_last_used_forcing()
        time_array = self._make_odd_length(time_array)
        signal = self._make_odd_length(signal)
        amplitude = self._forcing_model.amplitudes
        arrival_times = self._forcing_model.arrival_times
        idx = np.argsort(arrival_times)
        arrival_times = arrival_times[idx]
        amplitude = amplitude[idx]
        forcing = np.zeros(time_array.size)
        arrival_time_index = np.ceil(arrival_times / self._dt).astype(int)
        for i in range(arrival_time_index.size):
            forcing[arrival_time_index[i]] += amplitude[i]
        pulse_params = self._forcing_model.get_pulse_parameters(0)
        pulse_shape = self._model._pulse_generator.get_pulse(  # noqa: SLF001
            self._model._times - pulse_params.arrival_time,  # noqa: SLF001
            pulse_params.duration,
        )
        pulse_max = np.argmax(pulse_shape)
        half = len(pulse_shape) // 2
        roll = half - pulse_max
        pulse_shape = np.roll(pulse_shape, roll)
        tau = time_array
        if tau[0] != -tau[-1]:
            mid = len(tau) // 2
            tau = tau - tau[mid]
        if tau[0] != -tau[-1]:
            raise TauError(arr=tau)
        self.ds = xr.Dataset(
            data_vars={
                "signal": (
                    "time",
                    signal,
                    {
                        "description": "The original signal",
                        "long_name": "Signal",
                    },
                ),
                "forcing": (
                    "time",
                    forcing,
                    {"description": "The original forcing", "long_name": "Forcing"},
                ),
                "events": (
                    "arrival_time",
                    amplitude,
                    {
                        "description": "Amplitudes at the arrival times",
                        "long_name": "Amplitude",
                    },
                ),
                "pulse_shape": (
                    "tau",
                    pulse_shape,
                    {
                        "description": "The original shape of the response pulse",
                        "long_name": "Response pulse",
                    },
                ),
            },
            coords={
                "time": (
                    "time",
                    time_array,
                    {"long_name": "Time", "units": r"$\tau_d$"},
                ),
                "tau": ("tau", tau, {"long_name": "Time lag", "units": r"$\tau_d$"}),
                "arrival_time": (
                    "arrival_time",
                    arrival_times,
                    {"long_name": "Time", "units": r"$\tau_d$"},
                ),
            },
            attrs={
                "description": "A set of signals and forcings generated from one-sided exponential pulses.",
                "gamma": str(self._gamma),
                "time_step": str(self._dt),
                "total_pulses": str(self._total_pulses),
            },
        )

    @staticmethod
    def _make_odd_length(arr: np.ndarray) -> np.ndarray:
        if len(arr) % 2:
            return arr
        return arr[:-1]

    def _add_noise(
        self, epsilon: float, version: Literal["additive", "dynamic"]
    ) -> np.ndarray:
        self._model.add_noise(epsilon, noise_type=version)
        return self._reuse_realization()

    def _reuse_realization(self) -> np.ndarray:
        result = np.zeros_like(self.ds["signal"].data)

        for k in range(self._forcing_model.total_pulses):
            pulse_parameters = self._forcing_model.get_pulse_parameters(k)
            self._model._add_pulse_to_signal(result, pulse_parameters)  # noqa: SLF001

        if self._model._noise is not None:  # noqa: SLF001
            result += self._model._discretize_noise(self._forcing_model)  # noqa: SLF001

        return result


new_time_series = TimeSeriesModel()


class Wardrobe:
    """Dress numpy arrays up in nice xarray clothes."""

    @staticmethod
    def dress_response_pulse(
        data: np.ndarray, tau: np.ndarray, iterlist: list[int]
    ) -> xr.DataArray:
        """Dress a pulse function with tau and iterlist dimensions."""
        if tau[0] != -tau[-1]:
            mid = len(tau) // 2
            tau = tau - tau[mid]
        if tau[0] != -tau[-1]:
            raise TauError(arr=tau)
        return xr.DataArray(
            data,
            dims=["tau", "iterlist"],
            coords={
                "tau": ("tau", tau, {"long_name": "Time lag", "units": r"$\tau_d$"}),
                "iterlist": ("iterlist", iterlist, {"long_name": "Total iterations"}),
            },
            attrs={
                "long_name": "Response pulse",
                "description": "Result from deconvolution",
            },
        )

    @staticmethod
    def dress_response_pulse_error(data: np.ndarray) -> xr.DataArray:
        """Dress the error from the deconvolution with an iterations dimension."""
        return xr.DataArray(
            data,
            dims=["iterations"],
            coords={
                "iterations": (
                    "iterations",
                    np.arange(len(data)),
                    {"long_name": "Iterations"},
                )
            },
            attrs={
                "description": "Errors after result from deconvolution",
                "long_name": "Error",
            },
        )

    @staticmethod
    def dress_an_upsampled(
        name: str,
        /,
        data: np.ndarray,
        time: np.ndarray,
        ratio: fractions.Fraction,
        desc: str | None = None,
    ) -> xr.DataArray:
        """Dress an upsampled array named `name` with a time dimension."""
        _a = {"long_name": f"{name}, dense", "sample_factor": ratio}
        if desc is not None:
            _a.update({"description": desc})
        return xr.DataArray(
            data,
            dims=["time"],
            coords={"time": time},
            attrs=_a,
        )

    @staticmethod
    def dress_a_downsampled(
        name: str,
        /,
        data: np.ndarray,
        time: np.ndarray,
        ratio: fractions.Fraction,
        desc: str | None = None,
    ) -> xr.DataArray:
        """Dress a downsampled array named `name` with a time dimension.

        Parameters
        ----------
        name : str
            The name given to the array and used in labels. It is recommended to specify
            capital first letter.
        data : np.ndarray
            The data to record.
        time : np.ndarray
            The time dimension. This can either be the same length as `data` (i.e.
            downsampled), or the original time array which will then be attempted to be
            downsampled.
        ratio : fractions.Fraction
            The downsampling factor.
        desc : str | None
            An optional description of the array.

        Returns
        -------
        xr.DataArray
            A data array with all the necessary coordinates and attributes.

        Raises
        ------
        UnequalArrayLengthError
            If the data array and the time array have incompatible lengths.
        """
        if len(data) == len(time):
            _d = data
            _t = time
        elif len(data) == len(time[:: ratio.denominator]):
            _d = data
            _t = time[:: ratio.denominator]
        elif len(data) == len(time[:: ratio.denominator][:-1]):
            _d = data
            _t = time[:: ratio.denominator][:-1]
        else:
            raise UnequalArrayLengthError
        _a = {"long_name": f"{name}, sparse", "downsample_factor": str(ratio)}
        if desc is not None:
            _a.update({"description": desc})
        dim = f"time_sparse_{ratio.numerator}_{ratio.denominator}"
        return xr.DataArray(
            _d,
            dims=[dim],
            coords={
                dim: (
                    dim,
                    _t,
                    {"long_name": "Time", "units": r"$\tau_d$"},
                )
            },
            attrs=_a,
        )


class SampleStrategyForcing:
    """Container class specifying different ways of down- and up-sampling an array."""

    def __init__(self, array: xr.DataArray) -> None:
        self.arr = array.copy()
        self._arr = array.copy()

    def _check_da_dims(
        self, arr: xr.DataArray, new_dim: str, new_coord: float
    ) -> xr.DataArray:
        if new_dim not in self.arr.dims:
            self.arr = self.arr.expand_dims(dim={new_dim: [0]})
        return arr.expand_dims(dim={new_dim: [new_coord]})

    def _create_corresponding_upsampled(
        self, arr: np.ndarray, times: np.ndarray, ratio: fractions.Fraction
    ) -> None:
        da = Wardrobe.dress_an_upsampled(
            "Forcing",
            arr,
            times,
            ratio,
            desc="Forcing with the same length, but downsampled to keep every `n` sample.",
        )
        new_dim = "sample_ratio"
        da = self._check_da_dims(da, new_dim, float(ratio))
        self.arr = xr.concat((self.arr, da), dim=new_dim)

    @staticmethod
    def _sample_at_ratio(
        orig_arr: np.ndarray, orig_time: np.ndarray, ratio: fractions.Fraction
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample an array at a given ratio.

        Parameters
        ----------
        orig_arr : np.ndarray
            The input array that should be downsampled.
        orig_time : np.ndarray
            The time axis of the input array that should be downsampled.
        ratio : fractions.Fraction
            The ratio of the downsampling to original array.

        Returns
        -------
        np.ndarray
            The downsampled input array.
        np.ndarray
            The time axis of the downsampled input array.
        np.ndarray
            The upsampling of the input array with equal size to the input array.
        """

        def _reshape(arr: np.ndarray, ratio: fractions.Fraction) -> np.ndarray:
            rows = int(np.ceil(len(arr) / ratio.denominator))
            return np.pad(
                arr.astype(float),
                (0, ratio.denominator * rows - arr.size),
                mode="constant",
                constant_values=np.nan,
            ).reshape(rows, ratio.denominator)

        # Re-shape the array with column lengths equal to `denominator`, and pad with
        # NaN if it does not fit.
        # Keep only the first `numerator` number of columns in each row.
        reshaped = _reshape(orig_arr, ratio)
        frc = reshaped[:, : ratio.numerator].flatten()
        frc_time = _reshape(orig_time, ratio)[:, : ratio.numerator].flatten()
        if not len(frc) % 2:
            frc = frc[:-1]
            frc_time = frc_time[:-1]
        verify_odd_length(frc)
        return frc, frc_time, reshaped

    def keep_every_ratio_blind(
        self, ratio: fractions.Fraction
    ) -> tuple[xr.DataArray, xr.DataArray]:
        """Keep every `ratio`-th value in the forcing."""
        if ratio >= 1:
            raise ValueError
        _a = self._arr.data
        _t = self._arr.time.data
        # First, reshape so that the length of a row is `ratio.denominator`
        frc, frc_time, reshaped = self._sample_at_ratio(_a, _t, ratio)
        # The upsampled version is assumed to be filled with zeros where the
        # downsampling occurred.
        reshaped[:, ratio.numerator :] = 0
        full = reshaped.flatten()[: len(_a)]
        verify_equal_length(full, self._arr)
        down = Wardrobe.dress_a_downsampled(
            "Forcing",
            frc,
            frc_time,
            ratio,
            desc='Raw "choose every n" downsampling',
        )
        self._create_corresponding_upsampled(full, _t, ratio)
        return down, self.arr

    def keep_every_ratio_cumsum(
        self, ratio: fractions.Fraction = fractions.Fraction(1, 10)
    ) -> tuple[xr.DataArray, xr.DataArray]:
        """Keep every `ratio`-th value in the forcing, without loss.

        This procedure effectively pushes an even forward if it happen to be on a sample
        that would be removed by the downsampling. The `cumsum` method further has the
        effect of accumulating the magnitude of all missed events within a block.
        """
        if ratio >= 1:
            raise ValueError
        _a = np.cumsum(self._arr.data)
        _t = self._arr.time.data
        frc, frc_time, reshaped = self._sample_at_ratio(_a, _t, ratio)
        frc = np.diff(np.concatenate(([0], frc)))
        # The upsampled version is assumed to be filled with zeros where the
        # downsampling occurred.
        width = 2 * ratio.numerator - reshaped.shape[1]
        if ratio.numerator == 1:
            reshaped[:, 1:] = (
                np.ones_like(reshaped[:, 1:]) * np.atleast_2d(reshaped[:, 0]).T
            )
        else:
            reshaped[:, ratio.numerator :] = reshaped[:, width : ratio.numerator]
        full = reshaped.flatten()[: len(_a)]
        full = np.diff(np.concatenate(([0], full)))
        verify_equal_length(full, self._arr)
        down = Wardrobe.dress_a_downsampled(
            "Forcing",
            frc,
            frc_time,
            ratio,
            desc='Raw "choose every n" downsampling',
        )
        self._create_corresponding_upsampled(full, _t, ratio)
        return down, self.arr
