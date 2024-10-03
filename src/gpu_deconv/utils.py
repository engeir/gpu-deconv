"""Generic functions used across the project."""

import pathlib
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
    if any(len(x) != len(first) for x in signals[1:]):
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
        time_array, signal = self._model.make_realization()
        self._forcing_model = self._model.get_last_used_forcing()
        time_array = self._make_odd_length(time_array)
        signal = self._make_odd_length(signal)
        amplitude = self._forcing_model.amplitudes
        arrival_times = self._forcing_model.arrival_times
        arrival_times.sort()
        forcing = np.zeros(time_array.size)
        arrival_time_index = np.ceil(arrival_times / self._dt).astype(int)
        for i in range(arrival_time_index.size):
            forcing[arrival_time_index[i]] += amplitude[i]
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
            },
            coords={
                "time": ("time", time_array, {"long_name": "Time", "units": "yr"}),
                "arrival_time": (
                    "arrival_time",
                    arrival_times,
                    {"long_name": "Time", "units": "yr"},
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
            self._model._add_pulse_to_signal(result, pulse_parameters)

        if self._model._noise is not None:
            result += self._model._discretize_noise(self._forcing_model)

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
                "tau": ("tau", tau, {"long_name": "Time lag", "units": "yr"}),
                "iterlist": ("iterlist", iterlist, {"long_name": "Total iterations"}),
            },
            attrs={"description": "Result from deconvolution"},
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
        factor: int,
        desc: str | None = None,
    ) -> xr.DataArray:
        """Dress an upsampled array named `name` with a time dimension."""
        _a = {"long_name": f"{name}, dense", "upsample_factor": factor}
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
        factor: int,
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
        factor : int
            The downsampling factor.
        desc : str | None
            An optional description of the array.
        """
        if len(data) == len(time):
            _d = data
            _t = time
        elif len(data) == len(time[::factor]):
            _d = data
            _t = time[::factor]
        elif len(data) == len(time[::factor][:-1]):
            _d = data
            _t = time[::factor][:-1]
        else:
            raise UnequalArrayLengthError
        _a = {"long_name": f"{name}, sparse", "downsample_factor": factor}
        if desc is not None:
            _a.update({"description": desc})
        return xr.DataArray(
            _d,
            dims=["time_sparse"],
            coords={
                "time_sparse": (
                    "time_sparse",
                    _t,
                    {"long_name": "Time", "units": "yr"},
                )
            },
            attrs=_a,
        )
