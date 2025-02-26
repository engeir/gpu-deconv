"""Generic functions used across the project."""

import fractions
import pathlib
from collections.abc import Callable, Generator, Sequence
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import returns.maybe
import scipy.signal
import superposedpulses.forcing as pf
import superposedpulses.point_model as pm
import xarray as xr
from rich.console import Console
from rich.table import Table

from gpu_deconv.plotting.analyse_forcing_sampling import err_as_latex


def find_repo_root(path: pathlib.Path | str) -> pathlib.Path:
    """Find repository root from the path's parents."""
    for path_ in pathlib.Path(path).parents:
        # Check whether "path/.git" exists and is a directory
        git_dir = path_ / ".git"
        if git_dir.is_dir():
            return path_
    raise FileNotFoundError


ASSETS = find_repo_root(__file__) / "assets"


def print_table(  # noqa: PLR0913
    name: pathlib.Path,
    var_name: str,
    var: Sequence[str],
    error_1: list[float],
    error_2: list[float],
    error_3: list[float],
    error_4: list[float],
) -> None:
    """Print values as a latex table."""
    # Error table
    table = Table(box=None)
    table.add_column(rf"\({var_name}\)")
    table.add_column("& w/ loss, zeros")
    table.add_column(r"& w/o loss, zeros")
    table.add_column(r"& w/ loss, repeat")
    table.add_column(r"& w/o loss, repeat \\")
    data = [[str(s), None, None, None, None] for s in var]
    for r, p_ in zip(data, error_1, strict=False):
        r[1] = "& " + err_as_latex(p_, precision=3, inline_dollar=False)
    for r, p_ in zip(data, error_2, strict=False):
        r[2] = "& " + err_as_latex(p_, precision=3, inline_dollar=False)
    for r, p_ in zip(data, error_3, strict=False):
        r[3] = "& " + err_as_latex(p_, precision=3, inline_dollar=False)
    for r, p_ in zip(data, error_4, strict=False):
        r[4] = "& " + err_as_latex(p_, precision=3, inline_dollar=False) + r" \\"
    for row in data:
        table.add_row(*row)
    with (name.with_suffix(".tex")).open(mode="w") as f:
        console = Console(file=f, markup=False, width=500)
        console.print(table)


def inverse_cumsum(arr: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """Compute an inverse cumsum procedure to a flat, 1D array."""
    if arr.ndim > 1:
        raise ValueError
    return np.diff(np.concatenate(([0], arr)))


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
        arr: npt.NDArray[Any] | None = None,
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


def verify_odd_length(signal: npt.NDArray[Any] | xr.DataArray) -> None:
    """Raise an error if the signal does not have an odd number length.

    Parameters
    ----------
    signal : npt.NDArray[Any] | xr.DataArray
        The signal array.

    Raises
    ------
    EvenLengthError
        If the input array have even length.
    """
    if len(signal) % 2 == 0:
        raise EvenLengthError


def verify_equal_length(*signals: npt.NDArray[Any] | xr.DataArray) -> None:
    """Raise an error if the signals have unequal length.

    Parameters
    ----------
    *signals : npt.NDArray[Any] | xr.DataArray
        The signals to compare

    Raises
    ------
    UnequalArrayLengthError
        If the input arrays differ in length.
    """
    first = signals[0]
    if any(x.size != first.size for x in signals[1:]):
        raise UnequalArrayLengthError


class RandomStateForcingGenerator(pf.ForcingGenerator):  # type: ignore[misc]
    """Generates a standard forcing, with uniformly distributed arrival times.

    The resulting process is therefore a Poisson process. Amplitude and
    duration distributions can be customized.
    """

    def __init__(self, seed: int) -> None:
        self._random_state = np.random.default_rng(seed)
        self._amplitude_distribution: Callable[[int], npt.NDArray[Any]] | None = None
        self._duration_distribution: Callable[[int], npt.NDArray[Any]] | None = None

    def get_forcing(
        self, times: npt.NDArray[np.float64], waiting_time: float
    ) -> pf.Forcing:
        """Return the forcing object that the class implements."""
        total_pulses = int(max(times) / waiting_time)
        arrival_times = self._random_state.uniform(
            low=times[0], high=times[len(times) - 1], size=total_pulses
        )
        amplitudes = self._get_amplitudes(total_pulses)
        durations = self._get_durations(total_pulses)
        return pf.Forcing(total_pulses, arrival_times, amplitudes, durations)

    def set_amplitude_distribution(
        self,
        amplitude_distribution_function: Callable[[int], npt.NDArray[np.float64]],
    ) -> None:
        """Set the amplitude distribution."""
        self._amplitude_distribution = amplitude_distribution_function

    def set_duration_distribution(
        self, duration_distribution_function: Callable[[int], npt.NDArray[np.float64]]
    ) -> None:
        """Set the duration distribution."""
        self._duration_distribution = duration_distribution_function

    def _get_amplitudes(self, total_pulses: int) -> npt.NDArray[np.float64]:
        if self._amplitude_distribution is not None:
            return self._amplitude_distribution(total_pulses)
        return self._random_state.exponential(scale=1.0, size=total_pulses)

    def _get_durations(self, total_pulses: int) -> npt.NDArray[np.float64]:
        if self._duration_distribution is not None:
            return self._duration_distribution(total_pulses)
        return np.ones(total_pulses)


class TimeSeriesModel:
    """Generate a signal without noise, and with additive and dynamical noise."""

    def __init__(
        self,
        seed: int | None = None,
        total_pulses: int | None = None,
        gamma: float | None = None,
        dt: float | None = None,
    ) -> None:
        self._seed = seed
        self._total_pulses = int(1e3) if total_pulses is None else total_pulses
        self._gamma = 1e-1 if gamma is None else gamma
        self._dt = 1e-2 if dt is None else dt

    def __call__(
        self,
        total_pulses: int,
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
                    "epsilon": epsilon_dynamic,
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
                    "epsilon": epsilon_additive,
                },
            )
        if (
            epsilon_additive is not None
            and epsilon_dynamic is not None
            and epsilon_additive != epsilon_dynamic
        ):
            print(
                "I will only keep track of one epsilon value, with a preference to the additive."
            )
        elif epsilon_additive is not None:
            self.ds.attrs["epsilon"] = epsilon_additive
        elif epsilon_dynamic is not None:
            self.ds.attrs["epsilon"] = epsilon_dynamic
        return self.ds

    def _create_model(self) -> None:
        self._model = pm.PointModel(
            waiting_time=1 / self._gamma,
            total_duration=self._total_pulses / self._gamma,
            dt=self._dt,
        )
        if self._seed is not None:
            self._model.set_custom_forcing_generator(
                RandomStateForcingGenerator(self._seed)
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
    def _make_odd_length(arr: npt.NDArray[Any]) -> npt.NDArray[Any]:
        if len(arr) % 2:
            return arr
        return arr[:-1]

    def _add_noise(
        self, epsilon: float, version: Literal["additive", "dynamic"]
    ) -> npt.NDArray[Any]:
        self._model.add_noise(epsilon, noise_type=version, seed=self._seed)
        return self._reuse_realization()

    def _reuse_realization(self) -> npt.NDArray[Any]:
        result: npt.NDArray[Any] = np.zeros_like(self.ds["signal"].data)

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
        data: npt.NDArray[Any], tau: npt.NDArray[Any], iterlist: list[int]
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
    def dress_response_pulse_error(data: npt.NDArray[Any]) -> xr.DataArray:
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
        data: npt.NDArray[Any],
        time: npt.NDArray[Any],
        ratio: fractions.Fraction,
        desc: str | None = None,
    ) -> xr.DataArray:
        """Dress an upsampled array named `name` with a time dimension."""
        _a = {"long_name": f"{name}, dense", "sample_factor": str(ratio)}
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
        data: npt.NDArray[Any],
        time: npt.NDArray[Any],
        ratio: fractions.Fraction,
        desc: str | None = None,
    ) -> xr.DataArray:
        """Dress a downsampled array named `name` with a time dimension.

        Parameters
        ----------
        name : str
            The name given to the array and used in labels. It is recommended to specify
            capital first letter.
        data : npt.NDArray[Any]
            The data to record.
        time : npt.NDArray[Any]
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
        self, arr: npt.NDArray[Any], times: npt.NDArray[Any], ratio: fractions.Fraction
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
        orig_arr: npt.NDArray[Any],
        orig_time: npt.NDArray[Any],
        ratio: fractions.Fraction,
    ) -> tuple[npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[Any]]:
        """Sample an array at a given ratio.

        Parameters
        ----------
        orig_arr : npt.NDArray[Any]
            The input array that should be downsampled.
        orig_time : npt.NDArray[Any]
            The time axis of the input array that should be downsampled.
        ratio : fractions.Fraction
            The ratio of the downsampling to original array.

        Returns
        -------
        npt.NDArray[Any]
            The downsampled input array.
        npt.NDArray[Any]
            The time axis of the downsampled input array.
        npt.NDArray[Any]
            The upsampling of the input array with equal size to the input array.
        """

        def _reshape(arr: npt.ArrayLike, ratio: fractions.Fraction) -> npt.NDArray[Any]:
            _arr = np.asarray(arr)
            rows = int(np.ceil(len(_arr) / ratio.denominator))
            return np.pad(
                _arr.astype(float),
                (0, ratio.denominator * rows - _arr.size),
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

    @staticmethod
    def _extend(arr: npt.ArrayLike, ratio: fractions.Fraction) -> npt.NDArray[Any]:
        """Extend the array so that its length is perfectly divisible by the sampling."""
        _arr = np.asarray(arr)
        return np.pad(
            _arr, (0, len(_arr) % ratio.denominator), constant_values=_arr[-1]
        )

    @staticmethod
    def _downsample_lossy(
        arr: npt.ArrayLike, index: int, rate: int
    ) -> npt.NDArray[Any]:
        _arr = np.asarray(arr)
        # First, extend so that the length of the array is divisible by
        # `ratio.denominator`. Then slice it.
        _arr = np.pad(_arr, (0, len(_arr) % rate), constant_values=_arr[-1])
        return _arr[index::rate]

    @staticmethod
    def _upsample_nth(
        arr: npt.ArrayLike, index: int, rate: int, target_length: int
    ) -> npt.NDArray[Any]:
        _arr = np.asarray(arr)
        full = np.zeros(target_length + (target_length % rate) - 1)
        clip_length = len(full[index::rate])
        while len(_arr) < clip_length:
            _arr = np.concatenate((_arr, [0]))
        full[index::rate] = _arr[:clip_length]
        return full[:target_length]

    @staticmethod
    def _downsample_lossless(
        arr: npt.ArrayLike, index: int, rate: int
    ) -> npt.NDArray[Any]:
        _arr = np.cumsum(arr)
        _arr = np.pad(_arr, (0, len(_arr) % rate), constant_values=_arr[-1])
        return inverse_cumsum(_arr[index::rate])

    @staticmethod
    def _upsample_repeat(
        arr: npt.ArrayLike, rate: int, target_length: int
    ) -> npt.NDArray[Any]:
        _arr = np.repeat(arr, rate)
        while not len(out := _arr[:target_length]) % 2:
            _arr = np.pad(_arr, (0, 1), constant_values=_arr[-1])
        return out

    def sample_lossy_nth(
        self,
        ratio: fractions.Fraction,
        down_index: int = -1,
        up_index: int = -1,
    ) -> tuple[xr.DataArray, xr.DataArray]:
        """Keep every `ratio`-th value in the forcing."""
        if ratio >= 1:
            raise ValueError
        down_index %= ratio.denominator
        up_index %= ratio.denominator
        _a = self._arr.data
        _t = self._arr.time.data
        # The forcing is just every `ratio.denominator`-th index, starting from
        # `down_index`.
        frc = self._downsample_lossy(_a, down_index, ratio.denominator)
        frc_time = self._downsample_lossy(_t, down_index, ratio.denominator)
        # The upsampled version is assumed to be filled with zeros where the
        # downsampling occurred. We add an initial `up_index` number of zeros at the
        # start to account for downsampling placement.
        full = self._upsample_nth(frc, up_index, ratio.denominator, len(_a))
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

    def sample_lossy_repeat(
        self,
        ratio: fractions.Fraction,
        down_index: int = -1,
    ) -> tuple[xr.DataArray, xr.DataArray]:
        """Keep every `ratio`-th value in the forcing."""
        if ratio >= 1:
            raise ValueError
        if ratio.numerator != 1:
            raise ValueError
        down_index %= ratio.denominator
        _a = self._arr.data
        _t = self._arr.time.data
        # The forcing is just every `ratio.denominator`-th index, starting from
        # `down_index`.
        frc = self._downsample_lossy(_a, down_index, ratio.denominator)
        frc_time = self._downsample_lossy(_t, down_index, ratio.denominator)
        # The upsampled version is just the downsampled with repeated indices.
        full = self._upsample_repeat(frc, ratio.denominator, len(_a))
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

    def sample_lossless_nth(
        self,
        ratio: fractions.Fraction = fractions.Fraction(1, 10),
        down_index: int = -1,
        up_index: int = -1,
    ) -> tuple[xr.DataArray, xr.DataArray]:
        """Keep every `ratio`-th value in the forcing, without loss.

        This procedure places the sum of all removed events to the first possible
        sampled location.

        Parameters
        ----------
        ratio : fractions.Fraction
            The number of kept indices per index group, thus for every
            `ratio.denominator` indices, we keep the first `ratio.numerator` elements.
            Only a numerator of 1 is currently supported.
        down_index : int
            When downsampling, we need to deicide where the downsampled point is placed
            (for example, from daily to monthly, is it placed on Jan 1. or Jan 31.). If
            we let the downsampled index be placed at the first time within a group, the
            `group_index` would be zero. Similarly, if we let the downsampled point be
            placed at the last element, the `group_index` would be `-1` (default).
        up_index : int
            When zero-padding during upsampling, this decides if the data point should
            be at the first index, last index (default) or any other index in the
            padding region.

        Returns
        -------
        xr.DataArray
            The downsampled forcing array.
        xr.DataArray
            All the upsampled forcing arrays.

        Raises
        ------
        NotImplementedError
            If `ratio.numerator` is not one.
        ValueError
            If the ratio is greater than one.
        """
        if ratio.numerator != 1:
            raise NotImplementedError
        if ratio >= 1:
            raise ValueError
        down_index %= ratio.denominator
        up_index %= ratio.denominator
        _a = self._arr.data
        _t = self._arr.time.data
        frc = self._downsample_lossless(_a, down_index, ratio.denominator)
        frc_time = self._downsample_lossy(_t, down_index, ratio.denominator)
        full = self._upsample_nth(frc, up_index, ratio.denominator, len(_a))
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

    def sample_lossless_repeat(
        self,
        ratio: fractions.Fraction = fractions.Fraction(1, 10),
        down_index: int = -1,
    ) -> tuple[xr.DataArray, xr.DataArray]:
        """Keep every `ratio`-th value in the forcing, without loss.

        This procedure places the sum of all removed events to the first possible
        sampled location.

        Parameters
        ----------
        ratio : fractions.Fraction
            The number of kept indices per index group, thus for every
            `ratio.denominator` indices, we keep the first `ratio.numerator` elements.
            Only a numerator of 1 is currently supported.
        down_index : int
            When downsampling, we need to deicide where the downsampled point is placed
            (for example, from daily to monthly, is it placed on Jan 1. or Jan 31.). If
            we let the downsampled index be placed at the first time within a group,
            the `group_index` would be zero (default). Similarly, if we let the
            downsampled point be placed at the last element, the `group_index` should
            be `ratio.denominator - 1`.

        Returns
        -------
        xr.DataArray
            The downsampled forcing array.
        xr.DataArray
            All the upsampled forcing arrays.

        Raises
        ------
        NotImplementedError
            If `ratio.numerator` is not one.
        ValueError
            If the ratio is greater than one.
        """
        if ratio.numerator != 1:
            raise NotImplementedError
        if ratio >= 1:
            raise ValueError
        down_index %= ratio.denominator
        _a = self._arr.data
        _t = self._arr.time.data
        frc = self._downsample_lossless(_a, down_index, ratio.denominator)
        frc_time = self._downsample_lossy(_t, down_index, ratio.denominator)
        full = self._upsample_repeat(frc, ratio.denominator, len(_a))
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

    def fourier_transform(
        self, ratio: fractions.Fraction = fractions.Fraction(1, 10)
    ) -> tuple[xr.DataArray, xr.DataArray]:
        """Keep every `ratio`-th value in the forcing by low-pass filtering."""
        if ratio >= 1:
            raise ValueError
        _a = self._arr.data
        _t = self._arr.time.data
        # First, reshape so that the length of a row is `ratio.denominator`
        frc, frc_time, reshaped = self._sample_at_ratio(_a, _t, ratio)
        fourier = np.fft.rfft(_a)
        fourier[-100:] = 0
        # np.fft.irfft computes the inverse of the one-dimensional discrete
        # Fourier Transform for real input, as computed by rfft.
        # Ensures same length as the time array that hasn't been altered.
        full = np.fft.irfft(fourier, n=len(self._arr.data))
        # _a = np.cumsum(self._arr.data)
        # _t = self._arr.time.data
        # frc, frc_time, reshaped = self._sample_at_ratio(_a, _t, ratio)
        # frc = inverse_cumsum(frc)
        # # The upsampled version is assumed to be filled with zeros where the
        # # downsampling occurred.
        # width = 2 * ratio.numerator - reshaped.shape[1]
        # if ratio.numerator == 1:
        #     reshaped[:, 1:] = (
        #         np.ones_like(reshaped[:, 1:]) * np.atleast_2d(reshaped[:, 0]).T
        #     )
        # else:
        #     reshaped[:, ratio.numerator :] = reshaped[:, width : ratio.numerator]
        # full = reshaped.flatten()[: len(_a)]
        # full = inverse_cumsum(full)
        verify_equal_length(full, self._arr)
        down = Wardrobe.dress_a_downsampled(
            "Forcing",
            frc,
            frc_time,
            ratio,
            desc='Raw "choose every n" downsampling',
        )
        self._create_corresponding_upsampled(full, self._arr.time.data, ratio)
        return down, self.arr

    def lowpass(
        self, ratio: fractions.Fraction = fractions.Fraction(1, 10)
    ) -> tuple[xr.DataArray, xr.DataArray]:
        """Keep every `ratio`-th value in the forcing by low-pass filtering."""
        if ratio >= 1:
            raise ValueError

        def butter_lowpass(
            cutoff: float, fs: float, order: int = 5
        ) -> returns.maybe.Maybe[tuple[npt.NDArray[Any], npt.NDArray[Any]]]:
            b, a = scipy.signal.butter(order, cutoff, fs=fs, btype="low", analog=False)
            match b, a:
                case (np.ndarray(), np.ndarray()):
                    return returns.maybe.Some((b, a))
                case _:
                    return returns.maybe.Nothing

        def butter_lowpass_filter(
            data: npt.ArrayLike, cutoff: float, fs: float, order: int = 5
        ) -> returns.maybe.Maybe[npt.NDArray[Any]]:
            match butter_lowpass(cutoff, fs, order=order):
                case returns.maybe.Some(out):
                    b, a = out
                    c = scipy.signal.lfilter(b, a, data)
                    return returns.maybe.Some(c)
                case returns.maybe.Maybe.empty:
                    return returns.maybe.Nothing
                case _:
                    return returns.maybe.Nothing

        _a = self._arr.data
        _t = self._arr.time.data
        cutoff = ratio.numerator
        fs = ratio.denominator * 2
        order = 1
        # b, a = butter_lowpass(cutoff, fs, order=order).unwrap()
        # w, h = scipy.signal.freqz(b, a, fs=fs, worN=8000)
        # plt.figure()
        # plt.plot(w, np.abs(h), "b")
        # plt.plot(cutoff, 0.5 * np.sqrt(2), "ko")
        # plt.axvline(cutoff, color="k")
        # plt.xlim(0, 0.5 * fs)
        # plt.title("Lowpass Filter Frequency Response")
        # plt.xlabel("Frequency [Hz]")
        # plt.grid()
        # plt.show()
        full = butter_lowpass_filter(_a, cutoff=cutoff, fs=fs, order=order).unwrap()
        frc, frc_time, _ = self._sample_at_ratio(_a, _t, ratio)
        # fourier = np.fft.rfft(frc)
        # full = np.fft.irfft(fourier, n=len(self._arr.data))
        verify_equal_length(full, self._arr)
        down = Wardrobe.dress_a_downsampled(
            "Forcing",
            frc,
            frc_time,
            ratio,
            desc='Raw "choose every n" downsampling',
        )
        self._create_corresponding_upsampled(full, self._arr.time.data, ratio)
        return down, self.arr
