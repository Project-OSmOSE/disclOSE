"""Data containers and plotting accessors for APLOSE detection data.

``DetectionData`` handles, analyzes, and visualizes APLOSE-formatted detections.

>>> from pathlib import Path
>>> import matplotlib.pyplot as plt
>>> from pandas import Timedelta
>>> from disclose.dataclass.detection_data import DetectionData
>>> data = DetectionData.from_dict({"detection_file": Path("results.csv")})
>>> data.coordinates = (48.39, -4.49)
>>> fig, ax = plt.subplots()
>>> data.plot.histogram("annotator", "label", bin_size=Timedelta("1D"), ax=ax)
"""

from __future__ import annotations

import logging
from copy import copy as _copy
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Self

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from pandas import (
    DataFrame,
    MultiIndex,
    Timedelta,
    concat,
    date_range,
    to_datetime,
)
from pandas.tseries import offsets

from disclose.dataclass.detection_config import DetectionConfig
from disclose.dataclass.recording_effort import RecordingEffort
from disclose.utils.core import (
    add_season_period,
    get_count,
    get_labels_and_annotators,
    get_sun_times,
)
from disclose.utils.filtering import (
    get_annotators,
    get_dataset,
    get_labels,
    get_max_time,
    get_timezone,
    load_detections,
)
from disclose.utils.metric import (
    Classification,
    ConfusionMatrix,
    Scores,
    detection_perf,
)
from disclose.utils.visualisation import (
    add_sunrise_sunset,
    arc,
    heatmap,
    histo,
    overview,
    plot_annotator_agreement,
    polar,
    scatter,
    shade_no_effort,
    timeline,
    update_legend,
    set_plot_title,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from datetime import tzinfo

    from matplotlib.axes import Axes
    from pandas import Series, Timestamp
    from pandas.tseries.offsets import BaseOffset


logger = logging.getLogger(__name__)
plt.ioff()

APLOSE_COLUMNS = [
    "dataset",
    "filename",
    "start_datetime",
    "end_datetime",
    "start_time",
    "end_time",
    "label",
    "annotator",
]

_OFFSET_LOCATORS: dict[tuple[type, ...], Callable[[Any], mdates.DateLocator]] = {
    (offsets.YearEnd, offsets.YearBegin): lambda o: mdates.YearLocator(base=o.n),
    (
        offsets.MonthEnd,
        offsets.MonthBegin,
        offsets.BusinessMonthEnd,
        offsets.BusinessMonthBegin,
    ): lambda o: mdates.MonthLocator(interval=o.n),
    (offsets.Week,): lambda o: mdates.WeekdayLocator(
        byweekday=o.weekday,
        interval=o.n,
    ),
    (offsets.Day,): lambda o: mdates.DayLocator(interval=o.n),
    (offsets.Hour,): lambda o: mdates.HourLocator(interval=o.n),
    (offsets.Minute,): lambda o: mdates.MinuteLocator(interval=o.n),
    (offsets.Second,): lambda o: mdates.SecondLocator(interval=o.n),
}


def _get_locator_from_offset(
    offset: int | Timedelta | BaseOffset,
) -> mdates.DateLocator:
    """Map a pandas offset to the appropriate matplotlib DateLocator.

    Parameters
    ----------
    offset: int | Timedelta | BaseOffset
        An ``int`` is interpreted as a number of seconds.
        A ``Timedelta`` is mapped to the coarsest locator (hour, minute, second)
        that divides it exactly.

    Raises
    ------
    ValueError
        If the offset type is not supported.

    """
    if isinstance(offset, int):
        return mdates.SecondLocator(interval=offset)

    if isinstance(offset, Timedelta):
        total_seconds = int(offset.total_seconds())
        if total_seconds % 3600 == 0:
            return mdates.HourLocator(interval=total_seconds // 3600)
        if total_seconds % 60 == 0:
            return mdates.MinuteLocator(interval=total_seconds // 60)
        return mdates.SecondLocator(interval=total_seconds)

    for offset_classes, locator_fn in _OFFSET_LOCATORS.items():
        if isinstance(offset, offset_classes):
            return locator_fn(offset)

    msg = f"Unsupported offset type: {type(offset)}"
    raise ValueError(msg)


def empty_dataframe() -> DataFrame:
    """Return an empty DataFrame with the mandatory APLOSE columns."""
    return DataFrame(columns=APLOSE_COLUMNS)


@dataclass(eq=False)
class DetectionData:
    """Handle APLOSE-formatted detection data.

    Attributes
    ----------
    df: DataFrame
        APLOSE-formatted detections.
    config: DetectionConfig
        Configuration the data was loaded with.
    lat, lon: float | None
        Position of the recorder in decimal degrees.
        Required by the plots that display sunrise/sunset or seasons.
        See also the ``coordinates`` property.

    Notes
    -----
    ``reshape`` and ``change_tz`` modify the instance in place.
    Use ``copy`` to work on an independent instance.
    Equality is identity-based (comparing DataFrames is ambiguous).

    Examples
    --------
    >>> from pathlib import Path
    >>> import matplotlib.pyplot as plt
    >>> from pandas import Timedelta
    >>> data = DetectionData.from_dict({"detection_file": Path("results.csv")})
    >>> data.coordinates = (48.39, -4.49)
    >>> data.change_tz("Europe/Paris")
    >>> perf = data.detection_perf(("ann1", "ann2"), ("lbl1", "lbl1"))
    >>> _, ax = plt.subplots()
    >>> data.plot.histogram("ann1", "lbl1", Timedelta("1D"), ax=ax)
    >>> data.plot.feature(ax, season=True)

    """

    df: DataFrame = field(default_factory=empty_dataframe)
    config: DetectionConfig = field(default_factory=DetectionConfig.empty)

    lat: float | None = None
    lon: float | None = None

    _start_datetime: Timestamp | None = field(default=None, init=False, repr=False)
    _end_datetime: Timestamp | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Check that the DataFrame is APLOSE-formatted."""
        missing = [col for col in APLOSE_COLUMNS if col not in self.df.columns]
        if missing:
            msg = f"DataFrame is not APLOSE-formatted, missing columns: {missing}"
            raise ValueError(msg)

    def __str__(self) -> str:
        """Return a summary of the data bounds, annotators, labels, and datasets."""
        return (
            f"start_datetime: {self.start_datetime}\n"
            f"end_datetime: {self.end_datetime}\n"
            f"annotator: {self.annotator}\n"
            f"label: {self.label}\n"
            f"dataset: {self.dataset}"
        )

    def __repr__(self) -> str:
        """Return the same summary as :meth:`__str__`."""
        return self.__str__()

    # %% properties

    @property
    def annotator(self) -> str | list[str] | None:
        """Annotator(s) of the data (a ``str`` if there is only one)."""
        return get_annotators(self.df)

    @property
    def label(self) -> str | list[str] | None:
        """Label(s) of the data (a ``str`` if there is only one)."""
        return get_labels(self.df)

    @property
    def dataset(self) -> str | list[str] | None:
        """Dataset(s) of the data (a ``str`` if there is only one)."""
        return get_dataset(self.df)

    @property
    def start_datetime(self) -> Timestamp | None:
        """Beginning of the data.

        This is the bound set by ``reshape`` if any, otherwise the earliest of
        the configured start and the first detection.
        """
        if self._start_datetime is not None:
            return self._start_datetime
        return self._bound("start_datetime", self.config.start_datetime, earliest=True)

    @property
    def end_datetime(self) -> Timestamp | None:
        """End of the data.

        This is the bound set by ``reshape`` if any, otherwise the latest of
        the configured end and the last detection.
        """
        if self._end_datetime is not None:
            return self._end_datetime
        return self._bound("end_datetime", self.config.end_datetime, earliest=False)

    @property
    def coordinates(self) -> tuple[float | None, float | None]:
        """Coordinates (lat, lon) of the data."""
        return self.lat, self.lon

    @coordinates.setter
    def coordinates(self, value: tuple[float, float]) -> None:
        if not isinstance(value, tuple) or len(value) != 2:  # ruff: ignore[magic-value-comparison]
            msg = "Coordinates must be a tuple of two floats: (lat, lon)."
            raise ValueError(msg)
        self.lat, self.lon = value

    @property
    def plot(self) -> PlotAccessor:
        """Plotting namespace, e.g. ``data.plot.histogram(...)``."""
        return PlotAccessor(self)

    # %% private helpers

    def _localize(self, timestamp: Timestamp) -> Timestamp:
        """Localize a naive timestamp in the timezone of the data.

        Timezone-aware timestamps are returned untouched.
        """
        if timestamp.tz is not None or self.df.empty:
            return timestamp
        tz = get_timezone(self.df)
        if isinstance(tz, list):
            msg = (
                "Cannot localize a naive timestamp: the data has several timezones. "
                "Call `change_tz` first or provide a timezone-aware timestamp."
            )
            raise TypeError(msg)
        return timestamp.tz_localize(tz)

    def _bound(
        self,
        column: Literal["start_datetime", "end_datetime"],
        config_value: Timestamp | None,
        *,
        earliest: bool,
    ) -> Timestamp | None:
        """Return the earliest/latest between the config value and the data."""
        candidates = []
        if config_value is not None:
            candidates.append(self._localize(config_value))
        if not self.df.empty:
            series = self.df[column]
            candidates.append(series.min() if earliest else series.max())
        if not candidates:
            return None
        return min(candidates) if earliest else max(candidates)

    @staticmethod
    def _as_pairs(
        annotator: str | list[str],
        label: str | list[str],
    ) -> list[tuple[str, str]]:
        """Pair annotators and labels, broadcasting a lone ``str`` if needed."""
        annotators = (
            [annotator] * (1 if isinstance(label, str) else len(label))
            if isinstance(annotator, str)
            else list(annotator)
        )
        labels = (
            [label] * (1 if isinstance(annotator, str) else len(annotator))
            if isinstance(label, str)
            else list(label)
        )
        if len(annotators) != len(labels):
            msg = (
                f"Length of annotator ({len(annotators)}) and"
                f" label ({len(labels)}) must match."
            )
            raise ValueError(msg)
        return list(zip(annotators, labels, strict=True))

    # %% constructors

    @classmethod
    def from_config(
        cls,
        config: DetectionConfig,
        *,
        coordinates: tuple[float, float] | None = None,
    ) -> Self:
        """Load, filter and reshape detections from a configuration.

        Parameters
        ----------
        config: DetectionConfig
            Loading and filtering configuration.
        coordinates: tuple[float, float] | None
            Optional (lat, lon) of the recorder.

        """
        obj = cls(df=load_detections(config), config=config)
        if coordinates is not None:
            obj.coordinates = coordinates
        return obj.reshape(config.start_datetime, config.end_datetime)

    @classmethod
    def from_dict(
        cls,
        config: dict | list[dict],
        *,
        merge: bool = True,
        coordinates: tuple[float, float] | None = None,
    ) -> Self | list[Self]:
        """Build one or several instances from configuration dictionaries.

        Parameters
        ----------
        config: dict | list[dict]
            One configuration, or a list of them.
        merge: bool
            Only used if ``config`` is a list.
            If ``True``, the instances are merged into a single one,
            otherwise the list of instances is returned.
        coordinates: tuple[float, float] | None
            Optional (lat, lon) of the recorder, applied to every instance.

        Returns
        -------
        DetectionData | list[DetectionData]
            A single instance, or a list when ``config`` is a list and
            ``merge`` is ``False``.

        """
        configs = DetectionConfig.from_dict(config)

        if isinstance(configs, DetectionConfig):
            return cls.from_config(configs, coordinates=coordinates)

        datasets = [cls.from_config(conf, coordinates=coordinates) for conf in configs]
        return cls.merge(datasets) if merge else datasets

    @classmethod
    def merge(cls, datasets: list[DetectionData]) -> Self:
        """Merge several instances into one.

        Detections are concatenated and sorted.
        If the instances don't share a timezone, all timestamps are converted to UTC.
        Coordinates are kept if all the instances that have some agree.

        Raises
        ------
        ValueError
            If ``datasets`` is empty.

        """
        if not datasets:
            msg = "No DataAplose to merge."
            raise ValueError(msg)

        df = (
            concat([d.df for d in datasets], ignore_index=True)
            .sort_values(["start_datetime", "end_datetime", "annotator", "label"])
            .reset_index(drop=True)
        )
        obj = cls(df=df, config=DetectionConfig.merge([d.config for d in datasets]))

        coordinates = {d.coordinates for d in datasets if None not in d.coordinates}
        if len(coordinates) == 1:
            obj.coordinates = coordinates.pop()

        if isinstance(get_timezone(df), list):
            obj.change_tz("UTC")
            logger.info(
                "Several timezones found in DataFrame, all timestamps are converted "
                "to UTC.",
            )

        return obj

    # %% transformations

    def copy(self) -> Self:
        """Return an independent copy (the DataFrame is copied)."""
        new = replace(self, df=self.df.copy(), config=_copy(self.config))
        new._start_datetime = self._start_datetime  # ruff: ignore[private-member-access]
        new._end_datetime = self._end_datetime  # ruff: ignore[private-member-access]
        return new

    def reshape(
        self,
        start_datetime: Timestamp | None = None,
        end_datetime: Timestamp | None = None,
    ) -> Self:
        """Restrict the data to a new beginning and/or end, in place.

        Parameters
        ----------
        start_datetime, end_datetime: Timestamp | None
            New bounds. Naive timestamps are localized in the timezone of the data.
            A ``None`` bound is taken from the data.

        Returns
        -------
        The instance itself, to allow chaining.

        Raises
        ------
        ValueError
            If the beginning is not anterior to the end.

        """
        if start_datetime is None and end_datetime is None:
            logger.debug("No begin/end timestamps provided, nothing to reshape.")
            return self
        if self.df.empty:
            logger.debug("Empty DataFrame, nothing to reshape.")
            return self

        start = (
            self._localize(start_datetime)
            if start_datetime is not None
            else self.df["start_datetime"].min()
        )
        end = (
            self._localize(end_datetime)
            if end_datetime is not None
            else self.df["end_datetime"].max()
        )
        if start >= end:
            msg = "Begin timestamp is not anterior than end timestamp."
            raise ValueError(msg)

        self._start_datetime, self._end_datetime = start, end
        self.df = self.df[
            (self.df["start_datetime"] >= start) & (self.df["end_datetime"] <= end)
        ]
        return self

    def change_tz(self, tz: str | tzinfo) -> None:
        """Convert all timestamps to another timezone, in place.

        Parameters
        ----------
        tz: str | tzinfo
            Target timezone, e.g. ``"UTC"``, ``"Europe/Paris"`` or a ``pytz`` object.

        Examples
        --------
        >>> data.change_tz("Europe/Paris")
        >>> data.change_tz(pytz.timezone("Etc/GMT-2"))

        """
        self.df = self.df.assign(
            **{
                col: to_datetime(self.df[col], utc=True).dt.tz_convert(tz)
                for col in ("start_datetime", "end_datetime")
            },
        )
        start, end = self.start_datetime, self.end_datetime
        self._start_datetime = None if start is None else start.tz_convert(tz)
        self._end_datetime = None if end is None else end.tz_convert(tz)

    def filter_df(
        self,
        annotator: str | list[str],
        label: str | list[str],
    ) -> DataFrame:
        """Filter the detections on annotator/label pairs.

        Annotators and labels are paired element-wise (not as a cross-product):
        ``(["a1", "a2"], ["l1", "l2"])`` selects ``(a1, l1)`` and ``(a2, l2)``.
        A lone ``str`` is broadcast to the length of the other argument.

        Parameters
        ----------
        annotator: str | list[str]
            The annotator or list of annotators to filter.
        label: str | list[str]
            The label or list of labels to filter.

        Returns
        -------
        The filtered DataFrame, with a reset index.

        Raises
        ------
        ValueError
            If annotator or label are unknown, if the lists differ in length,
            or if a pair has no detection.

        """
        pairs = self._as_pairs(annotator, label)
        known_annotators = set(self.df["annotator"])
        known_labels = set(self.df["label"])

        for ant, lbl in pairs:
            if ant not in known_annotators:
                msg = f'Annotator "{ant}" not in APLOSE DataFrame'
                raise ValueError(msg)
            if lbl not in known_labels:
                msg = f'Label "{lbl}" not in APLOSE DataFrame'
                raise ValueError(msg)
            if not ((self.df["annotator"] == ant) & (self.df["label"] == lbl)).any():
                msg = (
                    f"DataFrame with annotator '{ant}' / label '{lbl}'"
                    f" contains no detection."
                )
                raise ValueError(msg)

        mask = MultiIndex.from_frame(self.df[["annotator", "label"]]).isin(pairs)
        return self.df.loc[mask].reset_index(drop=True)

    def filter(
        self,
        annotator: str | list[str],
        label: str | list[str],
    ) -> Self:
        """Return a new ``DetectionData`` containing only selected detections.

        Parameters
        ----------
        annotator: str | list[str]
            Annotator(s) to select.
        label: str | list[str]
            Label(s) to select.

        Returns
        -------
        DetectionData
            An independent filtered instance preserving the original
            configuration, coordinates, and datetime bounds.

        """
        filtered = self.copy()
        filtered.df = self.filter_df(annotator, label)
        return filtered

    # %% analysis

    def overview(self, annotator: str | list[str]) -> DataFrame:
        """Return a table of detection counts by label and annotator.

        If ``annotator`` is provided, the table is restricted to the selected
        annotator or annotators.
        """
        return overview(df=self.df, annotator=annotator)

    def detection_perf(
        self,
        annotator: tuple[str, str] | list[str],
        label: tuple[str, str] | list[str],
    ) -> tuple[Scores, ConfusionMatrix, Classification]:
        """Compute detection performance metrics.

        Performances are computed for the second annotator/label pair
        in regard to the first (reference) pair.

        Parameters
        ----------
        annotator: tuple[str, str] | list[str]
            The two annotators to compare. The first is the reference.
        label: tuple[str, str] | list[str]
            The two labels to compare. The first is the reference.

        Returns
        -------
        scores: Scores
            Precision, recall and f-score.
        confusion_matrix: ConfusionMatrix
            Count of true/false positives/negatives and errors.
        classification: Classification
            Timestamps of each of these categories.

        """
        df_filtered = self.filter_df(annotator, label)

        end_times = df_filtered["end_time"].dropna()
        if not end_times.empty and end_times.min() != end_times.max():
            msg = "Multiple time bins detected in DataFrame."
            raise ValueError(msg)
        timebin = Timedelta(df_filtered["end_time"].iloc[0], "s")

        effort = (
            RecordingEffort.from_config(config=self.config, bin_size=timebin)
            if self.config.recording_file
            else None
        )

        return detection_perf(
            df=df_filtered,
            ref=(annotator[0], label[0]),
            time=date_range(self.start_datetime, self.end_datetime, freq=timebin),
            effort=effort,
        )


class PlotAccessor:
    """Plotting namespace of a ``DetectionData``, available as ``data.plot``.

    Plot methods accept an optional keyword-only ``ax`` and create one when absent.
    Use `set_ax` to format the time axis and `feature` to add overlays.
    The accessor is callable, like in pandas: ``data.plot("polar", ...)``
    is the same as ``data.plot.polar(...)``.
    """

    _KINDS: ClassVar[frozenset[str]] = frozenset(
        {"histogram", "polar", "heatmap", "scatter", "timeline", "agreement"},
    )

    def __init__(self, data: DetectionData) -> None:
        """Bind the accessor to a ``DetectionData`` instance."""
        self._data = data

    def __call__(self, kind: str, *args: Any, **kwargs: Any) -> Any:  # ruff: ignore[any-type]
        """Dispatch to the plotting method named ``kind``."""
        if kind not in self._KINDS:
            msg = f"Unknown plot kind '{kind}', expected one of {sorted(self._KINDS)}."
            raise ValueError(msg)
        return getattr(self, kind)(*args, **kwargs)

    # %% helpers

    @staticmethod
    def _get_ax(ax: Axes | None = None, *, polar: bool = False) -> Axes:
        """Return the provided Axes or create one with the requested projection."""
        if ax is not None:
            return ax
        _, ax = plt.subplots(subplot_kw={"projection": "polar"} if polar else None)
        return ax

    @staticmethod
    def _tight_layout(ax: Axes | None) -> Axes:
        """Adjust the figure layout after plotting, preserving the Axes."""
        if ax is not None:
            ax.figure.tight_layout()
        return ax

    def _coordinates(
        self,
        override: tuple[float, float] | None,
        *,
        required: bool,
    ) -> tuple[float | None, float | None]:
        """Return the coordinates to use, failing early if they are required."""
        coordinates = override if override is not None else self._data.coordinates
        if required and None in coordinates:
            msg = (
                "Coordinates are required for this feature: set `data.coordinates = "
                "(lat, lon)` or pass `coordinates=(lat, lon)`."
            )
            raise ValueError(msg)
        return coordinates

    def set_ax(
        self,
        ax: Axes,
        x_ticks_res: Timedelta | BaseOffset | None = None,
        date_format: str | None = "%Y-%m-%d",
    ) -> Axes:
        """Configure a Matplotlib axis for a time-based plot.

        Parameters
        ----------
        ax: Axes
            The Axes object to configure.
        x_ticks_res: int | Timedelta | BaseOffset | None
            Optional resolution of the x-axis major ticks. Integers are treated
            as seconds; ``None`` leaves the locator unchanged.
        date_format: str
            Date format of the x-axis tick labels (e.g. ``"%b"``, ``"%Y-%m-%d %H:%M"``).

        Returns
        -------
        The configured Axes.

        """
        start, end = self._data.start_datetime, self._data.end_datetime

        ax.set_xlim(start, end)

        if x_ticks_res:
            ax.xaxis.set_major_locator(
                _get_locator_from_offset(offset=x_ticks_res),
            )

        ax.xaxis.set_major_formatter(
            mdates.DateFormatter(
                fmt=date_format,
                tz=None if start is None else start.tz,
            ),
        )

        ax.grid(
            linestyle="--",
            linewidth=0.2,
            axis="both",
            zorder=1,
        )
        return ax

    def feature(
        self,
        ax: Axes,
        *,
        legend: bool = True,
        season: bool = False,
        rise_set: bool = False,
        effort: bool = False,
        bin_size: Timedelta | BaseOffset | None = None,
    ) -> Axes | None:
        """Add optional overlays to an existing plot.

        Call this method after creating a plot to add seasonal information,
        sunrise/sunset times, or recording-effort shading. Coordinates are read
        from ``data.coordinates`` and must be set before requesting ``season``
        or ``rise_set``. On polar plots, ``rise_set`` adds a day/night band;
        ``season`` and ``effort`` are not supported.

        Parameters
        ----------
        ax: Axes
            Matplotlib Axes to which the overlays are added.
        legend: bool
            Whether to display the legend after updating it with overlay entries.
        season: bool
            Whether to add a bar showing the seasons. Requires the recorder's
            latitude in ``data.coordinates``.
        rise_set: bool
            Whether to show sunrise and sunset (or a day/night band on a polar
            plot). Requires the recorder's latitude and longitude in
            ``data.coordinates``.
        effort: bool
            Whether to shade periods without recording effort. Requires a
            ``recording_file`` in the data configuration and is not supported
            on polar plots.
        bin_size: Timedelta | BaseOffset | None
            Resolution used to aggregate recording effort. Defaults to the
            detection time-bin duration.

        Returns
        -------
        The Axes with the requested overlays and adjusted figure layout.

        """
        data = self._data
        lat, lon = self._data.coordinates

        if ax.name == "polar":
            if season or effort:
                msg = "Season and effort features are not supported on polar plots."
                raise ValueError(msg)

        if season:
            if lat is None:
                msg = "Latitude must be provided."
                raise ValueError(msg)
            add_season_period(ax, northern=lat >= 0)

        if rise_set:
            if (lat or lon) is None:
                msg = "Coordinates must be provided."
                raise ValueError(msg)

            if ax.name == "polar":
                self._add_polar_rise_set(ax=ax)

            else:
                tz = get_timezone(data.df)
                if isinstance(tz, list):
                    msg = "Several timezones not supported."
                    raise ValueError(msg)
                add_sunrise_sunset(ax, lat, lon, tz)

        if effort:
            effort_bin_size = bin_size or get_max_time(data.df)
            if effort_bin_size is None:
                msg = "Detection data are required to show recording effort."
                raise ValueError(msg)
            observed = RecordingEffort.from_config(
                config=data.config,
                bin_size=effort_bin_size,
            )
            shade_no_effort(ax=ax, observed=observed)

        update_legend(ax)
        if not legend:
            legend_label = ax.get_legend()
            if legend_label:
                legend_label.remove()

        return self._tight_layout(ax)

    def _add_polar_rise_set(
        self,
        ax: Axes,
    ) -> None:
        """Draw the day/night band on a polar plot."""
        lat, lon = self._data.coordinates
        start, end = self._data.start_datetime, self._data.end_datetime
        sunrise, sunset = get_sun_times(start, end, lat, lon)
        sr_min, sr_max = min(sunrise), max(sunrise)
        ss_min, ss_max = min(sunset), max(sunset)
        arc(ax, 0, sr_min, "tab:blue")
        arc(ax, sr_min, sr_max, "grey")
        arc(ax, sr_max, ss_min, "gold")
        arc(ax, ss_min, ss_max, "grey")
        arc(ax, ss_max, 24, "tab:blue")

    # %% plots

    def histogram(  # ruff: ignore[too-many-arguments]
        self,
        annotator: str | list[str],
        label: str | list[str],
        bin_size: Timedelta | BaseOffset,
        *,
        ax: Axes | None = None,
        **kwargs,
    ) -> Axes | None:
        """Plot a histogram of detections.

        Parameters
        ----------
        ax: Axes | None
            Axes to plot on.
        annotator, label: str | list[str]
            Selection.
        bin_size: Timedelta | BaseOffset
            Width of the bars.

        """
        ax = self._get_ax(ax)
        data = self._data

        color = kwargs.pop("color", None)
        labels, annotators = get_labels_and_annotators(data.filter_df(annotator, label))

        total_rank = len(labels)

        for i, (ann, lbl) in enumerate(zip(annotators, labels, strict=True)):
            df_filtered = data.filter_df(ann, lbl)

            if df_filtered.empty:
                continue

            bin_res = get_max_time(df_filtered)

            if color:
                kwargs["color"] = color[i]

            if len(set(annotators)) == 1:
                kwargs["label"] = labels[i]
            elif len(set(labels)) == 1:
                kwargs["label"] = annotators[i]
            else:
                kwargs["label"] = f"{annotators[i]} - {labels[i]}"

            histo(
                df=get_count(df_filtered, bin_size),
                ax=ax,
                bin_size=bin_size,
                time_bin=bin_res,
                rank=i,
                total=total_rank,
                **kwargs,
            )

        set_plot_title(ax, annotators, labels)
        update_legend(ax)
        self.set_ax(ax)
        return self._tight_layout(ax)

    def polar(
        self,
        annotator: str | list[str],
        label: str | list[str],
        *,
        ax: Axes | None = None,
        **kwargs: Any,
    ) -> tuple[Axes, Series]:
        """Plot the mean proportion of detected bins per hour of the day.

        Parameters
        ----------
        ax: Axes | None
            Optional polar Axes; created with a polar projection when omitted.
        annotator, label: str | list[str]
            Selection, see :meth:`DetectionData.filter_df`.
        **kwargs
            Forwarded to ``ax.bar`` (color, alpha, edgecolor...).

        Returns
        -------
        The Axes and the mean proportion of detections for each hour of the day.

        """
        ax = self._get_ax(ax, polar=True)

        result = polar(
            df=self._data.filter_df(annotator, label),
            ax=ax,
            **kwargs,
        )

        self._tight_layout(ax)
        return result

    def heatmap(
        self,
        annotator: str | list[str],
        label: str | list[str],
        bin_size: Timedelta | BaseOffset,
        *,
        ax: Axes | None = None,
    ) -> Axes:
        """Plot a heatmap of detections (hour of day vs. time).

        Parameters
        ----------
        ax: Axes | None
            Optional Axes to plot on; created automatically when omitted.
        annotator, label: str | list[str]
            Selection.
        bin_size: Timedelta | BaseOffset
            Size of the cells. Must be >= 24h.

        """
        ax = self._get_ax(ax)
        data = self._data
        df_filtered = data.filter_df(annotator, label)

        heatmap(
            df=df_filtered,
            ax=ax,
            bin_size=bin_size,
            time_range=date_range(data.start_datetime, data.end_datetime),
        )

        return self._tight_layout(ax)

    def scatter(
        self,
        annotator: str | list[str],
        label: str | list[str],
        *,
        ax: Axes | None = None,
        bin_size: Timedelta | BaseOffset | None = None,
        **kwargs: Any,
    ) -> Axes:
        """Scatter-plot detections (hour of day vs. time).

        Parameters
        ----------
        ax: Axes | None
            Optional Axes to plot on; created automatically when omitted.
        annotator, label: str | list[str]
            Selection, see :meth:`DetectionData.filter_df`.
        bin_size: Timedelta | BaseOffset | None
            Accepted for compatibility; it currently does not affect the plot.
        **kwargs
            Additional styling options forwarded to ``ax.scatter``.

        """
        ax = self._get_ax(ax)
        data = self._data

        bin_size = bin_size or get_max_time(data.df)

        labels, annotators = get_labels_and_annotators(data.filter_df(annotator, label))

        for ann, lbl in zip(annotators, labels, strict=True):
            df_filtered = data.filter_df(ann, lbl)

            if df_filtered.empty:
                continue

            scatter(
                df=df_filtered,
                ax=ax,
                # time_range=date_range(
                #     start=data.start_datetime,
                #     end=data.end_datetime,
                #     freq=bin_size,
                # ),
                **kwargs,
            )

        update_legend(ax)

        return self._tight_layout(ax)

    def timeline(
        self,
        annotator: str | list[str],
        label: str | list[str],
        *,
        ax: Axes | None = None,
        color: list[str] | None = None,
    ) -> Axes:
        """Plot detections on a timeline, one row per label.

        Parameters
        ----------
        ax: Axes | None
            Optional Axes to plot on; created automatically when omitted.
        annotator, label: str | list[str]
            Selection, see :meth:`DetectionData.filter_df`.
        color: list[str] | None
            One color per label.

        """
        ax = self._get_ax(ax)
        data = self._data
        df_filtered = data.filter_df(annotator, label)

        ax.set_xlim(data.start_datetime, data.end_datetime)
        result = timeline(
            df=df_filtered,
            ax=ax,
            color=color,
        )
        self._tight_layout(ax)
        return result

    def agreement(
        self,
        annotator: list[str],
        label: str | list[str],
        bin_size: Timedelta | BaseOffset | None = None,
        *,
        ax: Axes | None = None,
    ) -> Axes:
        """Plot the inter-annotator agreement (detection counts per bin).

        Parameters
        ----------
        ax: Axes | None
            Optional Axes to plot on; created automatically when omitted.
        annotator: list[str]
            The two annotators to compare.
        label: str | list[str]
            Label(s) to compare.
        bin_size: Timedelta | BaseOffset
            Size of the time bins in which detections are counted.

        Raises
        ------
        ValueError
            If ``bin_size`` is missing.

        """
        ax = self._get_ax(ax)
        if not bin_size:
            msg = "'bin_size' missing for agreement plot."
            raise ValueError(msg)
        df_filtered = self._data.filter_df(annotator, label)
        return self._tight_layout(
            plot_annotator_agreement(df=get_count(df_filtered, bin_size), ax=ax),
        )
