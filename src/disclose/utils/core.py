"""General functions."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import astral
from astral.sun import sunrise, sunset
from matplotlib.axes import Axes
from pandas import (
    DataFrame,
    DatetimeIndex,
    Series,
    Timedelta,
    Timestamp,
    cut,
    date_range,
    json_normalize,
    to_datetime,
)
from pandas.tseries import offsets
from pandas.tseries.offsets import BaseOffset

if TYPE_CHECKING:
    from datetime import tzinfo
    from pathlib import Path

    import matplotlib.pyplot as plt


def build_time_vector(df: DataFrame, bin_size: Timedelta) -> DatetimeIndex:
    """Build a time vector from a DataFrame and a frequency."""
    t1 = min(df["start_datetime"]).floor(bin_size)
    t2 = max(df["end_datetime"]).ceil(bin_size)
    return date_range(start=t1, end=t2, freq=bin_size)


def get_season(ts: Timestamp, *, northern: bool = True) -> tuple[str, int]:
    """Determine the meteorological season from a Timestamp.

    In the Northern Hemisphere
    Winter: Dec-Feb, Spring: Mar-May, Summer: Jun-Aug, Autumn: Sep-Nov

    In the Southern Hemisphere
    Winter: Jun-Aug, Spring: Sep-Nov, Summer: Dec-Feb, Autumn: Mar-May

    Parameters
    ----------
    northern: boolean, default True.
        Specify if the seasons are northern or austral
    ts: Timestamp
        Considered datetime

    Returns
    -------
    tuple[str, int]
        Season name and the year in which that season begins. January and
        February are assigned to the previous year's winter (or austral summer).

    """
    if northern:
        spring = [3, 4, 5]
        summer = [6, 7, 8]
        autumn = [9, 10, 11]
    else:
        spring = [9, 10, 11]
        summer = [1, 2, 12]
        autumn = [3, 4, 5]

    if ts.month in spring:
        season = "spring"
    elif ts.month in summer:
        season = "summer"
    elif ts.month in autumn:
        season = "autumn"
    else:
        season = "winter"

    return season, ts.year - 1 if ts.month in [1, 2] else ts.year


def get_sun_times(
    start: Timestamp,
    stop: Timestamp,
    lat: float,
    lon: float,
) -> tuple[list[float], list[float]]:
    """Return local sunrise and sunset times for each date from start through stop.

    Parameters
    ----------
    start: Timestamp
        start datetime of when to fetch sun hour
    stop: Timestamp
        end datetime of when to fetch sun hour
    lat: float
        latitude in decimal degrees
    lon: float
        longitude in decimal degrees

    Returns
    -------
    hour_sunrise: list[float]
        Sunrise decimal hours for each day between start and stop

    hour_sunset: list[float]
        Sunset decimal hours for each day between start and stop

    Raises
    ------
    ValueError
        If either timestamp is timezone-naive.

    """
    if start.tz is None or stop.tz is None:
        msg = "start and stop must be timezone-aware"
        raise ValueError(msg)

    tz = start.tz

    gps = astral.LocationInfo(latitude=lat, longitude=lon, timezone=tz)

    h_sunrise, h_sunset = [], []

    for date in [
        ts.date()
        for ts in date_range(
            start.normalize(),
            stop.normalize(),
            freq="D",
        )
    ]:
        dt_sunrise = Timestamp(sunrise(gps.observer, date=date)).tz_convert(tz)
        dt_sunset = Timestamp(sunset(gps.observer, date=date)).tz_convert(tz)

        # Convert sunrise and sunset to decimal hours
        h_sunrise.append(
            dt_sunrise.hour
            + dt_sunrise.minute / 60
            + dt_sunrise.second / 3600
            + dt_sunrise.microsecond / 3_600_000_000,
        )
        h_sunset.append(
            dt_sunset.hour
            + dt_sunset.minute / 60
            + dt_sunset.second / 3600
            + dt_sunset.microsecond / 3_600_000_000,
        )

    return h_sunrise, h_sunset


def json2df(json_path: Path) -> DataFrame:
    """Convert a metadatax JSON file into a DataFrame.

    Parameters
    ----------
    json_path: Path
        JSON file path

    """
    with json_path.open(encoding="utf-8") as f:
        metadatax_df = json_normalize(json.load(f))
        metadatax_df["deployment_date"] = to_datetime(
            metadatax_df["deployment_date"],
        )
        metadatax_df["recovery_date"] = to_datetime(metadatax_df["recovery_date"])

    return metadatax_df


def add_season_period(
    ax: Axes,
    bar_height: int = 10,
    *,
    northern: bool = True,
) -> None:
    """Add a color-coded meteorological season bar above a time-series plot.

    Parameters
    ----------
    northern: boolean, default True.
        Specify if the seasons are northern or austral
    ax: Axes
        Matplotlib Axes to add the bar to.
    bar_height: int
        Bar height in pixels

    """
    if not ax.has_data():
        msg = "Axes have no data"
        raise ValueError(msg)

    start = Timestamp(ax.get_xlim()[0], unit="D").round("1ms")
    end = Timestamp(ax.get_xlim()[1], unit="D").round("1ms")
    month_starts = date_range(
        start=start.normalize().replace(day=1),
        end=end,
        freq="MS",
    )
    bins = [start]
    bins.extend(month for month in month_starts if start < month < end)
    bins.append(end)

    season_colors = {
        "winter": "#84eceb",
        "spring": "#91de92",
        "summer": "#fce097",
        "autumn": "#f9c1a5",
    }

    bar_height = set_bar_height(ax, bar_height)
    bar_bottom = ax.get_ylim()[1] + (0.2 * bar_height)

    seasons = [
        get_season(start + (end - start) / 2, northern=northern)[0]
        for start, end in zip(bins[:-1], bins[1:], strict=True)
    ]
    season_intervals = []
    interval_start = bins[0]
    current_season = seasons[0]
    for index, season in enumerate(seasons[1:], start=1):
        if season != current_season:
            season_intervals.append((interval_start, bins[index], current_season))
            interval_start = bins[index]
            current_season = season
    season_intervals.append((interval_start, bins[-1], current_season))

    for start, end, season in season_intervals:
        center = start + (end - start) / 2
        ax.bar(
            center,
            height=bar_height,
            bottom=bar_bottom,
            width=(end - start),
            color=season_colors[season],
            align="center",
            zorder=2,
            alpha=1,
            linewidth=0,
        )

    ymin, _ = ax.get_ylim()
    ax.set_ylim(ymin, ax.dataLim.ymax)


def set_bar_height(ax: plt.Axes, pixel_height: int = 10) -> float:
    """Convert pixel height to data coordinates.

    Parameters
    ----------
    ax: Axes
        Figure plot

    pixel_height: int
        In pixel

    """
    if not ax.has_data():
        msg = "Axe have no data"
        raise ValueError(msg)

    display_to_data = ax.transData.inverted().transform
    _, data_bottom = display_to_data((0, 0))  # Bottom of the axis
    _, data_top = display_to_data((0, pixel_height))  # Top of the bar

    return data_top - data_bottom  # Convert pixel height to data scale


def add_recording_period(
    df: DataFrame,
    ax: plt.Axes,
    bar_height: int = 10,
) -> None:
    """Add a bar below a time-series plot showing deployment periods.

    Parameters
    ----------
    df: DataFrame
        Includes the recording campaign deployment
        and recovery dates (typically extracted from Metadatax)

    ax: Axes
        Figure plot

    bar_height: int
        Bar height in pixels

    """
    if not ax.has_data():
        msg = "Axe have no data"
        raise ValueError(msg)

    recorder_intervals = [
        (start, end - start)
        for start, end in zip(df["deployment_date"], df["recovery_date"], strict=False)
    ]

    bar_height = set_bar_height(ax=ax, pixel_height=bar_height)

    ax.broken_barh(
        recorder_intervals,
        (ax.get_ylim()[0] - (1.2 * bar_height), bar_height),
        facecolors="red",
        alpha=0.6,
    )
    ax.set_ylim(ax.dataLim.ymin, ax.dataLim.ymax)


def get_count(
    df: DataFrame,
    bin_size: Timedelta | BaseOffset,
    time: DatetimeIndex | None = None,
    effort: Any | None = None,
) -> DataFrame:
    """Count observations per label/annotator pair in time bins.

    This function groups a DataFrame of events into uniform time bins and counts the
    number of events for each (label, annotator) pair in each bin.

    Parameters
    ----------
    df : DataFrame
        APLOSE-formatted DataFrame.
    bin_size : Timedelta | offsets
        Width or frequency of bins.
    time: DatetimeIndex
        Optional bin-edge timestamps specifying the time range; ignored when
        ``effort`` is provided.
    effort: Any | None
        Optional ``RecordingEffort`` used to restrict counts to active bins.

    Returns
    -------
    DataFrame
        A DataFrame indexed by the start of each bin (datetime), with columns named
        "<label>-<annotator>", containing the count of observations in that bin.

    Raises
    ------
    ValueError
        If ``df`` is empty or the supplied recording effort contains no active bins.

    """
    if df.empty:
        msg = "`df` contains no data"
        raise ValueError(msg)

    if effort is not None:
        eff = Series(effort.counts.to_numpy(), index=effort.counts.index.left)
        bin_activity = eff.resample(bin_size, closed="left", label="left").sum()
        active_starts = bin_activity[bin_activity > 0].index

        if active_starts.empty:
            msg = "`effort` contains no recording activity"
            raise ValueError(msg)

        # Build cut edges only from active bins (start and start + bin_size)
        active_ends = active_starts + bin_size
        bins = DatetimeIndex(sorted(set(active_starts) | set(active_ends)))

        # Restrict events to the active window span (cheap pre-filter)
        df = df[
            (df["start_datetime"] >= active_starts.min())
            & (df["start_datetime"] < active_ends.max())
        ]
    else:
        datetime_list = list(df["start_datetime"]) if time is None else time.to_list()
        bins, bin_size = get_time_range_and_bin_size(datetime_list, bin_size)
        active_starts = bins[:-1]

    labels, annotators = get_labels_and_annotators(df)

    series_list = [
        df[(df["label"] == label) & (df["annotator"] == annotator)]["start_datetime"]
        for label, annotator in zip(labels, annotators, strict=False)
    ]

    counts_df = DataFrame(index=active_starts)
    for i, series in enumerate(series_list):
        binned = cut(series, bins=bins, right=False)
        counts_df[f"{labels[i]}-{annotators[i]}"] = binned.value_counts().sort_index()
    return counts_df


def get_labels_and_annotators(df: DataFrame) -> tuple[list, list]:
    """Extract and align labels and annotators from an APLOSE DataFrame.

    Parameters
    ----------
    df : DataFrame
        The APLOSE-formatted DataFrame.

    Returns
    -------
    tuple[list, list]
        A tuple containing the labels and annotators lists.

    """
    if df.empty:
        msg = "`df` contains no data"
        raise ValueError(msg)

    unique_pairs = df[["annotator", "label"]].drop_duplicates()
    return unique_pairs["label"].to_list(), unique_pairs["annotator"].to_list()


def localize_timestamps(timestamps: list[Timestamp], tz: tzinfo) -> list[Timestamp]:
    """Localize naive timestamps to ``tz`` and leave aware timestamps unchanged."""
    localized = []
    for ts in timestamps:
        if ts.tzinfo is None or ts.tzinfo.utcoffset(ts) is None:
            localized.append(ts.tz_localize(tz))
        else:
            localized.append(ts)
    return localized


def get_time_range_and_bin_size(
    timestamp_list: list[Timestamp],
    bin_size: Timedelta | BaseOffset,
) -> tuple[DatetimeIndex, Timedelta]:
    """Build inclusive bin edges spanning the timestamps and resolve bin width.

    For calendar offsets, the returned ``Timedelta`` is the duration of the
    first generated interval.
    """
    if not isinstance(timestamp_list, list) or not all(
        isinstance(ts, Timestamp) for ts in timestamp_list
    ):
        msg = "`timestamp_list` must be a list[Timestamp]"
        raise TypeError(msg)

    if len(timestamp_list) == 0:
        msg = "`timestamp_list` is empty"
        raise ValueError(msg)

    start, end, _ = round_begin_end_timestamps(timestamp_list, bin_size)
    timestamp_range = date_range(start=start, end=end, freq=bin_size)

    if isinstance(bin_size, Timedelta):
        return timestamp_range, bin_size
    if isinstance(bin_size, BaseOffset):
        return timestamp_range, timestamp_range[1] - timestamp_range[0]
    msg = "bin_size must be a Timedelta or BaseOffset."
    raise TypeError(msg)


def round_begin_end_timestamps(
    timestamp_list: list[Timestamp],
    bin_size: Timedelta | BaseOffset,
) -> tuple[Any, Any, Any]:
    """Round the timestamp range outward to bin boundaries.

    Returns the start boundary, exclusive end boundary, and resolved bin width.
    Calendar offsets are handled using their rollback/rollforward boundaries.
    """
    if not isinstance(timestamp_list, list) or not all(
        isinstance(ts, Timestamp) for ts in timestamp_list
    ):
        msg = "timestamp_list must be a list[Timestamp]"
        raise TypeError(msg)

    if not timestamp_list:
        msg = "`timestamp_list` is empty"
        raise ValueError(msg)

    if isinstance(bin_size, Timedelta):
        start = min(timestamp_list).floor(bin_size)
        end = max(timestamp_list).floor(bin_size) + bin_size
        return start, end, bin_size

    if isinstance(bin_size, BaseOffset):
        start = bin_size.rollback(min(timestamp_list))
        end = bin_size.rollforward(max(timestamp_list))
        if not isinstance(bin_size, (offsets.Hour, offsets.Minute, offsets.Second)):
            start = Timestamp(start).normalize()
            end = Timestamp(end).normalize() + bin_size
            if start == end:
                end += bin_size

        timestamp_range = date_range(start=start, end=end, freq=bin_size)
        bin_size = timestamp_range[1] - timestamp_range[0]
        if bin_size.resolution_string in {"s", "min", "h"}:
            return start.floor(bin_size), end.ceil(bin_size), bin_size
        return start, end, bin_size

    msg = "Could not get start/end timestamps."
    raise ValueError(msg)


def timedelta_to_str(td: Timedelta) -> str:
    """Format a whole-second ``Timedelta`` using day, hour, minute, or second units."""
    seconds = int(td.total_seconds())

    if seconds % 86400 == 0:
        return f"{seconds // 86400}D"
    if seconds % 3600 == 0:
        return f"{seconds // 3600}h"
    if seconds % 60 == 0:
        return f"{seconds // 60}min"
    return f"{seconds}s"
