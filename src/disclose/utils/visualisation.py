"""Plotting helpers for APLOSE-formatted detection data."""

from __future__ import annotations

from itertools import cycle, islice
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import dates as mdates
from matplotlib.dates import num2date
from matplotlib.patches import Patch
from numpy import ceil, polyfit
from pandas import (
    DataFrame,
    DatetimeIndex,
    Series,
    Timedelta,
    Timestamp,
    concat,
    crosstab,
    date_range,
)
from pandas.tseries import frequencies
from scipy.stats import pearsonr

from disclose.utils.core import (
    get_labels_and_annotators,
    get_sun_times,
    get_time_range_and_bin_size,
    round_begin_end_timestamps,
    timedelta_to_str,
)
from disclose.utils.filtering import (
    filter_by_annotator,
    get_labels,
    get_max_time,
)

if TYPE_CHECKING:
    from datetime import tzinfo

    from matplotlib.axes import Axes
    from pandas.tseries.offsets import BaseOffset

    from disclose.dataclass.recording_effort import RecordingEffort

default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def histo(
    df: DataFrame,
    ax: Axes,
    bin_size: Timedelta | BaseOffset,
    time_bin: Timedelta,
    rank: int,
    total: int,
    **kwargs: Any,
) -> None:
    """Draw grouped detection-count bars for a time-binned histogram.

    Parameters
    ----------
    df: DataFrame
        Data to plot.
    ax : matplotlib.axes.Axes
        Matplotlib Axes object on which to draw the histogram.
    bin_size: Timedelta | BaseOffset
        The size of the histogram bins.
    time_bin: Timedelta
        The size of detections.
    **kwargs: Additional keyword arguments depending on the mode.

    """
    bin_size_str = get_bin_size_str(bin_size)

    _, _, bin_size = round_begin_end_timestamps(list(df.index), bin_size)

    bar_width = bin_size / total
    bar_width_days = bar_width.total_seconds() / 86400
    bin_starts = mdates.date2num(df.index)

    kwargs.setdefault("edgecolor", "k")
    kwargs.setdefault("zorder", 2)

    ax.bar(
        bin_starts + rank * bar_width_days,
        df.iloc[:, 0],
        width=bar_width_days,
        align="edge",
        **kwargs,
    )

    ax.set_ylabel(f"Detections ({timedelta_to_str(time_bin)})")
    ax.set_xlabel(f"Bin size ({bin_size_str})")


def arc(ax: Axes, h0: float, h1: float, color: str, thickness: float = 0.025) -> None:
    """Draw a colored band on the outer edge of a polar plot, from hour h0 to h1."""
    ax.bar(
        np.deg2rad((h0 + h1) / 2 * 15),
        height=thickness,
        width=np.deg2rad((h1 - h0) * 15),
        bottom=1,
        color=color,
        edgecolor="none",
        clip_on=False,
    )


def polar(
    df: DataFrame,
    ax: Axes | None = None,
    **kwargs: Any,
) -> tuple[Axes, Series]:
    """Plot the mean proportion of detected bins per hour of day on a polar axis.

    Parameters
    ----------
    df: Dataframe
        Detections, with `start_datetime` and `end_datetime` columns.
    ax: Axes
        Polar axes to draw on. A square figure is created if None.
    **kwargs
        Forwarded to `ax.bar` (e.g. color, alpha, edgecolor).

    Returns
    -------
    The axes and the mean proportion of detections for each hour of the day.

    """
    if ax is None:
        _, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(6, 6))

    if ax.name != "polar":
        msg = "Provided ax must have a polar projection."
        raise ValueError(msg)

    kwargs.setdefault("alpha", 1)
    kwargs.setdefault("zorder", 3)

    label = get_labels(df)
    bin_size = get_max_time(df)
    start = df["start_datetime"]
    day, hour = start.dt.date, start.dt.hour

    # Detections per day and hour, with empty days and hours filled with zeros
    all_days = date_range(min(day), max(day), freq="D").date
    counts = crosstab(day, hour).reindex(
        index=all_days,
        columns=range(24),
        fill_value=0,
    )

    # Proportion of bins with a detection per hour, averaged over days
    prop = counts / (Timedelta("1h") / bin_size)
    mean_prop = prop.mean(axis=0)

    ax.bar(
        2 * np.pi * (mean_prop.index.values + 0.5) / 24,
        mean_prop.values,
        width=2 * np.pi / 24 * 0.9,
        **kwargs,
    )
    ax.set_ylim(0, 1)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_xticks(np.deg2rad(np.arange(0, 360, 15)))
    ax.set_xticklabels([f"{h:02d}h" for h in range(24)])
    ax.set_yticklabels([])
    n_days = len(all_days)
    ax.set_title(f"{label}  ({n_days} day{'s' if n_days > 1 else ''})")

    return ax, mean_prop


def _prepare_timeline_plot(
    df: DataFrame,
    ax: Axes,
) -> None:
    """Plot daily detection patterns for a given annotator and label.

    Parameters
    ----------
    df: DataFrame
        data to plot
    ax : plt.Axes
        The matplotlib axis to draw on.

    """
    labels, annotators = get_labels_and_annotators(df)

    ax.set_ylim(0, 24)
    ax.set_yticks(range(0, 25, 2))
    ax.grid(color="k", linestyle="-", linewidth=0.2)
    set_plot_title(ax=ax, annotators=annotators, labels=labels)


def scatter(
    df: DataFrame,
    ax: Axes,
    **kwargs: Any,
) -> None:
    """Scatter-plot of detections for a given annotator and label.

    Parameters
    ----------
    df: DataFrame
        data to plot
    ax : plt.Axes
        The matplotlib axis to draw on.
    **kwargs: Additional keyword forwarder to matplotlib scatter.

    """
    label, annotator = get_labels_and_annotators(df)

    _prepare_timeline_plot(
        df=df,
        ax=ax,
    )

    detect_time_dec = [
        ts.hour + ts.minute / 60 + ts.second / 3600 for ts in df["start_datetime"]
    ]

    kwargs.setdefault("marker", "x")
    ax.scatter(
        df["start_datetime"],
        detect_time_dec,
        label=f"{annotator[0]} - {label[0]}",
        **kwargs,
    )


def heatmap(
    df: DataFrame,
    ax: Axes,
    bin_size: Timedelta | BaseOffset,
    time_range: DatetimeIndex,
) -> None:
    """Heatmap of detections for a given annotator and label.

    Parameters
    ----------
    df: DataFrame
        data to plot
    ax : plt.Axes
        The matplotlib axis to draw on.
    bin_size: Timedelta | BaseOffset
        The size of the heatmap bins.
        Must be >= 24h.
    time_range: DatetimeIndex
        The time range of the heatmap.

    """
    datetime_list = list(df["start_datetime"])

    _, bin_size_dt = get_time_range_and_bin_size(datetime_list, bin_size)
    if bin_size_dt < Timedelta("1D"):
        msg = "`bin_size` must be >= 24h for heatmap mode."
        raise ValueError(msg)

    begin = time_range[0]
    end = time_range[-1]

    # Coarse bins (for display cells)
    cell_bins = date_range(begin, end, freq=bin_size)

    _prepare_timeline_plot(
        df=df,
        ax=ax,
    )

    freq = frequencies.to_offset(get_max_time(df))

    # Fine bins (for counting detection)
    fine_bins = date_range(begin, end, freq=freq)

    # Assign each timestamp to fine bin
    fine_idx = np.searchsorted(fine_bins, datetime_list, side="right") - 1

    # Map fine bins to coarse cell index
    fine_to_cell = np.searchsorted(cell_bins, fine_bins, side="right") - 1

    mat = np.zeros((24, len(cell_bins) - 1), dtype=int)

    for dt, f_idx in zip(datetime_list, fine_idx, strict=False):
        if 0 <= f_idx < len(fine_bins) - 1:
            c_idx = fine_to_cell[f_idx]
            if 0 <= c_idx < len(cell_bins) - 1:
                mat[dt.hour, c_idx] += 1

    im = ax.imshow(
        mat,
        extent=(begin, end, 0, 24),
        vmin=0,
        vmax=mat.max(),
        aspect="auto",
        origin="lower",
    )

    bin_size_str = get_bin_size_str(bin_size)
    freq_str = get_bin_size_str(freq)

    fig = ax.get_figure()
    fig.colorbar(im, ax=ax, pad=0.01)
    ax.set_ylabel(f"Detections per hour of day ({freq_str})")
    ax.set_xlabel(f"Time ({bin_size_str} bin)")


def overview(df: DataFrame, annotator: str | list[str]) -> DataFrame:
    """Return detection counts by label and annotator as a pivot table.

    Parameters
    ----------
    df: DataFrame
        The Dataframe to analyze.
    annotator: str | list[str] | None
        Annotator or annotators to include.

    Returns
    -------
    DataFrame
        Counts indexed by label and with one column per annotator.

    """
    df = filter_by_annotator(df, annotator)

    return df.pivot_table(
        index="label",
        columns="annotator",
        values="start_datetime",
        aggfunc="count",
        fill_value=0,
    )


def plot_annotator_agreement(
    df: DataFrame,
    ax: Axes,
) -> Axes:
    """Plot inter-annotator agreement with linear regression.

    Creates a scatter plot comparing detection counts between two annotators
    across time bins. Fits a linear regression line and displays the coefficient
    of determination (R²) in the legend.

    Parameters
    ----------
    df : DataFrame
        APLOSE-formatted DataFrame containing labels from exactly two annotators.
    ax : plt.Axes
        Matplotlib axes object where the scatter plot and regression line will be drawn.

    Returns
    -------
    Axes
        The modified axes.

    Notes
    -----
    Each point represents the two selected count series for one time bin.
    The input columns are expected to contain exactly two label/annotator pairs.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from pandas import DataFrame
    >>> df_count = DataFrame({"label-ann1": [1, 0], "label-ann2": [0, 1]})
    >>> fig, ax = plt.subplots()
    >>> ax = plot_annotator_agreement(df_count, ax)

    """
    labels, annotators = zip(*[col.rsplit("-", 1) for col in df.columns], strict=True)
    labels = list(labels)
    annotators = list(annotators)

    ax.scatter(
        df[f"{labels[0]}-{annotators[0]}"],
        df[f"{labels[1]}-{annotators[1]}"],
        zorder=3,
    )
    coefficients = polyfit(
        df[f"{labels[0]}-{annotators[0]}"],
        df[f"{labels[1]}-{annotators[1]}"],
        deg=1,
    )
    poly = np.poly1d(coefficients)
    r, _ = pearsonr(
        df[f"{labels[0]}-{annotators[0]}"],
        df[f"{labels[1]}-{annotators[1]}"],
    )  # R²
    ax.plot(
        sorted(df[f"{labels[0]}-{annotators[0]}"]),
        poly(sorted(df[f"{labels[0]}-{annotators[0]}"])),
        lw=0.5,
        color="k",
        alpha=0.5,
        linestyle="-",
        label=f"R²={r**2:.2f}",
        zorder=2,
    )

    ax.set_xlabel(f"""annotator: {annotators[0]}\nlabel: {labels[0]}""")
    ax.set_ylabel(f"""annotator: {annotators[1]}\nlabel: {labels[1]}""")
    ax.grid(
        linestyle="-",
        linewidth=0.2,
        zorder=1,
    )
    ax.legend(
        loc="upper left",
        frameon=True,
        framealpha=1,
        fontsize=8,
    )

    return ax


def timeline(
    df: DataFrame,
    ax: Axes,
    **kwargs: list[str],
) -> Axes:
    """Plot detections on a timeline.

    Parameters
    ----------
    df: DataFrame
        APLOSE DataFrame
    ax : Axes
        Matplotlib axes object on which detections are plotted.
    **kwargs
        Plot options. ``color`` may be a list with one color per label; default
        Matplotlib cycle colors are used when omitted.

    """
    color = kwargs.get("color")
    lat, _ = kwargs.get("coordinates") or (None, None)

    labels, _ = get_labels_and_annotators(df)

    color = color or [c for _, c in zip(range(len(labels)), cycle(default_colors))]

    for i, label in enumerate(labels):
        time_det = df[(df["label"] == label)]["start_datetime"].to_list()
        l_data = len(time_det)
        x = np.ones((l_data, 1), int) * i
        ax.scatter(time_det, x, color=color[i])

    ax.grid(color="k", linestyle="-", linewidth=0.2)
    ax.set_yticks(np.arange(0, len(labels), 1))
    ax.set_yticklabels(labels[::-1])
    ax.set_xlabel("Date")

    return ax


# %% utils


def get_colors(df: DataFrame) -> list[str]:
    """Return default plot colors."""
    if {"annotator", "label"}.issubset(df.columns):
        n_pairs = len(df[["annotator", "label"]].drop_duplicates())
        return list(islice(cycle(default_colors), n_pairs))
    return [c for _, c in zip(range(len(df.columns)), cycle(default_colors))]


def get_legend(annotators: str | list[str], labels: str | list[str]) -> list[str]:
    """Return plot legend."""
    if len(set(labels)) > 1 and len(set(annotators)) == 1:
        return labels
    if len(set(annotators)) > 1 and len(set(labels)) == 1:
        return annotators
    return [f"{ant}\n{lbl}" for ant, lbl in zip(annotators, labels, strict=False)]


def get_bin_size_str(bin_size: Timedelta | BaseOffset) -> str:
    """Return the bin size as a string."""
    if isinstance(bin_size, Timedelta):
        return timedelta_to_str(bin_size)

    return f"{bin_size.n}{bin_size.rule_code}"


def set_y_axis_to_percentage(ax: plt.Axes, max_val: float) -> None:
    """Set y-axis to percentage."""
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda y, _: f"{(y / max_val) * 100:.0f}%"),
    )

    current_label = ax.get_ylabel()
    if current_label and "%" not in current_label:
        ax.set_ylabel(f"{current_label} (%)")


def set_dynamic_ylim(
    ax: Axes,
    df: DataFrame,
    padding: float = 0.05,
    nticks: int = 4,
) -> None:
    """Set y-axis limits and ticks dynamically based on DataFrame values."""
    max_val = np.nanmax(df.to_numpy())
    upper_lim = int(ceil((1 + padding) * max_val))
    ax.set_ylim(0, upper_lim)

    step = int(max(1, ceil(max_val / (nticks - 1))))
    ax.set_yticks(range(0, upper_lim + 1, step))


def set_plot_title(ax: Axes, annotators: list[str], labels: list[str]) -> None:
    """Set plot title."""
    title = f"annotator: {', '.join(set(annotators))}\nlabel: {', '.join(set(labels))}"
    ax.set_title(title)


def shade_no_effort(
    ax: Axes,
    observed: RecordingEffort,
    legend: bool = True,
    **kwargs: Any,
) -> None:
    """Shade areas of the plot where no observation effort was made.

    Parameters
    ----------
    ax : plt.Axes
        The axes on which to draw the shaded regions.
    observed : RecordingEffort
        Recording-effort counts indexed by left-closed time intervals.
    legend : bool
        Whether to add legend entries for missing and partial effort.

    Notes
    -----
    ``observed.counts`` must contain at least two intervals of equal width.

    """
    # Convert effort IntervalIndex → DatetimeIndex (bin starts)
    effort_by_start = Series(
        observed.counts.values,
        index=[i.left for i in observed.counts.index],
    )

    bar_width = effort_by_start.index[1] - effort_by_start.index[0]
    width_days = bar_width.total_seconds() / 86400

    max_effort = bar_width / observed.timebin_origin
    effort_fraction = effort_by_start / max_effort

    first_elem = Series([0], index=[effort_fraction.index[0] - bar_width])
    last_elem = Series([0], index=[effort_fraction.index[-1] + bar_width])
    effort_fraction = concat([first_elem, effort_fraction, last_elem])

    no_effort = effort_fraction[effort_fraction == 0]
    partial_effort = effort_fraction[(effort_fraction > 0) & (effort_fraction < 1)]

    # Get legend handle
    handles1, labels1 = ax.get_legend_handles_labels()

    kwargs.setdefault("alpha", 0.1)
    kwargs.setdefault("facecolor", 0.65)
    facecolor = kwargs.pop("facecolor")

    _draw_effort_spans(
        ax=ax,
        effort_index=partial_effort.index,
        width_days=width_days,
        facecolor=str(facecolor),
        label="partial data",
        **kwargs,
    )

    _draw_effort_spans(
        ax=ax,
        effort_index=no_effort.index,
        width_days=width_days,
        facecolor=str(0.7 * facecolor),
        label="no data",
        **kwargs,
    )

    # Add effort legend to current plot legend
    handles_effort = []
    if len(partial_effort) > 0:
        handles_effort.append(
            Patch(facecolor="0.65", alpha=0.1, label="partial data"),
        )
    if len(no_effort) > 0:
        handles_effort.append(
            Patch(facecolor="0.45", alpha=0.15, label="no data"),
        )
    if handles_effort and legend:
        labels_effort = [h.get_label() for h in handles_effort]
        handles = handles1 + handles_effort
        labels = labels1 + labels_effort
        ax.legend(
            handles,
            labels,
        )


def _draw_effort_spans(
    ax: plt.Axes,
    effort_index: DatetimeIndex,
    width_days: float,
    *,
    facecolor: str,
    alpha: float,
    label: str,
) -> None:
    """Draw shaded spans for time bins with the given effort category."""
    for ts in effort_index:
        start = mdates.date2num(ts)
        ax.axvspan(
            start,
            start + width_days,
            facecolor=facecolor,
            alpha=alpha,
            linewidth=0,
            zorder=1,
            label=label,
        )


def add_sunrise_sunset(ax: Axes, lat: float, lon: float, tz: tzinfo) -> None:
    """Display local sunrise and sunset times across a time-based plot.

    The axes x-limits determine the date range; ``tz`` is used to interpret
    those dates and calculate the solar times.
    """
    x_min, x_max = ax.get_xlim()
    start_date = Timestamp(num2date(x_min)).tz_convert(tz)
    end_date = Timestamp(num2date(x_max)).tz_convert(tz)

    num_days = (end_date.date() - start_date.date()).days + 1
    dates = [start_date.date() + Timedelta(days=i) for i in range(num_days)]

    sunrise, sunset = get_sun_times(
        start=start_date,
        stop=end_date,
        lat=lat,
        lon=lon,
    )

    ax.plot(dates, sunrise, color="darkorange", label="Sunrise")
    ax.plot(dates, sunset, color="royalblue", label="Sunset")


def update_legend(ax: Axes, bbox: tuple[float, float] = (1.1, 1)) -> None:
    """Create a consistently positioned legend without duplicate labels."""
    handles, labels = ax.get_legend_handles_labels()

    unique = dict(zip(labels, handles, strict=True))

    if not unique:
        return

    sorted_items = sorted(unique.items(), key=lambda item: item[0].lower())

    ax.legend(
        [handle for _, handle in sorted_items],
        [label for label, _ in sorted_items],
        loc="upper left",
        bbox_to_anchor=bbox,
        frameon=True,
        framealpha=0.6,
    )
