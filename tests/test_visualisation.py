import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pytest
import pytz
from matplotlib.axes import Axes
from matplotlib.colors import to_rgba
from pandas import DataFrame, DatetimeIndex, Timedelta, Timestamp, date_range, to_datetime
from pandas.tseries import frequencies

from disclose.utils.filtering import get_timezone
from disclose.utils.visualisation import (
    _prepare_timeline_plot,
    _wrap_xtick_labels,
    arc,
    get_legend,
    heatmap,
    histo,
    overview,
    polar,
    scatter,
    set_y_axis_to_percentage,
    timeline, add_sunrise_sunset,
)


def test_overview_runs_without_error(sample_df) -> None:
    try:
        overview(sample_df, sorted(set(sample_df["annotator"])))
    except ValueError:
        pytest.fail("test_detection_perf raised ValueError unexpectedly.")


def test_wrap_xtick_labels_short_labels():
    fig, ax = plt.subplots()
    labels = ["A", "B", "C"]
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)

    _wrap_xtick_labels(ax, max_chars=2)

    wrapped_labels = [label.get_text() for label in ax.get_xticklabels()]
    assert wrapped_labels == labels


def test_wrap_xtick_labels_long_label():
    fig, ax = plt.subplots()
    labels = ["This is a long label"]
    ax.set_xticks([0])
    ax.set_xticklabels(labels)

    _wrap_xtick_labels(ax, max_chars=5)

    wrapped_labels = [label.get_text() for label in ax.get_xticklabels()]
    expected = "This\nis a\nlong\nlabel"
    assert wrapped_labels[0] == expected


def test_wrap_xtick_labels_no_spaces():
    fig, ax = plt.subplots()
    labels = ["abcdefghijk"]
    ax.set_xticks([0])
    ax.set_xticklabels(labels)

    _wrap_xtick_labels(ax, max_chars=4)

    wrapped_labels = [label.get_text() for label in ax.get_xticklabels()]

    expected = "abcd\nefgh\nijk"
    assert wrapped_labels[0] == expected


def test_set_y_axis_to_percentage():
    fig, ax = plt.subplots()
    ax.set_ylabel("Accuracy")
    set_y_axis_to_percentage(ax, max_val=200)
    formatter = ax.yaxis.get_major_formatter()
    assert formatter(100, None) == "50%"
    assert formatter(200, None) == "100%"
    assert ax.get_ylabel() == "Accuracy (%)"
    plt.close(fig)


def test_single_annotator_multiple_labels():
    annotators = ["Alice"]
    labels = ["Label1", "Label2", "Label3"]
    result = get_legend(annotators, labels)
    assert result == labels


def test_multiple_annotators_single_label():
    annotators = ["Alice", "Bob", "Charlie"]
    labels = ["CommonLabel"]
    result = get_legend(annotators, labels)
    assert result == annotators


def test_multiple_annotators_multiple_labels():
    annotators = ["Alice", "Bob", "Charlie"]
    labels = ["Label1", "Label2", "Label3"]
    result = get_legend(annotators, labels)
    expected = ["Alice\nLabel1", "Bob\nLabel2", "Charlie\nLabel3"]
    assert result == expected


def test_single_annotator_single_label():
    annotators = ["Alice"]
    labels = ["Label1"]
    result = get_legend(annotators, labels)
    assert result == ["Alice\nLabel1"]


def test_lists_and_strings_combined():
    annotators = ["Alice", "Bob"]
    labels = ["Label1", "Label2"]
    result = get_legend(annotators, labels)
    expected = ["Alice\nLabel1", "Bob\nLabel2"]
    assert result == expected

# %% polar plot

@pytest.mark.parametrize(
    ("h0", "h1"),
    [(0, 6), (6, 12), (18, 24)],
)
def test_arc_geometry(polar_ax: Axes, h0: float, h1: float) -> None:
    arc(polar_ax, h0, h1, "red")

    assert len(polar_ax.patches) == 1
    patch = polar_ax.patches[0]

    width = np.deg2rad((h1 - h0) * 15)
    center = np.deg2rad((h0 + h1) / 2 * 15)

    assert patch.get_width() == pytest.approx(width)
    assert patch.get_x() + patch.get_width() / 2 == pytest.approx(center)
    assert patch.get_y() == 1  # starts on the outer edge
    assert patch.get_height() == pytest.approx(0.025)  # default thickness


def test_arc_custom_thickness_and_color(polar_ax: Axes) -> None:
    arc(polar_ax, 0, 6, "tab:blue", thickness=0.1)

    patch = polar_ax.patches[0]
    assert patch.get_height() == pytest.approx(0.1)
    assert patch.get_facecolor() == to_rgba("tab:blue")


def test_polar_returns_axes_and_mean_proportion(
    sample_df: DataFrame, polar_ax: Axes,
) -> None:
    df = sample_df[
        (sample_df["annotator"] == "ann1")
        & (sample_df["label"] == "lbl1")
        & (sample_df["type"] == "WEAK")
    ]

    ax, mean_prop = polar(df, ax=polar_ax)

    assert ax is polar_ax
    assert list(mean_prop.index) == list(range(24))

    bins_per_hour = Timedelta("1h") / Timedelta("10s")  # 360
    expected = np.zeros(24)
    expected[6] = ((6 + 1) / bins_per_hour) / 2  # mean over 2 days, 6 detections on 25th + 1 on 26th
    assert mean_prop.to_numpy() == pytest.approx(expected)


def test_polar_bars_match_mean_proportion(sample_df: DataFrame, polar_ax: Axes) -> None:
    _, mean_prop = polar(sample_df, ax=polar_ax)

    assert len(polar_ax.patches) == 24
    heights = [p.get_height() for p in polar_ax.patches]
    assert heights == pytest.approx(mean_prop.to_numpy())


def test_polar_creates_polar_axes_when_none(sample_df: DataFrame) -> None:
    _, ax = plt.subplots()
    with pytest.raises(ValueError, match="Provided ax must have a polar projection"):
        polar(df=sample_df, ax=ax)


def test_polar_axes_layout(sample_df: DataFrame, polar_ax: Axes) -> None:
    polar(sample_df, ax=polar_ax)

    assert polar_ax.get_ylim() == (0, 1)
    assert polar_ax.get_theta_direction() == -1
    assert polar_ax.get_theta_offset() == pytest.approx(np.pi / 2)  # North
    assert len(polar_ax.get_xticklabels()) == 24
    assert polar_ax.get_xticklabels()[0].get_text() == "00h"
    assert polar_ax.get_xticklabels()[23].get_text() == "23h"


def test_polar_title_plural(sample_df: DataFrame, polar_ax: Axes) -> None:
    polar(sample_df[sample_df["label"] == "lbl1"], ax=polar_ax)
    assert polar_ax.get_title() == "lbl1  (2 days)"


def test_polar_title_single_day(sample_df: DataFrame, polar_ax: Axes) -> None:
    one_day = sample_df[(sample_df["start_datetime"].dt.day == 25) & (sample_df["label"] == "lbl2")]
    polar(one_day, ax=polar_ax)
    assert polar_ax.get_title() == "lbl2  (1 day)"


def test_polar_one_hour_bins(polar_ax: Axes) -> None:
    start = Timestamp("2025-01-01 05:00", tz="UTC")
    df = DataFrame({
        "start_datetime": [start],
        "end_datetime": [start + Timedelta(hours=1)],
        "end_time": 3600.0,
        "label": "lbl1",
        "annotator": "ann1",
    })

    _, mean_prop = polar(df, ax=polar_ax)

    assert mean_prop[5] == 1
    assert mean_prop.drop(5).sum() == 0


def test_polar_forwards_kwargs_to_bar(sample_df: DataFrame, polar_ax: Axes) -> None:
    polar(sample_df, ax=polar_ax, color="red", alpha=0.5)

    assert polar_ax.patches[0].get_facecolor() == to_rgba("red", 0.5)


def test_polar_default_alpha_is_one(sample_df: DataFrame, polar_ax: Axes) -> None:
    polar(sample_df, ax=polar_ax, color="red")

    assert polar_ax.patches[0].get_facecolor() == to_rgba("red", 1)


def test_polar_ignores_bin_size_kwarg(sample_df: DataFrame, polar_ax: Axes) -> None:
    # `plot()` forwards bin_size, which has no meaning for polar
    polar(sample_df, ax=polar_ax)


def test_polar_rise_set_requires_coordinates(
    sample_df: DataFrame, polar_ax: Axes,
) -> None:
    with pytest.raises(ValueError, match="`coordinates` is required"):
        polar(sample_df, ax=polar_ax, show_rise_set=True)


def test_polar_rise_set_draws_day_night_arcs(
    monkeypatch: pytest.MonkeyPatch,
    sample_df: DataFrame,
    polar_ax: Axes,
) -> None:
    monkeypatch.setattr(
        "disclose.utils.visualisation.get_sun_times",
        lambda *args, **kwargs: ([7.0, 7.5], [17.0, 17.5]),
    )

    polar(sample_df, ax=polar_ax, show_rise_set=True, coordinates=(48.39, -4.49))

    arcs = polar_ax.patches[24:]  # after the 24 hourly bars
    assert [p.get_facecolor() for p in arcs] == [
        to_rgba(c) for c in ("tab:blue", "grey", "gold", "grey", "tab:blue")
    ]

    # the 5 arcs cover the whole day without overlap
    assert sum(p.get_width() for p in arcs) == pytest.approx(2 * np.pi)


def test_polar_without_rise_set_draws_no_arcs(
    sample_df: DataFrame, polar_ax: Axes,
) -> None:
    polar(sample_df, ax=polar_ax)
    assert len(polar_ax.patches) == 24

# %% histo


@pytest.fixture
def df_counts() -> DataFrame:
    idx = date_range("2025-01-01", periods=5, freq="1D", tz="UTC")
    return DataFrame({"lbl1-ann1": [1, 2, 0, 3, 1]}, index=idx)


def test_histo_labels_and_bars(ax: Axes, df_counts: DataFrame) -> None:
    histo(
        df=df_counts,
        ax=ax,
        bin_size=Timedelta("1D"),
        time_bin=Timedelta("10s"),
        color="tab:blue",
        coordinates=(None, None),
    )

    assert len(ax.patches) == len(df_counts)
    assert [p.get_height() for p in ax.patches] == df_counts["lbl1-ann1"].tolist()
    assert ax.get_ylabel() == "Detections (10s)"
    assert ax.get_xlabel() == "Bin size (1D)"
    assert ax.get_title() == "annotator: ann1\nlabel: lbl1"


def test_histo_not_enough_data(
    ax: Axes, df_counts: DataFrame, caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level("WARNING"):
        histo(
            df=df_counts.iloc[:1],
            ax=ax,
            bin_size=Timedelta("1D"),
            time_bin=Timedelta("10s"),
            coordinates=(None, None),
        )

    assert "do not contains enough detections" in caplog.text
    assert len(ax.patches) == 0


# %% _prepare_timeline_plot

def test_prepare_timeline_plot_axes(
    ax: Axes, df_ann1_lbl1: DataFrame, time_range: DatetimeIndex,
) -> None:
    _prepare_timeline_plot(
        df_ann1_lbl1,
        ax,
        bins=time_range,
        coordinates=(48.39, -4.49),
    )

    assert ax.get_xlim() == pytest.approx(
        (mdates.date2num(time_range[0]), mdates.date2num(time_range[-1])),
    )
    assert ax.get_ylim() == (0, 24)
    assert ax.get_ylabel() == "Hour"
    assert ax.get_title() == "annotator: ann1\nlabel: lbl1"

def test_prepare_timeline_plot_rise_set(
    monkeypatch: pytest.MonkeyPatch,
    ax: Axes,
    df_ann1_lbl1: DataFrame,
    time_range: DatetimeIndex,
) -> None:
    called = []
    monkeypatch.setattr(
        "disclose.utils.visualisation.add_sunrise_sunset",
        lambda *args, **kwargs: called.append(args),
    )

    _prepare_timeline_plot(
        df_ann1_lbl1,
        ax,
        bins=time_range,
        coordinates=(48.39, -4.49),
        show_rise_set=True,
    )

    assert len(called) == 1


def test_prepare_timeline_plot_several_timezones(
    monkeypatch: pytest.MonkeyPatch,
    ax: Axes,
    df_ann1_lbl1: DataFrame,
    time_range: DatetimeIndex,
) -> None:
    monkeypatch.setattr(
        "disclose.utils.visualisation.get_timezone",
        lambda df: [pytz.utc, pytz.FixedOffset(420)],
    )

    with pytest.raises(ValueError, match="Several timezones not supported."):
        _prepare_timeline_plot(
            df_ann1_lbl1,
            ax,
            bins=time_range,
            coordinates=(48.39, -4.49),
            show_rise_set=True,
        )


def test_prepare_timeline_plot_missing_coordinates(
    monkeypatch: pytest.MonkeyPatch,
    ax: Axes,
    df_ann1_lbl1: DataFrame,
    time_range: DatetimeIndex,
) -> None:
    with pytest.raises(ValueError, match=r"coordinates must be provided"):
        _prepare_timeline_plot(
            df_ann1_lbl1,
            ax,
            bins=time_range,
            coordinates=(None, None),
            show_rise_set=True,
        )
# %% scatter


def test_scatter_points_and_legend(
    ax: Axes, df_ann1_lbl1: DataFrame, time_range: DatetimeIndex,
) -> None:
    scatter(df_ann1_lbl1, ax, time_range, coordinates=(0, 0))

    assert len(ax.collections) == 1
    assert len(ax.collections[0].get_offsets()) == len(df_ann1_lbl1)
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["ann1 - lbl1"]
    assert ax.get_ylim() == (0, 24)


# %% heatmap


def test_heatmap_counts_all_detections(
    ax: Axes, df_ann1_lbl1: DataFrame, time_range: DatetimeIndex,
) -> None:
    heatmap(
        df_ann1_lbl1,
        ax,
        bin_size=frequencies.to_offset("1D"),
        time_range=time_range,
        coordinates=(0, 0),
    )

    mat = ax.images[0].get_array()
    assert mat.shape == (24, len(time_range) - 1)
    assert mat.sum() == len(df_ann1_lbl1)
    assert ax.get_ylabel() == "Hour of day"
    assert ax.get_xlabel() == "Time (1D bin)"


# %% timeline


def test_timeline_one_scatter_per_label(ax: Axes, sample_df: DataFrame) -> None:
    df = sample_df[
        (sample_df["annotator"] == "ann5") & (sample_df["label"].isin(["lbl1", "lbl2"]))
    ]

    timeline(df, ax)

    assert len(ax.collections) == 2  # ruff: ignore[magic-value-comparison]
    assert [t.get_text() for t in ax.get_yticklabels()] == ["lbl2", "lbl1"]
    assert ax.get_xlabel() == "Date"


def test_timeline_custom_colors(ax: Axes, sample_df: DataFrame) -> None:
    df = sample_df[
        (sample_df["annotator"] == "ann5") & (sample_df["label"].isin(["lbl1", "lbl2"]))
    ]

    timeline(df, ax, color=["red", "blue"])

    assert ax.collections[0].get_facecolor()[0].tolist() == list(to_rgba("red"))
    assert ax.collections[1].get_facecolor()[0].tolist() == list(to_rgba("blue"))

# %% add_sunrise_sunset

def test_add_sunrise_sunset(sample_df: DataFrame):
    fig, ax = plt.subplots()

    try:
        _prepare_timeline_plot(
            df=sample_df,
            ax=ax,
            bins=date_range(start=min(sample_df["start_datetime"]), end=max(sample_df["end_datetime"]), freq=Timedelta("1min")),
            show_rise_set=True,
            coordinates=(40, -4),
        )

        lines = ax.get_lines()
        assert lines[-2].get_label() == "Sunrise"
        assert lines[-2].get_color() == "darkorange"
        assert lines[-1].get_label() == "Sunset"
        assert lines[-1].get_color() == "royalblue"

        assert ax.get_legend() is not None
        assert [text.get_text() for text in ax.get_legend().get_texts()][-2:] == [
            "Sunrise",
            "Sunset",
        ]
    finally:
        plt.close(fig)
