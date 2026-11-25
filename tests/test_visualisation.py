import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.colors import to_rgba
from pandas import (
    DataFrame,
    DatetimeIndex,
    Timedelta,
    Timestamp,
    date_range,
)
from pandas.tseries import frequencies

from disclose.dataclass.recording_effort import RecordingEffort
from disclose.utils.visualisation import (
    _prepare_timeline_plot,
    arc,
    get_legend,
    heatmap,
    histo,
    overview,
    polar,
    scatter,
    set_dynamic_ylim,
    set_y_axis_to_percentage,
    shade_no_effort,
    timeline,
    update_legend,
)


def test_overview(sample_df: DataFrame) -> None:
    expected = (
        DataFrame(
            {
                "annotator": ["ann1", "ann2", "ann3", "ann4", "ann5", "ann6"],
                "lbl1": [7, 7, 0, 0, 14, 2],
                "lbl2": [17, 15, 14, 11, 11, 7],
                "lbl3": [0, 0, 0, 0, 2, 0],
            },
        )
        .set_index("annotator")
        .T
    )
    actual = overview(sample_df, sorted(set(sample_df["annotator"])))

    assert actual.equals(expected)


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
    sample_df: DataFrame,
    polar_ax: Axes,
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
    expected[6] = (
        (6 + 1) / bins_per_hour
    ) / 2  # mean over 2 days, 6 detections on 25th + 1 on 26th
    assert mean_prop.to_numpy() == pytest.approx(expected)


def test_polar_bars_match_mean_proportion(sample_df: DataFrame, polar_ax: Axes) -> None:
    _, mean_prop = polar(sample_df, ax=polar_ax)

    assert len(polar_ax.patches) == 24
    heights = [p.get_height() for p in polar_ax.patches]
    assert heights == pytest.approx(mean_prop.to_numpy())


def test_polar_rejects_non_polar_axes(sample_df: DataFrame) -> None:
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
    one_day = sample_df[
        (sample_df["start_datetime"].dt.day == 25) & (sample_df["label"] == "lbl2")
    ]
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


def test_polar_draws_one_bar_per_hour(
    sample_df: DataFrame,
    polar_ax: Axes,
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
        rank=0,
        total=1,
        color="tab:blue",
    )

    assert len(ax.patches) == len(df_counts)
    assert [p.get_height() for p in ax.patches] == df_counts["lbl1-ann1"].tolist()
    assert ax.get_ylabel() == "Detections (10s)"
    assert ax.get_xlabel() == "Bin size (1D)"


# %% _prepare_timeline_plot


def test_prepare_timeline_plot_axes(
    ax: Axes,
    df_ann1_lbl1: DataFrame,
) -> None:
    _prepare_timeline_plot(
        df_ann1_lbl1,
        ax,
    )
    assert ax.get_ylim() == (0, 24)
    assert ax.get_title() == "annotator: ann1\nlabel: lbl1"


# %% scatter


def test_scatter_points_and_legend(
    ax: Axes,
    df_ann1_lbl1: DataFrame,
) -> None:
    scatter(df_ann1_lbl1, ax)
    update_legend(ax)

    assert len(ax.collections) == 1
    assert len(ax.collections[0].get_offsets()) == len(df_ann1_lbl1)
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["ann1 - lbl1"]
    assert ax.get_ylim() == (0, 24)


# %% heatmap


def test_heatmap_counts_all_detections(
    ax: Axes,
    df_ann1_lbl1: DataFrame,
    time_range: DatetimeIndex,
) -> None:
    heatmap(
        df_ann1_lbl1,
        ax,
        bin_size=frequencies.to_offset("1D"),
        time_range=time_range,
    )

    mat = ax.images[0].get_array()
    assert mat.shape == (24, len(time_range) - 1)
    assert mat.sum() == len(df_ann1_lbl1)
    assert ax.get_ylabel() == "Detections per hour of day (10s)"
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


# %% polar


def test_polar_creates_polar_axes_when_none(sample_df: DataFrame) -> None:
    ax, _ = polar(sample_df)

    assert ax.name == "polar"


def test_scatter_multiple_annotator_label_groups(ax: Axes, sample_df: DataFrame):
    for ann, lbl in zip(["ann1", "ann2"], 2 * ["lbl1"]):
        df_filtered = sample_df[
            (sample_df["annotator"] == ann) & (sample_df["label"] == lbl)
        ]
        scatter(df_filtered, ax)
    update_legend(ax)
    assert len(ax.collections) == 2
    assert sorted(c.get_label() for c in ax.collections) == [
        "ann1 - lbl1",
        "ann2 - lbl1",
    ]


@pytest.mark.parametrize(
    ("values", "expected_ylim", "expected_ticks"),
    [([1, 2, 3], (0, 4), [0, 1, 2, 3, 4]), ([2, 4, 8], (0, 9), [0, 3, 6, 9])],
)
def test_set_dynamic_ylim(
    ax: Axes,
    values: list[int],
    expected_ylim: tuple[int, int],
    expected_ticks: list[int],
):
    set_dynamic_ylim(ax, DataFrame({"counts": values}))
    assert ax.get_ylim() == pytest.approx(expected_ylim)
    assert ax.get_yticks().tolist() == expected_ticks


def test_shade_no_effort_adds_no_data_legend_entry(
    ax: Axes,
    sample_df_counts: DataFrame,
    effort: RecordingEffort,
):
    histo(
        sample_df_counts.iloc[:, [0]],
        ax,
        Timedelta("1D"),
        Timedelta("12h"),
        rank=0,
        total=1,
    )
    shade_no_effort(ax, observed=effort, legend=True)
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert "no data" in labels
