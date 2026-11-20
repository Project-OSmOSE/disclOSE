import logging
import re
from contextlib import nullcontext
from copy import copy
from pathlib import Path
from typing import ContextManager

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pytest
import pytz
from matplotlib.axes import Axes
from pandas import DataFrame, Timedelta, Timestamp
from pandas.tseries import frequencies

from disclose.dataclass.detection_data import (
    DetectionData,
    PlotAccessor,
    _get_locator_from_offset,
)
from disclose.utils.filtering import get_timezone


def test_empty_DataAplose() -> None:
    data = DetectionData()
    assert data.df.empty
    assert data.annotator is None
    assert data.label is None
    assert data.dataset is None
    assert data.start_datetime is None
    assert data.end_datetime is None
    assert data.lat is None
    assert data.lon is None


def test_data_aplose_str(sample_df: DataFrame) -> None:
    obj = DetectionData(sample_df)

    expected = (
        f"start_datetime: {obj.start_datetime}\n"
        f"end_datetime: {obj.end_datetime}\n"
        f"annotator: {obj.annotator}\n"
        f"label: {obj.label}\n"
        f"dataset: {obj.dataset}"
    )
    assert str(obj) == expected
    assert repr(obj) == str(obj)


def test_data_aplose_init(sample_df: DataFrame) -> None:
    data = DetectionData(sample_df)
    assert isinstance(data.df, DataFrame)
    assert sorted(data.annotator) == ["ann1", "ann2", "ann3", "ann4", "ann5", "ann6"]
    assert sorted(data.label) == ["lbl1", "lbl2", "lbl3"]
    assert data.dataset == "sample_dataset"
    assert data.start_datetime == sample_df["start_datetime"].min()
    assert data.end_datetime == sample_df["end_datetime"].max()


@pytest.mark.parametrize(
    ("value", "expected_msg"),
    [
        pytest.param((48.39, -4.49), None, id="valid"),
        pytest.param((0.0, 0.0), None, id="valid_zero"),
        pytest.param(
            [48.39, -4.49],
            re.escape("Coordinates must be a tuple of two floats: (lat, lon)."),
            id="invalid_list",
        ),
        pytest.param(
            (48.39,),
            re.escape("Coordinates must be a tuple of two floats: (lat, lon)."),
            id="invalid_missing_lon",
        ),
        pytest.param(
            (48, 5, 6),
            re.escape("Coordinates must be a tuple of two floats: (lat, lon)."),
            id="invalid_long_tuple",
        ),
    ],
)
def test_coordinates_setter(value: tuple, expected_msg: str | None) -> None:
    obj = DetectionData()

    if expected_msg is None:
        obj.coordinates = value
        assert (obj.lat, obj.lon) == value
    else:
        with pytest.raises(ValueError, match=expected_msg):
            obj.coordinates = value


def test_filter_df_single_pair(sample_df: DataFrame) -> None:
    data = DetectionData(sample_df)
    filtered_data = data.filter_df(annotator="ann1", label="lbl1")
    assert sorted(set(filtered_data["label"])) == ["lbl1"]
    assert sorted(set(filtered_data["annotator"])) == ["ann1"]
    expected = sample_df[
        (sample_df["annotator"] == "ann1") & (sample_df["label"] == "lbl1")
    ].reset_index(drop=True)
    assert filtered_data.equals(expected)


def test_change_tz(sample_df: DataFrame) -> None:
    data = DetectionData(sample_df)
    new_tz = "Etc/GMT-7"
    data.change_tz(new_tz)
    start_dt = data.df["start_datetime"]
    end_dt = data.df["end_datetime"]
    assert all(ts.tz.key == new_tz for ts in start_dt), (
        f"The detection start timestamps have to be in {new_tz} timezone"
    )
    assert all(ts.tz.key == new_tz for ts in end_dt), (
        f"The detection end timestamps have to be in {new_tz} timezone"
    )
    assert data.start_datetime.tz.key == new_tz, (
        f"The begin value of the DataAplose has to be in {new_tz} timezone"
    )
    assert data.end_datetime.tz.key == new_tz, (
        f"The end value of the DataAplose has to be in {new_tz} timezone"
    )


def test_filter_df_multiple_pairs(sample_df: DataFrame) -> None:
    data = DetectionData(sample_df)
    filtered_data = data.filter_df(annotator=["ann1", "ann2"], label=["lbl1", "lbl2"])
    assert sorted(set(filtered_data["label"])) == ["lbl1", "lbl2"]
    assert sorted(set(filtered_data["annotator"])) == ["ann1", "ann2"]
    pairs = [("ann1", "lbl1"), ("ann2", "lbl2")]
    expected = sample_df[
        sample_df[["annotator", "label"]].apply(tuple, axis=1).isin(pairs)
    ].reset_index(drop=True)
    assert filtered_data.equals(expected)


def test_filter_df_invalid_annotator(sample_df: DataFrame) -> None:
    data = DetectionData(sample_df)
    with pytest.raises(
        ValueError,
        match='Annotator "bbjuni" not in APLOSE DataFrame',
    ):
        data.filter_df(annotator="bbjuni", label="lbl1")


def test_filter_df_invalid_label(sample_df: DataFrame) -> None:
    data = DetectionData(sample_df)
    with pytest.raises(
        ValueError,
        match='Label "cool" not in APLOSE DataFrame',
    ):
        data.filter_df(annotator="ann2", label="cool")


def test_filter_df_invalid_combination(
    sample_df: DataFrame,
) -> None:
    data = DetectionData(sample_df)
    with pytest.raises(
        ValueError,
        match=r"DataFrame with annotator 'ann1' /"
        " label 'lbl3' contains no detection.",
    ):
        data.filter_df(annotator="ann1", label="lbl3")


def test_filter_df_invalid_lists_size(
    sample_df: DataFrame,
) -> None:
    data = DetectionData(sample_df)
    with pytest.raises(
        ValueError,
        match=r"Length of annotator \(2\) and label \(1\) must match.",
    ):
        data.filter_df(annotator=["ann1", "ann2"], label=["lbl2"])


def test_set_ax_uses_2hour_locator(sample_df: DataFrame) -> None:
    data = DetectionData(sample_df)
    _, ax = plt.subplots()
    tick_freq = frequencies.to_offset("2h")

    ax = data.plot.set_ax(
        ax=ax,
        x_ticks_res=tick_freq,
        date_format="%y-%m-%d",
    )

    locator = ax.xaxis.get_major_locator()
    assert isinstance(locator, mdates.HourLocator)
    assert locator._get_interval() == 2  # ruff: ignore[magic-value-comparison]


def test_histo(sample_csv_result: DataFrame, recording_planning_csv: DataFrame) -> None:
    dict_test = {
        "detection_file": sample_csv_result,
        "filename_format": "%Y_%m_%d_%H_%M_%S",
        "recording_file": recording_planning_csv,
    }
    data = DetectionData.from_dict(dict_test)
    _, ax = plt.subplots()
    bin_size = Timedelta("1h")
    tick_freq = frequencies.to_offset("2h")
    ax = data.plot.set_ax(
        ax=ax,
        x_ticks_res=tick_freq,
        date_format="%y-%m-%d",
    )
    data.plot.histogram(
        ax=ax,
        annotator="ann1",
        label="lbl1",
        bin_size=bin_size,
        effort=True,
    )


def test_histo_no_recording_file_provided(
    sample_csv_result: DataFrame,
    recording_planning_csv: DataFrame,
) -> None:
    dict_test = {
        "detection_file": sample_csv_result,
        "filename_format": "%Y_%m_%d_%H_%M_%S",
        "recording_file": None,
    }
    data = DetectionData.from_dict(dict_test)
    _, ax = plt.subplots()
    bin_size = Timedelta("1h")
    tick_freq = frequencies.to_offset("2h")
    ax = data.plot.set_ax(
        ax=ax,
        x_ticks_res=tick_freq,
        date_format="%y-%m-%d",
    )
    with pytest.raises(ValueError, match=r"No recording file provided."):
        data.plot.histogram(
            ax=ax,
            annotator="ann1",
            label="lbl1",
            bin_size=bin_size,
            effort=True,
        )


def test_histo_no_recording_fake_file_provided(
    sample_csv_result: DataFrame,
    recording_planning_csv: DataFrame,
) -> None:
    dict_test = {
        "detection_file": sample_csv_result,
        "filename_format": "%Y_%m_%d_%H_%M_%S",
        "recording_file": Path("fake_file.csv"),
    }
    data = DetectionData.from_dict(dict_test)
    _, ax = plt.subplots()
    bin_size = Timedelta("1h")
    tick_freq = frequencies.to_offset("2h")
    ax = data.plot.set_ax(
        ax=ax,
        x_ticks_res=tick_freq,
        date_format="%y-%m-%d",
    )
    with pytest.raises(FileNotFoundError, match=r"File not found"):
        data.plot.histogram(
            ax=ax,
            annotator="ann1",
            label="lbl1",
            bin_size=bin_size,
            effort=True,
        )


def test_heatmap_wrong_bin(sample_df: DataFrame) -> None:
    data = DetectionData(sample_df)
    data.lon = 0
    data.lat = 0
    bins = frequencies.to_offset("10s")
    _, ax = plt.subplots()
    with pytest.raises(
        ValueError,
        match=r"`bin_size` must be >= 24h for heatmap mode.",
    ):
        data.plot.heatmap(
            ax=ax,
            annotator="ann1",
            label="lbl1",
            bin_size=bins,
        )


def test_plot_histogram_missing_bin_size(monkeypatch, sample_df: DataFrame) -> None:
    obj = DetectionData(sample_df)

    monkeypatch.setattr(obj, "filter_df", lambda annotator, label: sample_df)
    monkeypatch.setattr(
        "disclose.dataclass.detection_data.histo",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "disclose.dataclass.detection_data.get_count",
        lambda *args, **kwargs: None,
    )

    _, ax = plt.subplots()

    with pytest.raises(
        TypeError,
        match=r"missing 1 required positional argument: 'bin_size'",
    ):
        obj.plot.histogram(
            ax=ax,
            annotator="ann1",
            label="lbl1",
        )


def test_plot_agreement_missing_bin_size(monkeypatch, sample_df: DataFrame) -> None:
    obj = DetectionData(sample_df)

    monkeypatch.setattr(obj, "filter_df", lambda annotator, label: sample_df)

    _, ax = plt.subplots()

    with pytest.raises(ValueError, match=r"'bin_size' missing for agreement plot."):
        obj.plot.agreement(
            ax=ax,
            annotator="ann1",
            label="lbl1",
            bin_size=None,
        )


def test_plot_agreement(sample_df: DataFrame) -> None:
    data = DetectionData(sample_df)
    _, ax = plt.subplots()
    data.plot.agreement(
        ax=ax,
        annotator=["ann1", "ann2"],
        label="lbl1",
        bin_size=Timedelta("10s"),
    )


def test_set_ax(sample_df: DataFrame) -> None:
    data = DetectionData(sample_df)
    _, ax = plt.subplots()
    ax = data.plot.set_ax(ax, Timedelta("7h"), "%Y-%m-%d")
    locator = ax.xaxis.get_major_locator()
    assert isinstance(locator, mdates.HourLocator)


def test_from_dict(
    sample_dict: dict,
    sample_df: DataFrame,
) -> None:
    df_from_dict = DetectionData.from_dict(sample_dict).df
    df_expected = (
        DetectionData(sample_df)
        .filter_df(annotator="ann1", label="lbl1")
        .reset_index(drop=True)
    )
    assert df_from_dict.equals(df_expected)


def test_from_dict_concat(
    sample_dict: dict,
    sample_df: DataFrame,
) -> None:
    data_expected = DetectionData.from_dict(sample_dict)
    start = data_expected.start_datetime
    end = data_expected.end_datetime
    median_time = start + (end - start) / 2
    config = [
        {
            "detection_file": sample_dict["detection_file"],
            "end_datetime": median_time,
            "filename_format": sample_dict["filename_format"],
            "annotator": sample_dict["annotator"],
            "label": sample_dict["label"],
        },
        {
            "detection_file": sample_dict["detection_file"],
            "start_datetime": median_time,
            "filename_format": sample_dict["filename_format"],
            "annotator": sample_dict["annotator"],
            "label": sample_dict["label"],
        },
    ]
    data_concat = DetectionData.from_dict(config, merge=True)

    assert all(data_expected.df == data_concat.df)
    assert repr(data_expected) == repr(data_concat)


def test_from_dict_no_concat(
    sample_dict: dict,
    sample_df: DataFrame,
) -> None:
    data_expected = DetectionData.from_dict(sample_dict)
    start = data_expected.start_datetime
    end = data_expected.end_datetime
    median_time = start + (end - start) / 2
    config = [
        {
            "detection_file": sample_dict["detection_file"],
            "end_datetime": median_time,
            "filename_format": sample_dict["filename_format"],
            "annotator": sample_dict["annotator"],
            "label": sample_dict["label"],
        },
        {
            "detection_file": sample_dict["detection_file"],
            "start_datetime": median_time,
            "filename_format": sample_dict["filename_format"],
            "annotator": sample_dict["annotator"],
            "label": sample_dict["label"],
        },
    ]
    data_concat = DetectionData.from_dict(config, merge=False)

    assert isinstance(data_concat, list)
    assert len(data_concat) == len(config)
    assert [isinstance(data, DetectionData) for data in data_concat]


def test_concatenate(sample_dict: Path, sample_df: DataFrame) -> None:
    data1 = DetectionData(sample_df.loc[: len(sample_df) / 2])
    data2 = DetectionData(sample_df.loc[len(sample_df) / 2 :])

    data_concat = DetectionData.merge([data1, data2])
    expected = DetectionData(sample_df)

    attrs = [
        name
        for name in dir(expected)
        if not name.startswith("_") and not callable(getattr(expected, name))
    ]

    for attr in attrs:
        got = getattr(data_concat, attr)
        exp = getattr(expected, attr)

        if isinstance(exp, DataFrame):
            assert got.equals(exp), f"Mismatch in {attr}"
        else:
            assert got == exp, f"Mismatch in {attr}"


def test_concatenate_change_tz(sample_df: DataFrame, caplog) -> None:
    data1 = DetectionData(sample_df.loc[: len(sample_df) / 2])
    data1.change_tz(pytz.timezone("Etc/GMT-7"))
    data2 = DetectionData(sample_df.loc[len(sample_df) / 2 :])

    with caplog.at_level(logging.INFO):
        data_concat = DetectionData.merge([data1, data2])

    assert get_timezone(data_concat.df) == pytz.utc
    assert (
        "Several timezones found in DataFrame, all timestamps are converted to UTC."
        in caplog.text
    )


# %% overview


def test_data_aplose_overview(sample_data_aplose: DetectionData) -> None:
    expected = (
        DataFrame(
            {
                "annotator": ["ann1", "ann2", "ann3", "ann4", "ann5", "ann6"],
                "lbl1": [2, 2, 0, 0, 2, 1],
                "lbl2": [2, 2, 2, 2, 1, 1],
                "lbl3": [0, 0, 0, 0, 1, 0],
            },
        )
        .set_index("annotator")
        .T
    )
    actual = sample_data_aplose.overview(sample_data_aplose.annotator)

    assert expected.equals(actual)


# %% detection_perf
def test_detection_perf_multiple_timebins(sample_df: DataFrame) -> None:
    obj = DetectionData(sample_df)

    with pytest.raises(ValueError, match="Multiple time bins detected"):
        obj.detection_perf(
            annotator=("ann1", "ann2"),
            label=("lbl1", "lbl2"),
        )


def test_detection_perf(sample_csv_result: Path, recording_planning_csv: Path) -> None:
    config = {
        "detection_file": sample_csv_result,
        "filename_format": "%Y_%m_%d_%H_%M_%S",
        "timebin_new": Timedelta("10min"),
        "recording_file": recording_planning_csv,
    }

    obj = DetectionData.from_dict(config=config)

    obj.detection_perf(
        annotator=("ann1", "ann2"),
        label=("lbl1", "lbl2"),
    )


# %% Reshape


@pytest.mark.parametrize(
    ("begin", "end", "expected"),
    [
        pytest.param(
            Timestamp("2025-01-26T06:20:09.999+00:00"),
            None,
            nullcontext(),
            id="new_begin_after_original_end",
        ),
        pytest.param(
            None,
            Timestamp("2025-01-25T06:20:00.001+00:00"),
            nullcontext(),
            id="new_end_before_original_begin",
        ),
        pytest.param(
            Timestamp("2024-12-31"),
            Timestamp("2024-01-01"),
            pytest.raises(
                ValueError,
                match=r"Begin timestamp is not anterior than end timestamp.",
            ),
            id="begin_after_end_inverted_range",
        ),
        pytest.param(
            Timestamp("2050-01-01", tz="UTC"),
            Timestamp("2050-12-31", tz="UTC"),
            nullcontext(),
            id="tz_aware_future_range_no_data",
        ),
        pytest.param(
            Timestamp("1990-01-01", tz="America/New_York"),
            Timestamp("1990-12-31", tz="America/New_York"),
            nullcontext(),
            id="tz_aware_past_range_no_data",
        ),
    ],
)
def test_reshape_errors(
    sample_data_aplose: DetectionData,
    begin: Timestamp | None,
    end: Timestamp | None,
    expected: ContextManager[Exception],
) -> None:
    """Test that reshape function handles error cases appropriately."""
    with expected:
        sample_data_aplose.reshape(begin, end)


@pytest.mark.parametrize(
    ("begin", "end", "should_filter"),
    [
        pytest.param(
            None,
            None,
            False,
            id="no_timestamps_provided",
        ),
        pytest.param(
            Timestamp("1990-01-01", tz="UTC"),
            None,
            True,
            id="tz_aware_begin_only",
        ),
        pytest.param(
            None,
            Timestamp("2050-12-31", tz="UTC"),
            True,
            id="tz_aware_end_only",
        ),
        pytest.param(
            Timestamp("2025-01-24", tz="Europe/Paris"),
            Timestamp("2025-01-27", tz="Europe/Paris"),
            True,
            id="tz_aware_both_timestamps",
        ),
        pytest.param(
            Timestamp("1990-01-01"),
            None,
            True,
            id="tz_naive_begin_only",
        ),
        pytest.param(
            None,
            Timestamp("2050-12-31"),
            True,
            id="tz_naive_end_only",
        ),
        pytest.param(
            Timestamp("1990-01-01"),
            Timestamp("2050-12-31"),
            True,
            id="tz_naive_both_timestamps",
        ),
        pytest.param(
            Timestamp("1990-01-01", tz="Europe/Paris"),
            Timestamp("2050-12-31", tz="America/New_York"),
            True,
            id="tz_aware_different_timezone",
        ),
        pytest.param(
            Timestamp("2025-01-26T00:00:00.000+00:00"),
            Timestamp("2025-01-26T10:00:00.000+00:00"),
            True,
            id="narrowing_detections",
        ),
    ],
)
def test_reshape_valid_cases(
    sample_data_aplose: DetectionData,
    begin: Timestamp | None,
    end: Timestamp | None,
    should_filter: bool,
) -> None:
    """Test that reshape handles valid timestamp cases appropriately."""
    reshaped = copy(sample_data_aplose)
    reshaped.reshape(start_datetime=begin, end_datetime=end)

    original_tz = get_timezone(sample_data_aplose.df)

    # Check timestamps were updated
    if begin is not None:
        if begin.tz is None:
            begin = begin.tz_localize(original_tz)
        assert reshaped.start_datetime != sample_data_aplose.start_datetime
        assert reshaped.start_datetime == begin

    if end is not None:
        if end.tz is None:
            end = end.tz_localize(original_tz)
        assert reshaped.end_datetime != sample_data_aplose.end_datetime
        assert reshaped.end_datetime == end

    # Check timezone was applied for tz-naive timestamps
    assert reshaped.start_datetime.tz is not None
    assert reshaped.end_datetime.tz is not None

    # Check filtering behavior
    if should_filter:
        assert all(reshaped.df["start_datetime"] >= reshaped.start_datetime)
        assert all(reshaped.df["end_datetime"] <= reshaped.end_datetime)


# %%


def test_get_locator_from_offset(monkeypatch) -> None:
    class FakeSecond:
        def __init__(self, interval) -> None:
            self.interval = interval

    class FakeMinute:
        def __init__(self, interval) -> None:
            self.interval = interval

    monkeypatch.setattr(mdates, "SecondLocator", FakeSecond)
    monkeypatch.setattr(mdates, "MinuteLocator", FakeMinute)

    @pytest.mark.parametrize(
        "offset, expected, interval",
        [
            (5, FakeSecond, 5),
            (Timedelta("45s"), FakeSecond, 45),
            (Timedelta("120s"), FakeMinute, 2),
        ],
    )
    def check_valid(offset, expected, interval):
        loc = _get_locator_from_offset(offset)
        assert isinstance(loc, expected)
        assert loc.interval == interval

    for offset, expected, interval in [
        (5, FakeSecond, 5),
        (Timedelta("45s"), FakeSecond, 45),
        (Timedelta("120s"), FakeMinute, 2),
    ]:
        loc = _get_locator_from_offset(offset)
        assert isinstance(loc, expected)
        assert loc.interval == interval


def test_get_locator_from_offset_unsupported() -> None:
    with pytest.raises(ValueError, match="Unsupported offset type"):
        _get_locator_from_offset(3.14)


# %% polar


def test_polar_plot(sample_csv_result: Path, polar_ax: Axes) -> None:
    config = {
        "detection_file": sample_csv_result,
        "filename_format": "%Y_%m_%d_%H_%M_%S",
        "timebin_new": Timedelta(seconds=60),
    }

    obj = DetectionData.from_dict(config=config)

    obj.plot.polar(
        annotator="ann1",
        label="lbl1",
        ax=polar_ax,
    )


# %% scatter


def test_scatter_plot(sample_csv_result: Path, recording_planning_csv: Path) -> None:
    config = {
        "detection_file": sample_csv_result,
        "filename_format": "%Y_%m_%d_%H_%M_%S",
        "timebin_new": Timedelta(seconds=60),
        "recording_file": recording_planning_csv,
    }

    obj = DetectionData.from_dict(config=config)
    _, ax = plt.subplots()

    obj.plot.scatter(
        annotator="ann1",
        label="lbl1",
        ax=ax,
        show_rise_set=True,
        coordinates=(0, 0),
        effort=True,
        bin_size=Timedelta("1h"),
    )


# %% timeline


def test_timeline_plot(sample_csv_result: Path) -> None:
    config = {
        "detection_file": sample_csv_result,
        "filename_format": "%Y_%m_%d_%H_%M_%S",
        "timebin_new": Timedelta(seconds=60),
    }

    obj = DetectionData.from_dict(config=config)
    _, ax = plt.subplots()

    obj.plot.timeline(
        annotator="ann1",
        label="lbl1",
        ax=ax,
        coordinates=(0, 0),
    )


# %% __post_init__ / _localize


def test_post_init_missing_columns(sample_df: DataFrame) -> None:
    incomplete = sample_df.drop(columns=["annotator", "label"])

    with pytest.raises(
        ValueError,
        match=r"not APLOSE-formatted, missing columns: \['label', 'annotator'\]",
    ):
        DetectionData(incomplete)


def test_localize_keeps_aware_timestamp(sample_df: DataFrame) -> None:
    data = DetectionData(sample_df)
    aware = Timestamp("2025-01-01 00:00:00", tz="Etc/GMT-7")

    assert data._localize(aware) == aware


def test_localize_uses_data_timezone(sample_df: DataFrame) -> None:
    data = DetectionData(sample_df)
    data.change_tz(pytz.timezone("Etc/GMT-7"))

    localized = data._localize(Timestamp("2025-01-01 00:00:00"))

    assert localized == Timestamp("2025-01-01 00:00:00", tz="Etc/GMT-7")


def test_localize_several_timezones(sample_df: DataFrame, monkeypatch) -> None:
    data = DetectionData(sample_df)
    monkeypatch.setattr(
        "disclose.dataclass.detection_data.get_timezone",
        lambda _: [pytz.utc, pytz.timezone("Etc/GMT-7")],
    )

    with pytest.raises(TypeError, match="several timezones"):
        data._localize(Timestamp("2025-01-01 00:00:00"))


def test_reshape_naive_bounds_with_several_timezones(
    sample_df: DataFrame,
    monkeypatch,
) -> None:
    data = DetectionData(sample_df)
    monkeypatch.setattr(
        "disclose.dataclass.detection_data.get_timezone",
        lambda _: [pytz.utc, pytz.timezone("Etc/GMT-7")],
    )

    with pytest.raises(TypeError, match="several timezones"):
        data.reshape(start_datetime=Timestamp("2025-01-25 06:20:00"))


# %% merge / copy / reshape / filter


@pytest.mark.parametrize(
    ("coordinates", "expected"),
    [
        pytest.param([(48.39, -4.49), None], (48.39, -4.49), id="only_one_has_some"),
        pytest.param(
            [(48.39, -4.49), (48.39, -4.49)],
            (48.39, -4.49),
            id="all_agree",
        ),
        pytest.param([(48.39, -4.49), (43.0, 5.0)], (None, None), id="conflict"),
        pytest.param([None, None], (None, None), id="none_set"),
    ],
)
def test_merge_coordinates(
    sample_df: DataFrame,
    coordinates: list[tuple[float, float] | None],
    expected: tuple[float | None, float | None],
) -> None:
    half = len(sample_df) // 2
    data1 = DetectionData(sample_df.iloc[:half])
    data2 = DetectionData(sample_df.iloc[half:])
    for data, coord in zip((data1, data2), coordinates, strict=True):
        if coord is not None:
            data.coordinates = coord

    merged = DetectionData.merge([data1, data2])

    assert merged.coordinates == expected


def test_merge_empty_list() -> None:
    with pytest.raises(ValueError, match="No DataAplose to merge"):
        DetectionData.merge([])


def test_copy_is_independent(sample_data_aplose: DetectionData) -> None:
    sample_data_aplose.coordinates = (48.39, -4.49)
    sample_data_aplose.reshape(
        start_datetime=Timestamp("2025-01-25 06:20:00", tz="UTC"),
        end_datetime=Timestamp("2025-01-25 06:40:00", tz="UTC"),
    )

    clone = sample_data_aplose.copy()

    assert clone is not sample_data_aplose
    assert clone.df is not sample_data_aplose.df
    assert clone.df.equals(sample_data_aplose.df)
    assert clone.config is not sample_data_aplose.config
    assert clone.coordinates == (48.39, -4.49)
    assert clone.start_datetime == sample_data_aplose.start_datetime
    assert clone.end_datetime == sample_data_aplose.end_datetime

    clone.df.loc[clone.df.index[0], "label"] = "changed"
    clone.change_tz("Etc/GMT-7")

    assert "changed" not in set(sample_data_aplose.df["label"])
    assert get_timezone(sample_data_aplose.df) == pytz.utc


def test_reshape_empty_dataframe_is_noop() -> None:
    data = DetectionData()

    result = data.reshape(
        start_datetime=Timestamp("2025-01-01", tz="UTC"),
        end_datetime=Timestamp("2025-01-02", tz="UTC"),
    )

    assert result is data
    assert data.df.empty
    assert data.start_datetime is None
    assert data.end_datetime is None


def test_reshape_without_bounds_is_noop(sample_df: DataFrame) -> None:
    data = DetectionData(sample_df)

    result = data.reshape()

    assert result is data
    assert data.df.equals(sample_df)


def test_filter_returns_independent_filtered_instance(
    sample_df: DataFrame,
) -> None:
    data = DetectionData(sample_df)
    data.coordinates = (48.39, -4.49)
    n_rows = len(data.df)

    filtered = data.filter("ann1", "lbl1")

    assert filtered is not data
    assert len(data.df) == n_rows
    assert set(filtered.df["annotator"]) == {"ann1"}
    assert set(filtered.df["label"]) == {"lbl1"}
    assert filtered.df.index.tolist() == list(range(len(filtered.df)))
    assert filtered.coordinates == (48.39, -4.49)
    assert filtered.config == data.config


def test_filter_multiple_pairs(sample_df: DataFrame) -> None:
    filtered = DetectionData(sample_df).filter(["ann1", "ann2"], ["lbl1", "lbl2"])

    pairs = set(zip(filtered.df["annotator"], filtered.df["label"], strict=True))
    assert pairs == {("ann1", "lbl1"), ("ann2", "lbl2")}


def test_filter_unknown_annotator(sample_df: DataFrame) -> None:
    with pytest.raises(ValueError, match='Annotator "nobody" not in APLOSE'):
        DetectionData(sample_df).filter("nobody", "lbl1")


# %% PlotAccessor


@pytest.mark.parametrize("kind", ["histogram", "polar", "heatmap"])
def test_plot_call_dispatches_to_method(
    sample_df: DataFrame,
    monkeypatch,
    kind: str,
) -> None:
    data = DetectionData(sample_df)
    calls: list[tuple] = []
    monkeypatch.setattr(
        type(data.plot),
        kind,
        lambda self, *args, **kwargs: calls.append((args, kwargs)) or "result",
    )

    result = data.plot(kind, "ann1", "lbl1", bin_size=Timedelta("1D"))

    assert result == "result"
    assert calls == [(("ann1", "lbl1"), {"bin_size": Timedelta("1D")})]


def test_plot_call_unknown_kind(sample_df: DataFrame) -> None:
    with pytest.raises(ValueError, match="Unknown plot kind 'pie'"):
        DetectionData(sample_df).plot("pie")


def test_get_ax_returns_provided_axes(ax: Axes) -> None:
    assert PlotAccessor._get_ax(ax) is ax


def test_get_ax_creates_axes() -> None:
    created = PlotAccessor._get_ax()

    assert isinstance(created, Axes)
    assert created.name == "rectilinear"
    plt.close("all")


def test_get_ax_creates_polar_axes() -> None:
    created = PlotAccessor._get_ax(polar=True)

    assert created.name == "polar"
    plt.close("all")


@pytest.mark.parametrize(
    "coordinates",
    [
        pytest.param((None, None), id="none_none"),
        pytest.param((48.39, None), id="lat_only"),
        pytest.param((None, -4.49), id="lon_only"),
    ],
)
def test_coordinates_required_but_missing(
    sample_df: DataFrame,
    coordinates: tuple[float | None, float | None],
) -> None:
    data = DetectionData(sample_df)
    data.lat, data.lon = coordinates

    with pytest.raises(ValueError, match="Coordinates are required"):
        data.plot._coordinates(None, required=True)


def test_coordinates_not_required_returns_unset(sample_df: DataFrame) -> None:
    assert DetectionData(sample_df).plot._coordinates(None, required=False) == (
        None,
        None,
    )


def test_coordinates_override_takes_precedence(sample_df: DataFrame) -> None:
    data = DetectionData(sample_df)
    data.coordinates = (48.39, -4.49)

    assert data.plot._coordinates((43.0, 5.0), required=True) == (43.0, 5.0)
    assert data.plot._coordinates(None, required=True) == (48.39, -4.49)
