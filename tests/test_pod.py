"""FPOD/ CPOD processing functions tests."""

from datetime import tzinfo
from pathlib import Path
from types import SimpleNamespace

import pytest
import pytz
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy import allclose, array, exp
from pandas import DataFrame, Series, Timedelta, Timestamp

from disclose.utils.pod import (
    _mixture_density,
    _shade_missing_bars,
    calendar,
    cluster_info,
    compute_ici,
    create_matrix,
    fit_gmm,
    get_feeding_buzz_datetime,
    get_group_colors,
    gmm_feeding_buzz,
    load_pod_folder,
    percent_barplot,
    percent_calc,
    plot_gmm_ici,
    pod2aplose,
    process_feeding_buzz,
    process_timelost,
)

CLICKS_CPOD = """Minute,microsec,cycles,SPL_Pa,kHz,Bandwidth,end kHz,Qn,TrN
25/1/2019 11:45,55643215,7,38,130,0,121,2,38
25/1/2019 11:45,55707365,7,44,130,0,125,2,38
25/1/2019 11:45,55770865,7,36,132,0,131,2,38
25/1/2019 11:45,55830500,11,34,136,1,108,2,38
25/1/2019 11:45,55890495,10,33,135,1,131,2,38
"""

CLICKS_FPOD = """File,Minute,microsec,ICI,TrnAvPRF,Ncyc,ClkKHZ,IPIbefore,IPIatMax,IPIplus1,IPIplus2,EndIPI,ClkIPIrange,maxPk,maxPkE,Pkminus1%,Pkplus1%,PkAt,AmpReversals,tRateScore,Qn,TrnIDn,ClassID,Log(PRF)*10
CETIROISEPHASE1POINTB 2022 05 05 FPOD_6669 file0.FP3,64358756,40266515,10595,98,11,121,256,33,34,34,31,3,78,78,98,91,6,1,10,2,1,0,19
CETIROISEPHASE1POINTB 2022 05 05 FPOD_6669 file0.FP3,64358756,40276675,10160,98,11,121,256,33,33,33,33,3,79,79,98,91,5,1,10,2,1,0,19
CETIROISEPHASE1POINTB 2022 05 05 FPOD_6669 file0.FP3,64358756,40286600,9925,98,11,121,256,33,33,33,33,2,84,84,88,94,4,1,10,2,1,0,20
CETIROISEPHASE1POINTB 2022 05 05 FPOD_6669 file0.FP3,64358756,40296440,9840,98,10,121,256,33,33,34,33,3,79,79,91,100,4,1,10,2,1,0,20
CETIROISEPHASE1POINTB 2022 05 05 FPOD_6669 file0.FP3,64358756,40306520,10080,98,11,121,256,33,33,34,33,3,76,76,92,96,4,1,10,2,1,0,19
"""

TIMELOST = """File	podN,ChunkEnd,Minute,Temp,Angle,MinutesON,NBHF_DPM,DPM,Nfiltered/m,kHz_continuous_noise,NBHFclx,DOL_DPM,DOLclx,SONAR_DPM,SONARclx,Nall/m,%TimeLost,%m SonarRisk,%mSediment noise,LandmarkSeq_total,avOpThreshold
CETIROISEPHASE1POINTB 2022 05 05 FPOD_6669 file0.FP3,6669,05/05/2022 10:59,64348499,21.4,0,0m ON,0,108,14,0,0,0,0,0,0,,,0,0,0,0
CETIROISEPHASE1POINTB 2022 05 05 FPOD_6669 file0.FP3,6669,05/05/2022 11:59,64348559,21.4,0,0m ON,0,108,14,0,0,0,0,0,0,548.9,100,0,0,0,0
CETIROISEPHASE1POINTB 2022 05 05 FPOD_6669 file0.FP3,6669,05/05/2022 12:59,64348619,22.4,0,0,0,81.6,60,0,0,0,0,0,0,0.2,100,0,0,0,0
CETIROISEPHASE1POINTB 2022 05 05 FPOD_6669 file0.FP3,6669,05/05/2022 13:59,64348679,23,4,1.62,20,78,60,0,0,0,0,0,0,0,100,0,0,0,0
CETIROISEPHASE1POINTB 2022 05 05 FPOD_6669 file0.FP3,6669,05/05/2022 14:59,64348739,23,3,0.28,0,78,60,0,0,0,0,0,0,0,100,0,0,0,0
"""


@pytest.fixture
def pod_dataframe() -> DataFrame:
    return DataFrame({
        "File": [
            "Site A ile Haute 2019 01 25 POD3055 file01.CP3",
            "Site A ile Haute 2019 01 25 POD3055 file01.CP3",
            "Site A ile Haute 2019 01 25 POD3055 file01.CP3",
            "Site A ile Haute 2019 01 25 POD3055 file01.CP3",
            "Site A ile Haute 2019 01 25 POD3055 file01.CP3",
        ],
        "podN": [6669, 6669, 6669, 6669, 6669],
        "ChunkEnd": [
            "24/01/2019 06:17",
            "24/01/2019 06:18",
            "24/01/2019 06:19",
            "24/01/2019 06:20",
            "24/01/2019 06:21",
        ],
        "Minute": [64348546, 64348547, 64348548, 64348549, 64348550],
        "DPM": [0, 1, 1, 0, 0],
        "Nall": [0, 216, 75, 0, 28],
        "MinsOn": [0, 1, 1, 1, 1],
    })


@pytest.fixture
def click_dataframe() -> DataFrame:
    return DataFrame({
        "File": [
            "CETIROISEPHASE1POINTB 2022 05 05 FPOD_6669 file0.FP3",
            "CETIROISEPHASE1POINTB 2022 05 05 FPOD_6669 file0.FP3",
            "CETIROISEPHASE1POINTB 2022 05 05 FPOD_6669 file0.FP3",
            "CETIROISEPHASE1POINTB 2022 05 05 FPOD_6669 file0.FP3",
            "CETIROISEPHASE1POINTB 2022 05 05 FPOD_6669 file0.FP3",
        ],
        "microsec": [40255920, 40266515, 40276675, 40286600, 40296440],
        "Minute": [64348546, 64348547, 64348548, 64348549, 64348550],
    })


@pytest.fixture
def pod_aplose(sample_df: DataFrame) -> DataFrame:
    sample_df["type"] = "WEAK"
    return sample_df


# csv_folder
def test_folder_multiple(pod_dataframe: DataFrame, tmp_path: Path) -> None:
    """Test processing multiple CSV files."""
    folder = tmp_path / "pod_folder"
    folder.mkdir(parents=True, exist_ok=True)

    pod_dataframe.to_csv(folder / "pod_dataframe1.csv", index=False)
    pod_dataframe.to_csv(folder / "pod_dataframe2.csv", index=False)

    result = load_pod_folder(folder, ext="csv")

    assert isinstance(result, DataFrame)
    assert set(result["dataset"]) == {"pod_dataframe1", "pod_dataframe2"}
    assert list(result.columns) == [
        "File",
        "podN",
        "ChunkEnd",
        "Minute",
        "DPM",
        "Nall",
        "MinsOn",
        "dataset",
        "Datetime",
    ]


def test_folder_single_txt(
    monkeypatch: pytest.MonkeyPatch,
    click_dataframe: DataFrame,
    tmp_path: Path,
) -> None:
    """Test processing a single CSV file."""
    monkeypatch.setattr(
        "disclose.utils.pod.process_feeding_buzz",
        lambda df, species: df,
    )
    txt_file = tmp_path / "click_folder" / "click_dataframe.txt"
    txt_file.parent.mkdir(parents=True, exist_ok=True)
    click_dataframe.to_csv(txt_file, index=False)
    result = load_pod_folder(txt_file.parent, ext="txt")

    assert isinstance(result, DataFrame)
    assert "dataset" in result.columns
    assert all(result["dataset"] == "click_dataframe")
    assert list(result.columns) == [
        "File",
        "microsec",
        "Minute",
        "dataset",
        "Datetime",
    ]


def test_folder_multiple_txt(click_dataframe: DataFrame, tmp_path: Path) -> None:
    """Test processing multiple txt files."""
    folder = tmp_path / "click_folder"
    folder.mkdir(parents=True, exist_ok=True)

    click_dataframe.to_csv(folder / "click_dataframe1.txt", index=False)
    click_dataframe.to_csv(folder / "click_dataframe2.txt", index=False)

    result = load_pod_folder(folder, ext="txt")

    assert isinstance(result, DataFrame)
    assert "dataset" in result.columns
    assert set(result["dataset"]) == {"click_dataframe1", "click_dataframe2"}
    assert list(result.columns) == [
        "File",
        "microsec",
        "Minute",
        "dataset",
        "Datetime",
    ]


@pytest.mark.parametrize(
    ("mocked_df", "should_raise"),
    [
        pytest.param(
            DataFrame({
                "ChunkEnd": ["01/01/2024 12:00"],
                "DPM": [1],
                "MinsOn": [30.0],
                "microsec": [100],
            }),
            False,
            id="valid-dpm-columns",
        ),
        pytest.param(
            DataFrame({
                "ChunkEnd": ["01/01/2024 12:00"],
                "%TimeLost": [0.1],
                "Nall/m": [1.0],
                "File": ["f1"],
                "microsec": [100],
            }),
            False,
            id="valid-timelost-columns",
        ),
        pytest.param(
            DataFrame({
                "ChunkEnd": ["01/01/2024 12:00"],
                "col1": [0.1],
                "Nall/m": [1.0],
                "File": ["f1"],
                "microsec": [100],
            }),
            True,
            id="invalid-missing-timelost",
        ),
        pytest.param(
            DataFrame({
                "ChunkEnd": ["01/01/2024 12:00"],
                "%TimeLost": [0.1],
                "col1": [1.0],
                "File": ["f1"],
                "microsec": [100],
            }),
            True,
            id="invalid-missing-nall",
        ),
        pytest.param(
            DataFrame({
                "ChunkEnd": ["01/01/2024 12:00"],
                "File": ["f1"],
                "col1": [1],
                "MinsOn": ["x"],
                "microsec": [100],
            }),
            True,
            id="invalid-missing-dpm",
        ),
        pytest.param(
            DataFrame({
                "ChunkEnd": ["01/01/2024 12:00"],
                "File": ["f1"],
                "DPM": [1],
                "col3": ["x"],
                "microsec": [100],
            }),
            True,
            id="invalid-missing-minson",
        ),
        pytest.param(
            DataFrame({"col1": [1], "col2": [2], "col3": [3]}),
            True,
            id="invalid-no-required-columns",
        ),
    ],
)
def test_right_csv_format(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mocked_df: DataFrame,
    should_raise: bool,
) -> None:
    """Mocked read_csv to test load_pod_folder column validation."""
    fake_path = Path("fake/deploy_01.csv")

    monkeypatch.setattr(Path, "rglob", lambda self, pattern: [fake_path])
    monkeypatch.setattr("disclose.utils.pod.find_delimiter", lambda f: ";")
    monkeypatch.setattr(
        "disclose.utils.pod.read_csv",
        lambda *args, **kwargs: mocked_df,
    )

    if should_raise:
        with pytest.raises((ValueError, KeyError)):
            load_pod_folder(Path("fake/folder"), "csv")
    else:
        result = load_pod_folder(Path("fake/folder"), "csv")
        assert isinstance(result, DataFrame)


# pod2aplose
@pytest.fixture
def sample_df() -> DataFrame:
    return DataFrame({
        "Datetime": [
            Timestamp("15-01-2024 10:30:00"),
            Timestamp("15-01-2024 11:00:00"),
            Timestamp("15-01-2024 09:45:00"),
        ],
        "Deploy": ["deploy1", "deploy2", "deploy1"],
    })


@pytest.fixture
def empty_df() -> DataFrame:
    return DataFrame({
        "Datetime": [],
        "Deploy": [],
    })


@pytest.fixture
def timezone() -> tzinfo:
    return pytz.UTC


def test_pod2aplose_basic_structure(sample_df: DataFrame, timezone: tzinfo) -> None:
    """Test that basic structure and required columns are present."""
    result = pod2aplose(
        df=sample_df,
        tz=pytz.UTC,
        dataset_name="dataset",
        label="porpoise",
        annotator="fpod",
        bin_size=Timedelta(seconds=60),
    )

    expected_columns = [
        "dataset",
        "filename",
        "start_time",
        "end_time",
        "min_frequency",
        "max_frequency",
        "label",
        "annotator",
        "start_datetime",
        "end_datetime",
        "type",
    ]

    assert isinstance(result, DataFrame)
    assert list(result.columns) == expected_columns
    assert len(result) == len(sample_df)
    assert result["dataset"].iloc[0] == "dataset"
    assert all(result["dataset"] == "dataset")
    assert result["filename"].iloc[0] != 0
    assert all(result["start_time"] == 0)
    assert all(result["end_time"] == 60)
    assert all(result["min_frequency"] == 0)
    assert all(result["max_frequency"] == 0)
    assert all(result["label"] == "porpoise")
    assert all(result["annotator"] == "fpod")
    assert len(result["start_datetime"].iloc[0]) > 0
    assert len(result["end_datetime"].iloc[0]) > 0
    assert len(result["dataset"]) == len(sample_df)


def test_pod2aplose_empty_dataframe(empty_df: DataFrame, timezone: tzinfo) -> None:
    """Test handling of empty DataFrame."""
    result = pod2aplose(
        df=empty_df,
        tz=pytz.UTC,
        dataset_name="dataset",
        label="porpoise",
        annotator="fpod",
        bin_size=Timedelta(seconds=60),
    )

    assert len(result) == 0
    assert list(result.columns) == [
        "dataset",
        "filename",
        "start_time",
        "end_time",
        "min_frequency",
        "max_frequency",
        "label",
        "annotator",
        "start_datetime",
        "end_datetime",
        "type",
    ]


# process_feeding_buzz
@pytest.fixture
def sample_fb() -> DataFrame:
    return DataFrame({
        "Datetime": [
            Timestamp("2018-10-26 08:47:21.524095"),
            Timestamp("2018-10-26 08:47:21.561215"),
            Timestamp("2018-10-26 08:47:21.597925"),
            Timestamp("2018-10-26 08:47:21.706350"),
            Timestamp("2018-10-26 08:47:21.934405"),
            Timestamp("2019-05-03 19:55:05.985310"),
            Timestamp("2019-05-03 19:55:05.983675"),
            Timestamp("2019-05-03 19:55:05.982035"),
            Timestamp("2019-05-15 01:38:25.499480"),
        ],
        "dataset": [
            "deploy1",
            "deploy1",
            "deploy1",
            "deploy2",
            "deploy2",
            "deploy2",
            "deploy3",
            "deploy3",
            "deploy3",
        ],
    })


def test_fit_gmm_output(sample_fb: DataFrame) -> None:
    """Test that basic structure and required columns are present."""
    clustering, ici_log, _ = fit_gmm(
        df=sample_fb,
        comp=3,
    )

    expected_columns = [
        "Datetime",
        "dataset",
        "ICI_minutes",
        "cluster",
    ]

    assert isinstance(clustering, DataFrame)
    assert list(clustering.columns) == expected_columns
    assert clustering["Datetime"].iloc[0] != sample_fb["Datetime"].iloc[0]
    assert set(clustering["dataset"]) == {"deploy1", "deploy2", "deploy3"}
    assert len(clustering["cluster"].unique()) == 3
    assert len(clustering) == len(ici_log)


@pytest.mark.parametrize(
    ("mocked_species", "mocked_df", "should_raise"),
    [
        pytest.param(
            "commerson",
            DataFrame({
                "Datetime": [
                    Timestamp("2019-04-16 16:06:19.948345"),
                    Timestamp("2019-04-16 16:06:19.950840"),
                    Timestamp("2019-04-16 16:06:19.953345"),
                ],
            }),
            False,
            id="valid-species-commerson",
        ),
        pytest.param(
            "porpoise",
            DataFrame({
                "Datetime": [
                    Timestamp("2020-04-09 13:51:24.133750"),
                    Timestamp("2020-04-09 13:51:24.124155"),
                    Timestamp("2020-04-09 13:51:24.114335"),
                    Timestamp("2020-04-09 13:51:24.104345"),
                ],
            }),
            False,
            id="valid-species-porpoise",
        ),
        pytest.param(
            "delphinid",
            DataFrame({
                "Datetime": [
                    Timestamp("2019-05-14 00:16:41.327605"),
                    Timestamp("2019-05-14 00:16:41.345310"),
                    Timestamp("2019-05-14 00:16:41.363285"),
                    Timestamp("2019-05-14 00:16:41.382405"),
                ],
            }),
            False,
            id="valid-species-delphinid",
        ),
        pytest.param(
            "elephant",
            DataFrame({
                "Datetime": [
                    Timestamp("2020-04-16 00:06:32.327605"),
                    Timestamp("2020-04-16 00:06:32.345310"),
                    Timestamp("2020-04-16 00:06:32.363285"),
                    Timestamp("2020-04-16 00:06:32.382405"),
                ],
            }),
            True,
            id="invalid-species",
        ),
    ],
)
def test_species_clustering(
    mocked_species: str,
    mocked_df: DataFrame,
    should_raise: bool,
) -> None:
    """Mocked read_csv to test load_pod_folder column validation."""
    if should_raise:
        with pytest.raises((ValueError, KeyError)):
            process_feeding_buzz(df=mocked_df, species=mocked_species)
        return

    result = process_feeding_buzz(df=mocked_df, species=mocked_species)

    assert isinstance(result, DataFrame)
    assert list(result.columns) == ["Datetime", "Buzz", "fbm_count"]
    assert len(result) == 1


# process_timelost
@pytest.fixture
def sample_tl_df() -> DataFrame:
    return DataFrame({
        "File": ["filename1", "filename2", "filename1"],
        "Temp": [20.9, 20.1, 0],
        "Angle": [0, 0, 100],
        "%TimeLost": [0, 0, 100],
        "Deploy": ["deploy1", "deploy2", "deploy1"],
        "Datetime": [
            Timestamp("2022-11-30 10:59:00"),
            Timestamp("2022-11-30 11:59:00"),
            Timestamp("2022-11-30 12:59:00"),
        ],
        "Nall/m": [57, 106, 0],
    })


def test_timelost_process(sample_tl_df: DataFrame, timezone: tzinfo) -> None:
    """Test that basic structure and required columns are present."""
    result = process_timelost(
        df=sample_tl_df,
        threshold=50,
    )

    expected_columns = [
        "File",
        "Temp",
        "Angle",
        "%TimeLost",
        "Deploy",
        "Datetime",
    ]

    assert isinstance(result, DataFrame)
    assert list(result.columns) == expected_columns
    assert set(result["Deploy"]) == {"deploy1", "deploy2"}


# Additional POD utility tests
@pytest.mark.parametrize(
    ("minute", "microsec", "expected"),
    [
        (1, 500_000, Timestamp("1899-12-30 00:01:00.500")),
        ("25/1/2019 11:45", 500_000, Timestamp("2019-01-25 11:45:00.500")),
    ],
)
def test_get_feeding_buzz_datetime(
    minute: int | str,
    microsec: int,
    expected: Timestamp,
) -> None:
    row = Series({"Minute": minute, "microsec": microsec})
    assert get_feeding_buzz_datetime(row) == expected


def test_get_feeding_buzz_datetime_invalid() -> None:
    with pytest.raises(ExceptionGroup, match="Could not convert"):
        get_feeding_buzz_datetime(Series({"Minute": "invalid", "microsec": 10}))


def test_compute_ici() -> None:
    df = DataFrame({
        "Datetime": [
            Timestamp("2024-01-01 00:00:00"),
            Timestamp("2024-01-01 00:01:00"),
            Timestamp("2024-01-01 00:01:00"),
            Timestamp("2024-01-01 00:03:00"),
        ],
    })
    result = compute_ici(df)
    assert result["ICI_minutes"].tolist() == [1.0, 2.0]
    assert "ICI_minutes" not in df.columns


def test_create_matrix() -> None:
    df = DataFrame({"Site": ["A", "A", "B", "B"], "DPM": [2, 4, 6, 8]})
    result = create_matrix(df, ["Site"], ["DPM"])
    assert result.columns.tolist() == ["Site", "DPM_mean", "DPM_std"]
    assert result["DPM_mean"].tolist() == [3.0, 7.0]
    assert allclose(result["DPM_std"], [2**0.5, 2**0.5])


def test_percent_calc() -> None:
    df = DataFrame({
        "Site": ["A", "A", "B"],
        "DPh": [1, 0, 0],
        "FBh": [1, 0, 0],
        "dpm_count": [30, 0, 0],
        "fbm_count": [6, 0, 0],
        "Day": [1, 1, 1],
    })
    result = percent_calc(df, "Site").set_index("Site")
    assert result.loc["A", "%click"] == 25
    assert result.loc["A", "%buzzes"] == 5
    assert result.loc["A", "%DPh"] == 50
    assert result.loc["A", "%FBh"] == 50
    assert result.loc["A", "FBR"] == 20
    assert result.loc["B", "FBR"] == 0


@pytest.mark.parametrize("threshold", [-1, 101, 1.5])
def test_process_timelost_invalid_threshold(threshold: float) -> None:
    with pytest.raises(ValueError, match="Threshold"):
        process_timelost(DataFrame(), threshold=threshold)


def test_get_group_colors() -> None:
    df = DataFrame({"Site": ["A", "B", "A"]})
    colors = get_group_colors(df, "Site")
    assert list(colors) == ["A", "B"]
    assert colors["A"] != colors["B"]


def test_shade_missing_bars() -> None:
    fig, ax = plt.subplots()
    try:
        _shade_missing_bars(ax, array([False, True, True, False, True]))
        assert len(ax.patches) == 2
    finally:
        plt.close(fig)


@pytest.fixture
def simple_gmm() -> SimpleNamespace:
    return SimpleNamespace(
        n_components=2,
        means_=array([[1.0], [0.0]]),
        covariances_=array([[[4.0]], [[1.0]]]),
        weights_=array([0.25, 0.75]),
    )


def test_cluster_info(simple_gmm: SimpleNamespace) -> None:
    result = cluster_info(simple_gmm)
    assert [item["name"] for item in result] == ["Buzz ICIs", "Regular ICIs"]
    assert [item["id"] for item in result] == [0, 1]
    assert allclose([item["mean_log"] for item in result], [0, 1])
    assert allclose([item["mean_minutes"] for item in result], [1, exp(1)])
    assert allclose([item["std_log"] for item in result], [2, 1])
    assert allclose([item["mean_ms"] for item in result], [60_000, exp(1) * 60_000])


def test_mixture_density(simple_gmm: SimpleNamespace) -> None:
    from scipy.stats import norm

    x = array([0.0, 1.0])
    expected = 0.25 * norm.pdf(x, 1, 2) + 0.75 * norm.pdf(x, 0, 1)
    assert allclose(_mixture_density(simple_gmm, x), expected)


def test_gmm_feeding_buzz(monkeypatch: pytest.MonkeyPatch) -> None:
    clustered = DataFrame({
        "Datetime": [
            Timestamp("2024-01-01 12:00:01"),
            Timestamp("2024-01-01 12:00:02"),
            Timestamp("2024-01-01 12:01:01"),
        ],
        "cluster": [0, 0, 1],
    })
    monkeypatch.setattr(
        "disclose.utils.pod.fit_gmm",
        lambda df, comp: (clustered.copy(), array([[0.0]]), None),
    )
    result = gmm_feeding_buzz(DataFrame(), comp=2)
    assert result["Buzz"].tolist() == [2.0, 0.0]
    assert result["fbm_count"].tolist() == [1, 0]
    assert result["start_datetime"].tolist() == [
        Timestamp("2024-01-01 12:00"),
        Timestamp("2024-01-01 12:01"),
    ]


def test_plot_gmm_ici(
    monkeypatch: pytest.MonkeyPatch,
    simple_gmm: SimpleNamespace,
) -> None:
    monkeypatch.setattr(
        "disclose.utils.pod.fit_gmm",
        lambda df, comp: (DataFrame(), array([[0.0], [1.0]]), simple_gmm),
    )
    monkeypatch.setattr(plt, "show", lambda: None)
    fig, ax = plot_gmm_ici(DataFrame(), comp=2)
    try:
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        assert len(ax.lines) == 3  # Two components and the mixture
        assert ax.get_xlabel() == "Log ICI (log minutes)"
    finally:
        plt.close(fig)


def test_calendar(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(plt, "show", lambda: None)
    df = DataFrame({
        "Site": ["A"],
        "Phase": [1],
        "start_deployment": ["2024-01-01"],
        "end_deployment": ["2024-01-05"],
        "start_recording": ["2023-12-31"],
        "end_recording": ["2024-01-06"],
    })
    before = set(plt.get_fignums())
    try:
        assert calendar(df) is None
        assert df.loc[0, "start_recording"] == Timestamp("2024-01-01")
        assert df.loc[0, "end_recording"] == Timestamp("2024-01-05")
        ax = plt.gcf().axes[0]
        assert [label.get_text() for label in ax.get_yticklabels()] == ["A"]
        assert len(ax.collections) == 2  # Deployment and recording
    finally:
        for number in set(plt.get_fignums()) - before:
            plt.close(number)


def test_percent_barplot() -> None:
    df = DataFrame({
        "Month": ["Jan", "Feb"],
        "Site": ["A", "B"],
        "FBR_mean": [10.0, 20.0],
        "FBR_std": [1.0, 2.0],
    })
    fig, ax = plt.subplots()
    try:
        result = percent_barplot(
            df,
            "Month",
            "FBR",
            ax=ax,
            show_std=True,
            shade_missing=False,
            dynamic_ylim=False,
        )
        assert result is ax
        assert len(ax.patches) == 2
        assert allclose([bar.get_height() for bar in ax.patches], [10, 20])
        assert all(bar.get_hatch() == "/" for bar in ax.patches)
        assert ax.get_legend() is not None
        assert ax.get_ylabel() == "FBR"
    finally:
        plt.close(fig)


def test_percent_barplot_empty() -> None:
    fig, ax = plt.subplots()
    try:
        result = percent_barplot(DataFrame({"FBR": []}), "Month", "FBR", ax=ax)
        assert result is ax
        assert not ax.patches
    finally:
        plt.close(fig)


def test_percent_barplot_missing_std(caplog: pytest.LogCaptureFixture) -> None:
    df = DataFrame({"Month": ["Jan"], "Site": ["A"], "FBR_mean": [10.0]})

    ax = percent_barplot(
        df,
        "Month",
        "FBR",
        show_std=True,
        legend=False,
        shade_missing=False,
        dynamic_ylim=False,
    )

    assert ax.get_title() == "FBR per Month"
    assert "show_std=True but column 'FBR_std' not found in df." in caplog.text
    plt.close(ax.figure)


def test_percent_barplot_all_nan(caplog: pytest.LogCaptureFixture) -> None:
    df = DataFrame({"Month": ["Jan"], "Site": ["A"], "FBR_mean": [float("nan")]})

    assert percent_barplot(df, "Month", "FBR") is None
    assert "has no plottable data" in caplog.text


@pytest.mark.parametrize(
    ("ext", "message"),
    [
        ("tsv", "Invalid file extension"),
        ("csv", r"No \.csv files found"),
    ],
)
def test_load_pod_folder_rejects_invalid_or_empty_input(
    tmp_path: Path,
    ext: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        load_pod_folder(tmp_path, ext)
