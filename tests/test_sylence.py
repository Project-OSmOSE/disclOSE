from pandas import DataFrame, Timestamp

from disclose.utils.sylence import clean_pamguard_false_detection


def _df(index: list[int]) -> DataFrame:
    return DataFrame(
        {
            "filename": [
                "2025-01-01_00-00-00",
                "2025-01-01_00-00-00",
                "2025-01-01_00-10-00",
            ],
            "start_datetime": [
                Timestamp("2025-01-01T00:00:00.5+00:00"),
                Timestamp("2025-01-01T00:00:05+00:00"),
                Timestamp("2025-01-01T00:10:00.2+00:00"),
            ],
        },
        index=index,
    )


def test_clean_pamguard_false_detection_drops_first_second() -> None:
    result = clean_pamguard_false_detection(_df([0, 1, 2]))

    assert result["filename"].tolist() == ["2025-01-01_00-00-00"]


def test_clean_pamguard_false_detection_non_default_index() -> None:
    result = clean_pamguard_false_detection(_df([10, 11, 12]))

    assert result.index.tolist() == [11]


def test_clean_pamguard_false_detection_empty_dataframe() -> None:
    df = DataFrame(columns=["filename", "start_datetime"])

    result = clean_pamguard_false_detection(df)

    assert result.empty
    assert result.equals(df)
