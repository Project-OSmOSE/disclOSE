"""Functions to manipulate Sylence data."""

from osekit.utils.timestamp import strptime_from_text
from pandas import DataFrame, Series, Timedelta


def clean_pamguard_false_detection(
    df: DataFrame,
    max_time: Timedelta = Timedelta(seconds=1),
) -> DataFrame:
    """Clean PAMGuard false detections.

    The first detection on each audio file corresponds to the detection of
    an electronic buzz made by the recorder (might be very specific to Sylence data).
    This function identifies and removes these false detections by checking if
    a detection occurs within the first ``max_time`` of the corresponding audio file.

    Parameters
    ----------
    df: DataFrame
        APLOSE DataFrame (presumably from PAMGuard)
    max_time: Timedelta
        Detections starting less than ``max_time`` after the beginning of their
        audio file are removed. Defaults to one second.

    Returns
    -------
    A cleaned DataFrame with false detections removed

    """
    if df.empty:
        return df

    tz_data = df["start_datetime"].iloc[0].tz
    file_begin: Series = df["filename"].map(
        lambda fn: strptime_from_text(fn, "%Y-%m-%d_%H-%M-%S").tz_localize(tz_data),
    )

    delay = df["start_datetime"] - file_begin
    return df[delay >= max_time]
