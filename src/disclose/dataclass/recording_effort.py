"""Recording-effort data and constructors.

``RecordingEffort`` stores recording coverage aggregated into time bins.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pandas import (
    DataFrame,
    IntervalIndex,
    Series,
    Timedelta,
    concat,
    date_range,
    read_csv,
    to_datetime,
)

from disclose.utils.core import round_begin_end_timestamps
from disclose.utils.filtering import find_delimiter

if TYPE_CHECKING:
    from pathlib import Path

    from pandas.tseries.offsets import BaseOffset

    from disclose.dataclass.detection_config import DetectionConfig


@dataclass(frozen=True)
class RecordingEffort:
    """Recording coverage aggregated into bins.

    Attributes
    ----------
    counts : Series
        Recording-effort counts indexed by left-closed time intervals.
    timebin_origin : Timedelta
        Resolution of the underlying recording-coverage timeline.
    """

    counts: Series
    timebin_origin: Timedelta

    @classmethod
    def from_config(
        cls,
        config: DetectionConfig,
        *,
        bin_size: Timedelta | BaseOffset,
    ) -> RecordingEffort:
        """Build binned recording coverage from a configuration's recording file.

        This method reads a CSV with columns:
        - "start_recording"
        - "end_recording"
        - "start_deployment"
        - "end_deployment"

        It intersects recording and deployment periods, builds a fine-grained
        timeline at the configuration's ``timebin_origin`` resolution, and sums
        active intervals into ``bin_size`` bins.

        Parameters
        ----------
        config
            ``DetectionConfig`` with a ``recording_file`` and ``timebin_origin``.
        bin_size : Timedelta or BaseOffset
            Width of the aggregation bins (for example, ``Timedelta("1h")`` or
            a daily offset).

        Returns
        -------
        RecordingEffort
            Object containing `counts` (Series indexed by IntervalIndex) and
            `timebin_origin`.

        """
        # Read CSV and parse datetime columns
        recording_file = config.recording_file

        if not recording_file:
            msg = "No recording file provided."
            raise ValueError(msg)

        if not recording_file.exists():
            msg = f"File not found: {recording_file}"
            raise FileNotFoundError(msg)

        df = cls.from_csv(recording_file)

        # Compute effective recording intervals (intersection)
        df["effective_start_recording"] = df[
            ["start_recording", "start_deployment"]
        ].max(axis=1)

        df["effective_end_recording"] = df[["end_recording", "end_deployment"]].min(
            axis=1,
        )

        # Build fine-grained timeline at `timebin_origin` resolution
        origin = (
            config.timebin_origin
            if isinstance(config.timebin_origin, Timedelta)
            else max(config.timebin_origin)
        )
        time_index = date_range(
            start=df["effective_start_recording"].min(),
            end=df["effective_end_recording"].max(),
            freq=origin,
        )

        starts = df["effective_start_recording"]
        ends = df["effective_end_recording"]

        # +1 at each start, -1 at each end
        events = concat([
            Series(1, index=starts),
            Series(-1, index=ends),
        ]).sort_index()

        # combine events that fall on the exact same timestamp
        events = events.groupby(level=0).sum()

        # running count of "how many intervals are active" at each event time
        coverage = events.cumsum()

        # project onto your regular time_index: carry the last known coverage forward
        effort = coverage.reindex(time_index, method="ffill").fillna(0)
        effort = (effort > 0).astype(int)

        # Aggregate effort into user-defined bin_size
        counts = effort.resample(bin_size, closed="left", label="left").sum()

        counts.index = IntervalIndex.from_arrays(
            counts.index,
            counts.index + round_begin_end_timestamps(list(counts.index), bin_size)[-1],
            closed="left",
        )

        return cls(counts=counts, timebin_origin=origin)

    @classmethod
    def from_csv(
        cls,
        csv_file: Path,
    ) -> DataFrame:
        """Read recording/deployment intervals from CSV and parse their dates.

        The input must contain ``start_recording``, ``end_recording``,
        ``start_deployment``, and ``end_deployment`` columns.
        """
        delim = find_delimiter(csv_file)
        df = read_csv(
            csv_file,
            parse_dates=[
                "start_recording",
                "end_recording",
                "start_deployment",
                "end_deployment",
            ],
            delimiter=delim,
        )

        if df.empty:
            msg = "CSV is empty."
            raise ValueError(msg)

        # Normalize timezones: convert to UTC, then remove tz info (naive)
        for col in df.columns:
            df[col] = to_datetime(df[col], utc=True)

        return df

    @classmethod
    def from_json(
        cls,
        json_file: Path,
    ) -> DataFrame:
        """Extract recording/deployment interval fields from metadata JSON.

        The JSON records are expected to contain deployment and recovery dates
        and at least one channel configuration with recording start/end dates.
        """
        with json_file.open() as f:
            data = json.load(f)

        series_list = [
            Series({
                "start_recording": datum["channel_configurations"][0][
                    "record_start_date"
                ],
                "end_recording": datum["channel_configurations"][0]["record_end_date"],
                "start_deployment": datum["deployment_date"],
                "end_deployment": datum["recovery_date"],
            })
            for datum in data
        ]

        return DataFrame(series_list)
