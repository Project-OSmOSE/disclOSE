from datetime import datetime

from disclose.instrument.datetime_format import (
    DORI_DATETIME_FORMAT,
    SOUNDTRAP_DATETIME_FORMAT,
    SYLENCE_DATETIME_FORMAT,
)
from disclose.instrument.glider import NAV_STATE, SEA034_SENSITIVITY
from disclose.instrument.soundtrap import RECORDER_GAIN


def test_datetime_formats_parse_file_names() -> None:
    expected = datetime(2025, 3, 4, 5, 6, 7)

    assert datetime.strptime("250304050607", SOUNDTRAP_DATETIME_FORMAT) == expected
    assert datetime.strptime("2025-03-04_05-06-07", SYLENCE_DATETIME_FORMAT) == expected
    assert datetime.strptime("2025y03m04d_05h06m07s", DORI_DATETIME_FORMAT) == expected


def test_recorder_gain_maps_serial_numbers_to_negative_db() -> None:
    assert RECORDER_GAIN
    assert all(isinstance(serial, str) and serial.isdigit() for serial in RECORDER_GAIN)
    assert all(isinstance(gain, float) and gain < 0 for gain in RECORDER_GAIN.values())


def test_glider_constants() -> None:
    assert NAV_STATE[100] == "Going down"
    assert all(isinstance(state, str) for state in NAV_STATE.values())
    assert SEA034_SENSITIVITY == -165
