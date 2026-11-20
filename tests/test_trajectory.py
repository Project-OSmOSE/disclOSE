import math

import pytest

from disclose.dataclass.trajectory import Trajectory


def test_empty_trajectory_has_no_position() -> None:
    lat, lon = Trajectory().get_position(0.0)

    assert math.isnan(lat)
    assert math.isnan(lon)


def test_single_position_is_not_enough_to_interpolate() -> None:
    traj = Trajectory()
    traj.add_position(0.0, 10.0, 20.0)

    assert traj.timestamps.tolist() == [0.0]
    assert all(math.isnan(x) for x in traj.get_position(0.0))


def test_linear_interpolation_and_extrapolation() -> None:
    traj = Trajectory()
    traj.add_position(0.0, 10.0, 20.0)
    traj.add_position(10.0, 20.0, 40.0)

    assert traj.get_position(5.0) == pytest.approx((15.0, 30.0))
    assert traj.get_position(20.0) == pytest.approx((30.0, 60.0))


def test_duplicated_timestamp_is_ignored() -> None:
    traj = Trajectory()
    traj.add_position(0.0, 10.0, 20.0)
    traj.add_position(10.0, 20.0, 40.0)
    traj.add_position(10.0, 99.0, 99.0)

    assert traj.timestamps.tolist() == [0.0, 10.0]
    assert traj.latitudes.tolist() == [10.0, 20.0]


def test_identical_consecutive_coordinates_are_ignored() -> None:
    traj = Trajectory()
    traj.add_position(0.0, 10.0, 20.0)
    traj.add_position(5.0, 10.0, 20.0)
    traj.add_position(10.0, 20.0, 40.0)

    assert traj.timestamps.tolist() == [0.0, 10.0]
    assert traj.get_position(5.0) == pytest.approx((15.0, 30.0))
