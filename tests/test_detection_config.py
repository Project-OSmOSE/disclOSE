from pandas import Timestamp

from disclose.dataclass.detection_config import DetectionConfig


def test_merge_scalar_and_list_fields() -> None:
    merged = DetectionConfig.merge(
        [
            DetectionConfig(annotator="ann1", label=["lbl1", "lbl2"]),
            DetectionConfig(annotator="ann2", label=["lbl2", "lbl3"]),
        ],
    )

    assert merged.annotator == ["ann1", "ann2"]
    assert merged.label == ["lbl1", "lbl2", "lbl3"]


def test_merge_identical_values_collapse() -> None:
    merged = DetectionConfig.merge(
        [DetectionConfig(label="lbl1"), DetectionConfig(label=["lbl1"])],
    )

    assert merged.label == "lbl1"


def test_merge_empty_configs() -> None:
    assert DetectionConfig.merge([]) == DetectionConfig.empty()


def test_merge_datetime_ranges() -> None:
    merged = DetectionConfig.merge(
        [
            DetectionConfig(
                start_datetime=Timestamp("2025-01-02"),
                end_datetime=Timestamp("2025-01-05"),
            ),
            DetectionConfig(
                start_datetime=Timestamp("2025-01-01"),
                end_datetime=Timestamp("2025-01-06"),
            ),
        ],
    )

    assert merged.start_datetime == Timestamp("2025-01-01")
    assert merged.end_datetime == Timestamp("2025-01-06")
