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
