from disclose.dataclass.data_aplose_config import DataAploseConfig


def test_merge_scalar_and_list_fields() -> None:
    merged = DataAploseConfig.merge(
        [
            DataAploseConfig(annotator="ann1", label=["lbl1", "lbl2"]),
            DataAploseConfig(annotator="ann2", label=["lbl2", "lbl3"]),
        ],
    )

    assert merged.annotator == ["ann1", "ann2"]
    assert merged.label == ["lbl1", "lbl2", "lbl3"]


def test_merge_identical_values_collapse() -> None:
    merged = DataAploseConfig.merge(
        [DataAploseConfig(label="lbl1"), DataAploseConfig(label=["lbl1"])],
    )

    assert merged.label == "lbl1"
