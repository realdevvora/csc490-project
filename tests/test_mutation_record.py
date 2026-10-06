from regression_lens.mutations import MutationRecord


def test_mutation_record_serializes_provenance() -> None:
    record = MutationRecord(
        mutation_family="typography",
        mutation_type="font_size",
        target="#hero h2",
        before_value="16px",
        after_value="20px",
        parameters={"delta_px": 4},
    )

    assert record.to_dict() == {
        "mutation_family": "typography",
        "mutation_type": "font_size",
        "target": "#hero h2",
        "before_value": "16px",
        "after_value": "20px",
        "parameters": {"delta_px": 4},
    }
