import pytest

from regression_lens.data import GeneratedSample
from regression_lens.mutations import MutationRecord


def mutation_record() -> MutationRecord:
    return MutationRecord(
        mutation_family="typography",
        mutation_type="font_size",
        target="#hero h2",
        before_value="16px",
        after_value="20px",
    )


def test_regression_sample_serializes_and_normalizes_paths() -> None:
    sample = GeneratedSample(
        sample_id="design2code_0042_font_size_001",
        source_page_id="0042",
        split="train",
        reference_image=r"processed\0042\reference.png",
        candidate_image=r"processed\0042\font_size_001.png",
        is_regression=True,
        mutation=mutation_record(),
        target_bbox=(412, 201, 318, 44),
    )

    serialized = sample.to_dict()

    assert serialized["reference_image"] == "processed/0042/reference.png"
    assert serialized["target_bbox"] == [412, 201, 318, 44]
    assert serialized["mutation"]["mutation_type"] == "font_size"


def test_benign_sample_requires_an_explicit_variation_type() -> None:
    with pytest.raises(ValueError, match="variation_type"):
        GeneratedSample(
            sample_id="design2code_0042_benign_001",
            source_page_id="0042",
            split="train",
            reference_image="processed/0042/reference.png",
            candidate_image="processed/0042/benign.png",
            is_regression=False,
        )


def test_invalid_label_value_is_rejected() -> None:
    with pytest.raises(TypeError, match="boolean"):
        GeneratedSample(
            sample_id="bad_label",
            source_page_id="0042",
            split="train",
            reference_image="processed/0042/reference.png",
            candidate_image="processed/0042/candidate.png",
            is_regression="regression",  # type: ignore[arg-type]
            mutation=mutation_record(),
        )


@pytest.mark.parametrize("split", ["dev", "external", ""])
def test_unknown_internal_split_is_rejected(split: str) -> None:
    with pytest.raises(ValueError, match="split"):
        GeneratedSample(
            sample_id="bad_split",
            source_page_id="0042",
            split=split,
            reference_image="processed/0042/reference.png",
            candidate_image="processed/0042/candidate.png",
            is_regression=True,
            mutation=mutation_record(),
        )


@pytest.mark.parametrize(
    "path",
    [r"C:\data\reference.png", "/data/reference.png", "../reference.png"],
)
def test_nonportable_artifact_paths_are_rejected(path: str) -> None:
    with pytest.raises(ValueError, match="portable relative path"):
        GeneratedSample(
            sample_id="bad_path",
            source_page_id="0042",
            split="train",
            reference_image=path,
            candidate_image="processed/0042/candidate.png",
            is_regression=True,
            mutation=mutation_record(),
        )
