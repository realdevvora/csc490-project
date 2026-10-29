import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from PIL import Image  # noqa: E402
from transformers import CLIPConfig, CLIPModel, SiglipConfig, SiglipModel  # noqa: E402

from regression_lens.data.feature_cache import FeatureCache, build_feature_cache  # noqa: E402
from regression_lens.data.pairs import PairDataset  # noqa: E402
from regression_lens.models.uiclip import SiameseDetector  # noqa: E402

TEXT = dict(hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2)
VISION = dict(hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4, image_size=224)
BACKBONES = {
    "clip": (lambda: CLIPModel(CLIPConfig(vision_config={**VISION, "patch_size": 32}, text_config=TEXT)), 32),
    "siglip": (lambda: SiglipModel(SiglipConfig(vision_config={**VISION, "patch_size": 16}, text_config=TEXT)), 16),
}


@pytest.fixture(params=sorted(BACKBONES))
def backbone(request, tmp_path):
    build, patch = BACKBONES[request.param]
    torch.manual_seed(0)
    path = tmp_path / request.param
    build().save_pretrained(path)
    return str(path), patch


@pytest.fixture
def dataset_parts(tmp_path):
    """Two pages; the first pair's candidate is taller than its reference."""

    generator = torch.Generator().manual_seed(0)
    sizes = {"p0.png": (1280, 300), "p0_m.png": (1280, 520), "p1.png": (1280, 230), "p1_b.png": (1280, 230)}
    (tmp_path / "data").mkdir()
    for name, (w, h) in sizes.items():
        pixels = torch.randint(0, 256, (h, w, 3), generator=generator, dtype=torch.uint8)
        Image.fromarray(pixels.numpy()).save(tmp_path / "data" / name)
    records = [
        {"sample_id": "p0_m", "parent_id": "p0", "label": "regression", "operator": "margin",
         "reference_path": "p0.png", "candidate_path": "p0_m.png",
         "size_ref": list(sizes["p0.png"]), "size_cand": list(sizes["p0_m.png"]), "diff_bbox": [10, 200, 300, 250]},
        {"sample_id": "p1_b", "parent_id": "p1", "label": "benign", "operator": None,
         "reference_path": "p1.png", "candidate_path": "p1_b.png",
         "size_ref": list(sizes["p1.png"]), "size_cand": list(sizes["p1_b.png"]), "diff_bbox": None},
    ]
    return records, tmp_path / "data", tmp_path / "cache"


def test_split_encoding_matches_huggingface_forward(backbone) -> None:
    checkpoint, _ = backbone
    model = SiameseDetector(checkpoint, n_operators=2).eval()
    tiles = torch.randn(3, 3, 224, 224)

    expected = model.encoder(pixel_values=tiles).last_hidden_state
    if model.has_cls:
        expected = expected[:, 1:]
    torch.testing.assert_close(model.encode(tiles), expected)


@pytest.mark.parametrize("train_last_n", [0, 1])
def test_cached_features_reproduce_uncached_outputs(backbone, dataset_parts, train_last_n) -> None:
    checkpoint, patch = backbone
    records, data_root, cache_root = dataset_parts
    model = SiameseDetector(checkpoint, n_operators=2, train_last_n=train_last_n).eval()
    kwargs = dict(records=records, data_root=data_root, pages=["p0", "p1"],
                  operators=["background_color", "margin"], patch=patch)
    plain, cached = PairDataset(**kwargs), PairDataset(**kwargs)

    cache = FeatureCache.open(cache_root, checkpoint, model.cache_layer, 224, patch,
                              plain.cols, tuple(plain.mean.flatten().tolist()), tuple(plain.std.flatten().tolist()))
    build_feature_cache(cache, model, plain.images(), data_root, torch.device("cpu"), workers=0)
    assert cache.missing(plain.images()) == []
    cached.feature_cache = cache

    for index in range(len(plain)):
        a, b = plain[index], cached[index]
        assert b["cached"] and not a["cached"]
        assert a["grid"] == b["grid"] and torch.equal(a["patch_target"], b["patch_target"])
        with torch.no_grad():
            for x, y in zip(model(a["ref"], a["cand"]), model(b["ref"], b["cand"], cached=True)):
                torch.testing.assert_close(x, y, rtol=1e-4, atol=1e-5)


def test_cache_rejects_different_settings(tmp_path) -> None:
    FeatureCache.open(tmp_path, "some/model", 12, 224, 32, 6, (0.5,) * 3, (0.5,) * 3)

    with pytest.raises(ValueError, match="different settings"):
        FeatureCache.open(tmp_path, "some/model", 12, 224, 32, 6, (0.4,) * 3, (0.5,) * 3)


def test_self_calibration_scores_identical_pairs_alike(backbone) -> None:
    checkpoint, _ = backbone
    model = SiameseDetector(checkpoint, n_operators=2).eval()
    with torch.no_grad():
        model.patch_offset.fill_(-2.0)
        model.image_offset.fill_(-3.0)
        page_a, page_b = torch.randn(4, 3, 224, 224), torch.randn(6, 3, 224, 224)

        for page in (page_a, page_b):
            patch_logits, image_logit, _ = model(page, page.clone())
            assert torch.equal(patch_logits, torch.full_like(patch_logits, -2.0))
            assert image_logit.item() == -3.0
        # A real difference moves the score away from the offset.
        _, changed, _ = model(page_a, page_a + 0.5 * torch.randn_like(page_a))
        assert changed.item() != -3.0


def test_without_self_calibration_identical_pairs_differ_by_page(backbone) -> None:
    checkpoint, _ = backbone
    model = SiameseDetector(checkpoint, n_operators=2, self_calibrate=False).eval()
    with torch.no_grad():
        scores = [model(p, p.clone())[1].item() for p in (torch.randn(4, 3, 224, 224), torch.randn(6, 3, 224, 224))]
    assert scores[0] != scores[1]
