import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from petct.config import ConfigError
from petct.models import (
    PETCTSpatialFusionClassifier,
    SpatialBidirectionalCrossAttention,
    SpatialTokenEncoder,
    VolumeBackbone,
    build_model,
)


@pytest.fixture(autouse=True)
def fixed_seed():
    torch.manual_seed(42)


@pytest.mark.parametrize(
    ("name", "params", "shape"),
    [
        ("fusion3d", {"pretrained": False, "num_heads": 8}, (1, 16, 32, 32)),
        (
            "fusion3d_spatial",
            {"pretrained": False, "embedding_dim": 128, "num_heads": 4},
            (1, 16, 32, 32),
        ),
        ("densenet", {"pretrained": False, "depth": 64}, (1, 64, 64, 64)),
        (
            "feature_extractor",
            {"pretrained": False, "depth": 64, "backbone": "resnet"},
            (1, 64, 64, 64),
        ),
    ],
)
def test_registered_model_forward_and_gradient(name: str, params: dict, shape: tuple) -> None:
    model = build_model({"model": {"name": name, "params": params}})
    model.eval()
    ct = torch.randn(shape, requires_grad=True)
    pet = torch.randn(shape, requires_grad=True)
    output = model(ct, pet)
    assert tuple(output.shape) == (1, 2)
    output.sum().backward()
    assert ct.grad is not None
    assert pet.grad is not None


def test_spatial_token_encoder_returns_64_low_dimensional_tokens():
    encoder = SpatialTokenEncoder(feature_dim=16, embedding_dim=128)
    features = torch.randn(2, 16, 8, 4, 4, requires_grad=True)
    tokens = encoder(features)
    assert tuple(tokens.shape) == (2, 64, 128)
    assert encoder.num_tokens == 64
    tokens.square().mean().backward()
    assert torch.isfinite(features.grad).all()
    assert features.grad.abs().sum() > 0


def test_spatial_backbone_preserves_original_pooled_features():
    backbone = VolumeBackbone(pretrained=False).eval()
    volume = torch.randn(1, 16, 32, 32)
    with torch.no_grad():
        features = backbone.forward_features(volume)
        original = backbone(volume)
        explicit_pool = backbone.network.avgpool(features).flatten(1)
    assert tuple(features.shape) == (1, 512, 2, 2, 2)
    torch.testing.assert_close(explicit_pool, original)


def test_zero_attention_preserves_both_residual_inputs():
    fusion = SpatialBidirectionalCrossAttention(dropout=0.0).eval()
    with torch.no_grad():
        for attention in (fusion.ct_to_pet, fusion.pet_to_ct):
            attention.out_proj.weight.zero_()
            attention.out_proj.bias.zero_()
    ct, pet = torch.randn(2, 64, 128), torch.randn(2, 64, 128)
    ct_updated, pet_updated = fusion(ct, pet)
    torch.testing.assert_close(ct_updated, ct, rtol=0, atol=0)
    torch.testing.assert_close(pet_updated, pet, rtol=0, atol=0)


def test_cross_attention_delta_depends_on_queries_with_multiple_tokens():
    fusion = SpatialBidirectionalCrossAttention(dropout=0.0).eval()
    ct, pet = torch.randn(2, 64, 128), torch.randn(2, 64, 128)
    with torch.no_grad():
        first, _ = fusion(ct, pet)
        second, _ = fusion(-ct, pet)
    # Subtract the direct residual to isolate the CT-to-PET attention output.
    assert not torch.allclose(first - ct, second + ct)


def test_both_attention_directions_receive_query_key_value_gradients():
    fusion = SpatialBidirectionalCrossAttention(dropout=0.0)
    ct = torch.randn(2, 64, 128, requires_grad=True)
    pet = torch.randn(2, 64, 128, requires_grad=True)
    ct_updated, pet_updated = fusion(ct, pet)
    (ct_updated.square().mean() + pet_updated.square().mean()).backward()
    for attention in (fusion.ct_to_pet, fusion.pet_to_ct):
        gradients = attention.in_proj_weight.grad
        for gradient in gradients.chunk(3, dim=0):
            assert torch.isfinite(gradient).all()
            assert gradient.abs().sum() > 0


def test_spatial_model_uses_64_tokens_for_full_64_cube_inputs():
    model = PETCTSpatialFusionClassifier(pretrained=False).eval()
    observed = {}

    def capture(name):
        def hook(module, inputs, output):
            observed[name] = tuple(output.shape)

        return hook

    handles = [
        model.ct_tokens.register_forward_hook(capture("ct")),
        model.pet_tokens.register_forward_hook(capture("pet")),
    ]
    try:
        with torch.no_grad():
            output = model(torch.randn(1, 64, 64, 64), torch.randn(1, 64, 64, 64))
        assert tuple(output.shape) == (1, 2)
        assert observed == {"ct": (1, 64, 128), "pet": (1, 64, 128)}
        assert torch.isfinite(output).all()
    finally:
        for handle in handles:
            handle.remove()


def test_spatial_model_completes_a_synthetic_training_step():
    model = PETCTSpatialFusionClassifier(pretrained=False).train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    before = model.ct_tokens.projection.weight.detach().clone()
    ct, pet = torch.randn(2, 16, 32, 32), torch.randn(2, 16, 32, 32)
    logits = model(ct, pet)
    loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1]))
    assert torch.isfinite(loss)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    for encoder in (model.ct_tokens, model.pet_tokens):
        gradient = encoder.projection.weight.grad
        assert torch.isfinite(gradient).all()
        assert gradient.abs().sum() > 0
    optimizer.step()
    assert not torch.equal(before, model.ct_tokens.projection.weight)


@pytest.mark.parametrize("embedding_dim, num_heads", [(0, 4), (128, 0), (130, 4), (128.5, 4)])
def test_invalid_spatial_attention_dimensions_fail_before_backbone_construction(
    monkeypatch, embedding_dim, num_heads
):
    monkeypatch.setattr(
        "petct.models.r3d_18", lambda **kwargs: pytest.fail("Backbone must not be constructed")
    )
    with pytest.raises(ConfigError, match="embedding_dim"):
        PETCTSpatialFusionClassifier(embedding_dim=embedding_dim, num_heads=num_heads)
