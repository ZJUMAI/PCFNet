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


@pytest.mark.parametrize("stages", ["layer4", ["layer5"], ["layer4", "layer4"], [1], False])
def test_invalid_backbone_stages_fail_before_weight_download(monkeypatch, stages):
    monkeypatch.setattr(
        "petct.models.r3d_18", lambda **kwargs: pytest.fail("Backbone must not be constructed")
    )
    with pytest.raises(ConfigError, match="backbone_trainable_stages"):
        VolumeBackbone(trainable_stages=stages)


def test_backbone_default_keeps_full_fine_tuning():
    backbone = VolumeBackbone(pretrained=False).train()
    assert all(parameter.requires_grad for parameter in backbone.parameters())
    assert all(getattr(backbone.network, stage).training for stage in backbone.stages)


def test_partial_backbone_freezing_keeps_frozen_batchnorm_in_eval():
    backbone = VolumeBackbone(pretrained=False, trainable_stages=["layer4"])
    for _ in range(2):
        backbone.eval()
        backbone.train()
        for stage in backbone.stages:
            module = getattr(backbone.network, stage)
            assert module.training == (stage == "layer4")
            assert all(p.requires_grad == (stage == "layer4") for p in module.parameters())
    frozen_bn = backbone.network.stem[1]
    before_mean = frozen_bn.running_mean.clone()
    before_count = frozen_bn.num_batches_tracked.clone()
    output = backbone(torch.randn(2, 8, 16, 16))
    output.square().mean().backward()
    assert backbone.network.stem[0].weight.grad is None
    assert backbone.network.layer4[0].conv1[0].weight.grad is not None
    torch.testing.assert_close(frozen_bn.running_mean, before_mean, rtol=0, atol=0)
    assert torch.equal(frozen_bn.num_batches_tracked, before_count)


@pytest.mark.parametrize("name", ["fusion3d", "fusion3d_spatial"])
def test_frozen_streams_and_smaller_classifier_complete_training_step(name):
    model = build_model(
        {
            "model": {
                "name": name,
                "params": {
                    "pretrained": False,
                    "backbone_trainable_stages": [],
                    "classifier_hidden_dim": 32,
                    "dropout": 0.6,
                    "attention_dropout": 0.2,
                },
            }
        }
    ).train()
    assert model.classifier[0].out_features == 32
    before = model.classifier[0].weight.detach().clone()
    frozen_before = model.ct_backbone.network.stem[0].weight.detach().clone()
    bn_before = model.ct_backbone.network.stem[1].running_mean.clone()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-3)
    logits = model(torch.randn(2, 8, 16, 16), torch.randn(2, 8, 16, 16))
    loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1]))
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    assert tuple(logits.shape) == (2, 2)
    assert torch.isfinite(loss)
    for stream in (model.ct_backbone, model.pet_backbone):
        assert all(not p.requires_grad and p.grad is None for p in stream.parameters())
        assert not stream.network.stem[1].training
    assert not torch.equal(before, model.classifier[0].weight)
    torch.testing.assert_close(model.ct_backbone.network.stem[0].weight, frozen_before)
    torch.testing.assert_close(model.ct_backbone.network.stem[1].running_mean, bn_before)


@pytest.mark.parametrize("name", ["fusion3d", "fusion3d_spatial"])
@pytest.mark.parametrize("hidden_dim", [0, -1, True, 12.5])
def test_invalid_classifier_width_fails_before_weight_download(monkeypatch, name, hidden_dim):
    monkeypatch.setattr(
        "petct.models.r3d_18", lambda **kwargs: pytest.fail("Backbone must not be constructed")
    )
    with pytest.raises(ConfigError, match="classifier_hidden_dim"):
        build_model({"model": {"name": name, "params": {"classifier_hidden_dim": hidden_dim}}})
