import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from petct.models import build_model


@pytest.mark.parametrize(
    ("name", "params", "shape"),
    [
        ("fusion3d", {"pretrained": False, "num_heads": 8}, (1, 16, 32, 32)),
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
