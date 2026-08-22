import pytest

from thoracic_ai.models import create_model, gradcam_target_layer


def test_gradcam_uses_final_densenet_convolution() -> None:
    model = create_model("densenet121", num_classes=9, pretrained=False)

    target = gradcam_target_layer(model, "densenet121")

    assert target is model.features.denseblock4.denselayer16.conv2


def test_gradcam_rejects_unsupported_architecture() -> None:
    model = create_model("densenet121", num_classes=9, pretrained=False)

    with pytest.raises(ValueError, match="DenseNet-121"):
        gradcam_target_layer(model, "vit_b16")
