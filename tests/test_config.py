from pathlib import Path

from thoracic_ai.config import ExperimentConfig


def test_densenet_config_loads() -> None:
    config_path = Path(__file__).parents[1] / "configs" / "densenet121.yaml"
    config = ExperimentConfig.from_yaml(config_path)

    assert config.model.name == "densenet121"
    assert len(config.data.classes) == 9
    assert config.data.image_size == 224
    assert config.training.epochs >= 1

