import numpy as np

from thoracic_ai.metrics import calculate_metrics, tune_thresholds


def test_metrics_are_perfect_for_separable_predictions() -> None:
    targets = np.array([[0, 1], [1, 0], [1, 1], [0, 0]], dtype=np.float32)
    probabilities = np.array(
        [[0.1, 0.9], [0.9, 0.1], [0.8, 0.8], [0.2, 0.2]], dtype=np.float32
    )
    classes = ["Disease A", "Disease B"]

    thresholds = tune_thresholds(targets, probabilities)
    summary, class_metrics = calculate_metrics(targets, probabilities, classes, thresholds)

    assert summary["macro_auroc"] == 1.0
    assert summary["macro_auprc"] == 1.0
    assert summary["macro_f1"] == 1.0
    assert list(class_metrics["class"]) == classes


def test_metrics_validate_shapes() -> None:
    targets = np.zeros((4, 2), dtype=np.float32)
    probabilities = np.zeros((4, 3), dtype=np.float32)

    try:
        calculate_metrics(targets, probabilities, ["A", "B"])
    except ValueError as error:
        assert "identical shapes" in str(error)
    else:
        raise AssertionError("Expected calculate_metrics to reject mismatched shapes")

