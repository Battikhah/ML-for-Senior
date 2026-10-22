"""Evaluate probabilities with explicit selection and robustness boundaries."""
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             average_precision_score, roc_auc_score, confusion_matrix)


def evaluate_probabilities(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict:
    """Retain class-specific outcomes instead of hiding missed infestations in accuracy."""
    labels = np.asarray(labels).ravel(); p = np.asarray(probabilities).ravel()
    if len(labels) == 0 or len(labels) != len(p) or not np.isfinite(p).all():
        raise ValueError('Empty, nonfinite or mismatched predictions')
    if not set(labels.tolist()) <= {0, 1} or np.any((p < 0) | (p > 1)) or not 0 <= threshold <= 1:
        raise ValueError('Invalid binary labels, probabilities or threshold')
    predicted = p >= threshold
    both = len(set(labels.tolist())) == 2
    return dict(accuracy=float(accuracy_score(labels, predicted)),
                precision=float(precision_score(labels, predicted, zero_division=0)),
                recall=float(recall_score(labels, predicted, zero_division=0)),
                f1=float(f1_score(labels, predicted, zero_division=0)),
                pr_auc=float(average_precision_score(labels, p)) if both else None,
                roc_auc=float(roc_auc_score(labels, p)) if both else None,
                confusion=confusion_matrix(labels, predicted, labels=[0, 1]).tolist(),
                threshold=float(threshold), count=len(labels))


def select_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    """Tune on validation predictions, with stable preference for thresholds near 0.5."""
    from sklearn.metrics import precision_recall_curve
    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    candidates = np.flatnonzero(np.isclose(f1, f1.max(), rtol=0, atol=1e-12))
    return float(min(thresholds[candidates], key=lambda t: (abs(t - .5), -t)))


def robustness_inputs(x: np.ndarray, seed: int) -> dict:
    """Declare feature perturbations explicitly; these are not acoustic field validation."""
    rng = np.random.default_rng(seed)
    scale = np.maximum(x.std(axis=(0, 1), keepdims=True), 1e-6)
    result = {'feature_noise_5pct': x + rng.normal(size=x.shape) * scale * .05,
              'coefficient_scale_0.8': x * .8, 'coefficient_scale_1.2': x * 1.2}
    for bits in [8, 12]:
        lo, hi = float(x.min()), float(x.max())
        step = max((hi - lo) / (2 ** bits - 1), 1e-12)
        result[f'transmission_{bits}bit'] = (np.rint((x - lo) / step) * step + lo).astype('float32')
    return result
