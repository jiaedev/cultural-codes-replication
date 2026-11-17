"""Binary Macro-F1 with an explicit two-class universe."""
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

def from_predictions(references, predictions):
    references = np.asarray(references)
    predictions = np.asarray(predictions)
    if references.ndim != 1 or predictions.shape != references.shape or not len(references):
        raise ValueError("Expected nonempty aligned one-dimensional labels")
    if not np.isin(references, [0, 1]).all() or not np.isin(predictions, [0, 1]).all():
        raise ValueError("Metrics accept only class IDs 0 and 1")
    return {"macro_f1": float(f1_score(references, predictions, labels=[0, 1],
                                    average="macro", zero_division=0)),
            "accuracy": float(accuracy_score(references, predictions))}

def compute_metrics(eval_prediction):
    logits, references = eval_prediction
    if isinstance(logits, tuple):
        logits = logits[0]
    logits = np.asarray(logits)
    if logits.ndim != 2 or logits.shape[1] != 2 or not np.isfinite(logits).all():
        raise ValueError("Expected finite (N, 2) logits")
    return from_predictions(references, np.argmax(logits, axis=1))
