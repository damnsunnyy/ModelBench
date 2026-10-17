"""Metrics selected for honest, readable holdout comparisons."""
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score


def evaluate(task, expected, predicted):
    if task == "classification":
        return {
            "accuracy": float(accuracy_score(expected, predicted)),
            "balanced_accuracy": float(balanced_accuracy_score(expected, predicted)),
            "f1_macro": float(f1_score(expected, predicted, average="macro", zero_division=0)),
        }
    return {
        "mae": float(mean_absolute_error(expected, predicted)),
        "rmse": float(mean_squared_error(expected, predicted) ** 0.5),
        "r2": float(r2_score(expected, predicted)),
    }


def primary_metric(task):
    return "f1_macro" if task == "classification" else "r2"
