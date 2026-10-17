"""Leakage-aware holdout benchmark orchestration."""
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, cross_val_score

from .data import DatasetError, dataset_profile, load_dataset
from .metrics import evaluate, primary_metric
from .models import candidate_estimators, make_pipeline


def run_benchmark(path, target, task="classification", test_size=0.2, seed=42, cv=0):
    features, labels = load_dataset(path, target)
    if task == "regression":
        missing_target_rows = labels.attrs.get("missing_removed", 0)
        try:
            labels = labels.astype(float)
            labels.attrs["missing_removed"] = missing_target_rows
        except (TypeError, ValueError) as exc:
            raise DatasetError("Regression target must contain numeric values.") from exc
    elif task != "classification":
        raise DatasetError("Task must be classification or regression.")
    classes, counts = np.unique(labels, return_counts=True)
    test_rows = int(np.ceil(len(features) * test_size))
    stratify = labels if task == "classification" and len(classes) > 1 and counts.min() >= 2 and test_rows >= len(classes) and len(features) - test_rows >= len(classes) else None
    if task == "classification" and len(classes) < 2:
        raise DatasetError("Classification needs at least two target classes.")
    train_x, test_x, train_y, test_y = train_test_split(features, labels, test_size=test_size, random_state=seed, stratify=stratify)
    results = []
    fitted = {}
    metric = primary_metric(task)
    for name, estimator in candidate_estimators(task, seed):
        pipeline = make_pipeline(train_x, estimator)
        started = time.perf_counter()
        try:
            pipeline.fit(train_x, train_y)
            fit_seconds = time.perf_counter() - started
            started = time.perf_counter()
            predictions = pipeline.predict(test_x)
            predict_seconds = time.perf_counter() - started
            metrics = evaluate(task, test_y, predictions)
            row = {"model": name, "metrics": metrics, "fit_seconds": fit_seconds, "predict_seconds": predict_seconds, "error": None, "cv_mean": None, "cv_std": None, "cv_error": None}
            if cv:
                try:
                    training_counts = np.unique(train_y, return_counts=True)[1]
                    fold_count = cv
                    if task == "classification" and int(training_counts.min()) < cv:
                        raise DatasetError(f"Cross-validation with {cv} folds needs at least {cv} training rows in every class.")
                    if fold_count < 2:
                        raise DatasetError("Each classification label needs at least two training examples for cross-validation.")
                    splitter = StratifiedKFold(n_splits=fold_count, shuffle=True, random_state=seed) if task == "classification" else KFold(n_splits=fold_count, shuffle=True, random_state=seed)
                    fold_scores = cross_val_score(make_pipeline(train_x, estimator), train_x, train_y, cv=splitter, scoring=metric if task == "classification" else "r2", n_jobs=1)
                    row["cv_mean"] = float(fold_scores.mean())
                    row["cv_std"] = float(fold_scores.std())
                except Exception as exc:
                    row["cv_error"] = f"{type(exc).__name__}: {exc}"
            results.append(row)
            fitted[name] = pipeline
        except Exception as exc:
            results.append({"model": name, "metrics": {}, "fit_seconds": None, "predict_seconds": None, "error": f"{type(exc).__name__}: {exc}", "cv_mean": None, "cv_std": None, "cv_error": None})
    valid = [row for row in results if metric in row["metrics"] and np.isfinite(row["metrics"][metric])]
    if not valid:
        raise DatasetError("Every model failed. Check the target values and feature columns.")
    valid.sort(key=lambda row: row["metrics"][metric], reverse=True)
    ranked = {row["model"]: index + 1 for index, row in enumerate(valid)}
    for row in results:
        row["rank"] = ranked.get(row["model"])
    results.sort(key=lambda row: (row["rank"] is None, row["rank"] or 999))
    winning = fitted[valid[0]["model"]]
    importance = []
    estimator = winning.named_steps["model"]
    raw_importance = getattr(estimator, "feature_importances_", None)
    if raw_importance is None and hasattr(estimator, "coef_"):
        raw_importance = np.abs(estimator.coef_).mean(axis=0) if estimator.coef_.ndim > 1 else np.abs(estimator.coef_)
    if raw_importance is not None:
        try:
            names = winning.named_steps["preprocess"].get_feature_names_out()
            total = float(np.sum(raw_importance)) or 1.0
            importance = [{"feature": str(name), "weight": float(value / total)} for name, value in zip(names, raw_importance)]
            importance.sort(key=lambda item: item["weight"], reverse=True)
            importance = importance[:12]
        except (AttributeError, ValueError):
            importance = []
    confusion = None
    if task == "classification":
        predicted = winning.predict(test_x)
        labels_order = np.unique(labels)
        confusion = {"labels": [str(value) for value in labels_order], "matrix": confusion_matrix(test_y, predicted, labels=labels_order).tolist()}
    return {
        "dataset": Path(path).name,
        "target": target,
        "task": task,
        "seed": seed,
        "test_fraction": test_size,
        "train_rows": int(len(train_x)),
        "test_rows": int(len(test_x)),
        "cv_folds": cv,
        "primary_metric": metric,
        "best_model": valid[0]["model"],
        "feature_importance": importance,
        "confusion_matrix": confusion,
        "profile": dataset_profile(features, labels),
        "results": results,
    }
