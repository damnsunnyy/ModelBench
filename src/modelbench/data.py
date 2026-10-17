"""Input validation and train-only preprocessing pipelines."""
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class DatasetError(ValueError):
    """Raised when a table cannot support the requested benchmark."""


def load_dataset(path: Path, target: str):
    if not path.is_file():
        raise DatasetError(f"Dataset file does not exist: {path}")
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        raise DatasetError(f"Could not read CSV: {exc}") from exc
    if frame.empty:
        raise DatasetError("The CSV has a header but no data rows.")
    if target not in frame.columns:
        raise DatasetError(f"Target column {target!r} is missing. Available columns: {', '.join(map(str, frame.columns))}")
    if frame.columns.duplicated().any():
        duplicates = frame.columns[frame.columns.duplicated()].tolist()
        raise DatasetError(f"Duplicate column names are not supported: {', '.join(duplicates)}")
    missing_targets = int(frame[target].isna().sum())
    frame = frame.dropna(subset=[target]).copy()
    if len(frame) < 12:
        raise DatasetError("At least 12 labeled rows are required for a holdout benchmark.")
    y = frame.pop(target)
    y.attrs["missing_removed"] = missing_targets
    if y.nunique(dropna=True) < 2:
        raise DatasetError("The target column must contain at least two distinct values.")
    if frame.shape[1] == 0:
        raise DatasetError("No feature columns remain after removing the target.")
    return frame, y


def make_preprocessor(features: pd.DataFrame):
    numeric = features.select_dtypes(include=["number"]).columns.tolist()
    categorical = [column for column in features.columns if column not in numeric]
    transformers = []
    if numeric:
        numeric_pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
        transformers.append(("numeric", numeric_pipe, numeric))
    if categorical:
        categorical_pipe = Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("encode", OneHotEncoder(handle_unknown="ignore"))])
        transformers.append(("categorical", categorical_pipe, categorical))
    return ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=False)


def dataset_profile(features: pd.DataFrame, target: pd.Series):
    return {
        "rows": int(len(features)),
        "features": int(features.shape[1]),
        "numeric_features": int(len(features.select_dtypes(include=["number"]).columns)),
        "categorical_features": int(features.shape[1] - len(features.select_dtypes(include=["number"]).columns)),
        "missing_feature_cells": int(features.isna().sum().sum()),
        "target_missing_rows_removed": int(target.attrs.get("missing_removed", 0)),
        "target_values": int(target.nunique()),
    }
