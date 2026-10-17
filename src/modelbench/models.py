"""Small, seeded model sets used by the benchmark engine."""
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline

from .data import make_preprocessor


def candidate_estimators(task: str, seed: int):
    if task == "classification":
        return [
            ("majority baseline", DummyClassifier(strategy="most_frequent")),
            ("logistic regression", LogisticRegression(max_iter=1500, random_state=seed)),
            ("random forest", RandomForestClassifier(n_estimators=240, min_samples_leaf=2, class_weight="balanced", random_state=seed, n_jobs=-1)),
            ("extra trees", ExtraTreesClassifier(n_estimators=260, min_samples_leaf=2, class_weight="balanced", random_state=seed, n_jobs=-1)),
        ]
    return [
        ("mean baseline", DummyRegressor(strategy="mean")),
        ("ridge regression", Ridge(alpha=1.0)),
        ("random forest", RandomForestRegressor(n_estimators=240, min_samples_leaf=2, random_state=seed, n_jobs=-1)),
        ("extra trees", ExtraTreesRegressor(n_estimators=260, min_samples_leaf=2, random_state=seed, n_jobs=-1)),
    ]


def make_pipeline(features, estimator):
    return Pipeline([("preprocess", make_preprocessor(features)), ("model", estimator)])
