# Evaluation notes

## Split and preprocessing

ModelBench creates one seeded holdout split and uses the same rows for every candidate. For classification, it stratifies the split when every class has enough examples and both partitions can contain each class. If there are too few examples, it falls back to an unstratified split rather than fabricating labels.

Each estimator is wrapped in its own scikit-learn pipeline. Numeric columns use median imputation and standard scaling; categorical columns use most-frequent imputation and one-hot encoding. The `ColumnTransformer` is fit as part of each pipeline, so held-out rows never influence imputation statistics or category vocabularies.

## Metrics

- **Classification:** macro F1 is the ranking metric; accuracy and balanced accuracy are shown alongside it. Macro F1 gives each class equal weight, which is useful when the target distribution is uneven.
- **Regression:** R² is the ranking metric; MAE and RMSE are also shown in the target's units.
- **Cross-validation:** when requested, `--cv` folds are built from training rows only. The report shows mean and population standard deviation for the ranking metric. If a class is too small for the selected fold count, cross-validation is marked unavailable rather than silently changing its design.
- **Runtime:** fit and prediction times are measured locally and are affected by machine load, CPU, and library versions.

The highest holdout score identifies the displayed leader. Because the same holdout ranks several models, its score can be optimistic after selecting that leader. Use cross-validation for stability, keep a truly untouched final test set for consequential projects, and compare models against a domain-specific baseline.

## Interpretation limits

Coefficient magnitude and tree impurity importance depend on encoding, scaling, model family, and correlated features. They indicate which transformed inputs influenced a fitted model's internal scoring; they do not establish causation, fairness, or feature necessity. Review source features and use a suitable explainability method for the actual application.
