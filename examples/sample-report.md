# ML Pipeline Audit

**Score: 0/100 (grade F)** · 1 file(s) scanned · 8 finding(s)

> Reported scores are likely INFLATED: fix the high-severity leakage/validation findings before trusting them.

| Category | High | Medium | Low |
|---|---|---|---|
| leakage | 2 | 4 | 0 |
| validation | 1 | 0 | 0 |
| metrics | 0 | 1 | 0 |
| reproducibility | 0 | 0 | 0 |
| deep-learning | 0 | 0 | 0 |

## 1. [HIGH] MLA106 Target-based feature (target/mean encoding) computed on the full dataset

- **Where:** `leaky_titanic.py:22` (related lines [32])
- **Found:** Target encoding (mean of target 'survived' per group) computed before train_test_split (line 32).
- **Code:** `df[col + "_rate"] = df.groupby(col)["survived"].transform("mean")`
- **Why it matters:** Each row's feature contains its own label, so the model reads the answer.
- **Fix:** Compute encodings inside CV on training folds only (sklearn TargetEncoder in a Pipeline uses cross-fitting).
- **Confidence:** high

## 2. [HIGH] MLA104 Resampling (SMOTE/over/under-sampling) before the split or outside cross-validation

- **Where:** `leaky_titanic.py:31` (related lines [32])
- **Found:** SMOTE applied to 'X' (all rows) before train_test_split (line 32): resampled rows appear in both train and test.
- **Code:** `X, y = SMOTE(random_state=SEED).fit_resample(X, y)`
- **Why it matters:** Synthetic or duplicated rows end up in both train and test folds, which inflates recall/F1/AUC.
- **Fix:** Resample only the training part, e.g. imblearn.pipeline.Pipeline([('smote', SMOTE()), ('model', clf)]) inside CV.
- **Confidence:** high

## 3. [HIGH] MLA201 Test set used to choose hyperparameters or models

- **Where:** `leaky_titanic.py:37`
- **Found:** A model/parameter is chosen inside this loop by its score on test data.
- **Code:** `f1 = f1_score(y_test, model.predict(X_test))`
- **Why it matters:** Picking the best setting by its test score makes the test score optimistic; it is no longer an unbiased estimate.
- **Fix:** Tune with cross-validation or a separate validation set; touch the test set once, at the end.
- **Confidence:** high

## 4. [MEDIUM] MLA108 Missing values filled with full-dataset statistics before the split

- **Where:** `leaky_titanic.py:17` (related lines [32])
- **Found:** missing values filled with full-data statistics before train_test_split (line 32).
- **Code:** `df["age"] = df["age"].fillna(df["age"].mean())`
- **Why it matters:** fillna(df.mean()) uses test rows to compute the fill value.
- **Fix:** Use SimpleImputer inside a Pipeline, or compute the statistic on the training split only.
- **Confidence:** high

## 5. [MEDIUM] MLA108 Missing values filled with full-dataset statistics before the split

- **Where:** `leaky_titanic.py:18` (related lines [32])
- **Found:** missing values filled with full-data statistics before train_test_split (line 32).
- **Code:** `df["fare"] = df["fare"].fillna(df["fare"].median())`
- **Why it matters:** fillna(df.mean()) uses test rows to compute the fill value.
- **Fix:** Use SimpleImputer inside a Pipeline, or compute the statistic on the training split only.
- **Confidence:** high

## 6. [MEDIUM] MLA108 Missing values filled with full-dataset statistics before the split

- **Where:** `leaky_titanic.py:23` (related lines [32])
- **Found:** missing values filled with full-data statistics before train_test_split (line 32).
- **Code:** `df[col + "_rate"] = df[col + "_rate"].fillna(df["survived"].mean())`
- **Why it matters:** fillna(df.mean()) uses test rows to compute the fill value.
- **Fix:** Use SimpleImputer inside a Pipeline, or compute the statistic on the training split only.
- **Confidence:** high

## 7. [MEDIUM] MLA102 Preprocessing statistics computed on the full dataset before the split

- **Where:** `leaky_titanic.py:30` (related lines [32])
- **Found:** StandardScaler fitted on 'X' (all rows) before train_test_split (line 32).
- **Code:** `X = StandardScaler().fit_transform(X)`
- **Why it matters:** Scaler/imputer/PCA/vectorizer statistics include the test rows, so the test set is no longer unseen.
- **Fix:** Fit the transformer on the training data only (or inside a Pipeline) and only .transform() the test data.
- **Confidence:** high

## 8. [MEDIUM] MLA301 Probability metric computed from hard class labels

- **Where:** `leaky_titanic.py:43`
- **Found:** roc_auc_score() receives hard labels from .predict(); use predict_proba/decision_function.
- **Code:** `print(json.dumps({"f1": f1_score(y_test, pred), "auc_reported": roc_auc_score(y_test, pred)}))`
- **Why it matters:** roc_auc/average_precision/log_loss need scores or probabilities; with .predict() labels the number is wrong.
- **Fix:** Use model.predict_proba(X)[:, 1] or model.decision_function(X).
- **Confidence:** high

