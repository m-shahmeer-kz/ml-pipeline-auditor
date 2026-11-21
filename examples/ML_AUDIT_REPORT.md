# ML Pipeline Audit: titanic survival notebook

**Verdict:** the reported F1 of 0.99 cannot be trusted. Three problems let the model see the test answers, so the score measures leakage, not skill.

**Score: 0/100 (grade F)** · 1 file (`leaky_titanic.py`) + data check on `titanic3.csv` · Reported metric: F1 0.986, "AUC" 0.985

| Area | High | Medium | Low |
|---|---|---|---|
| Leakage | 2 | 4 | 0 |
| Validation | 1 | 0 | 1 |
| Metrics | 0 | 1 | 0 |
| Reproducibility | 0 | 0 | 0 |
| Deep learning | 0 | 0 | 0 |

## Fix first
1. Remove the full-data target encoding (`ticket_rate`, `cabin_rate`, `home.dest_rate`), or rebuild it with `TargetEncoder` inside a Pipeline.
2. Split before SMOTE, and put SMOTE inside an `imblearn` Pipeline.
3. Tune `max_depth` with cross-validation on the training data, then use the test set once.

After these three fixes the expected score is 100/100. Re-run training: honest F1 will be clearly lower (see the end).

## Findings

### 1. [HIGH] MLA106 Target encoding on the full dataset (lines 21-23)
- **What:** `df.groupby(col)["survived"].transform("mean")` for `ticket`, `cabin`, `home.dest`, computed on all 1,309 rows before the split.
- **Why it matters here:** 713 passengers (54%) have a ticket number nobody else has. For them `ticket_rate` **is their own survival label** (0 or 1). The forest simply reads it. This alone explains most of the 0.99.
- **Fix:**
```python
# before
df["ticket_rate"] = df.groupby("ticket")["survived"].transform("mean")
# after: learned inside each training fold with cross-fitting
from sklearn.preprocessing import TargetEncoder
("te", TargetEncoder(cv=KFold(5, shuffle=True, random_state=SEED)), ["ticket", "cabin", "home.dest"])
```

### 2. [HIGH] MLA104 SMOTE before the split (line 31)
- **What:** `SMOTE().fit_resample(X, y)` on all rows, then `train_test_split`.
- **Why it matters here:** synthetic survivors are interpolated from real passengers, so near-copies of training rows land in the test set (and the test set no longer has the real 38% survival rate).
- **Fix:** split first; `Pipeline([("pre", pre), ("smote", SMOTE()), ("rf", RandomForestClassifier())])` from `imblearn.pipeline`.

### 3. [HIGH] MLA201 Test set used to choose `max_depth` (lines 35-40)
- **What:** the loop keeps the depth with the best **test** F1, then reports test F1 again.
- **Fix:** `GridSearchCV(pipe, {"rf__max_depth": [3, 5, 8, 12, None]}, cv=StratifiedKFold(5, shuffle=True, random_state=SEED), scoring="f1")` on the training data only.

### 4. [MEDIUM] MLA301 "AUC" computed from hard labels (line 43)
- `roc_auc_score(y_test, pred)` with `pred = model.predict(...)`. Use `model.predict_proba(X_test)[:, 1]`.

### 5. [MEDIUM] MLA108 Fill values from all rows (lines 17, 18, 23)
- `age` mean and `fare` median include the test rows; line 23 even fills with the mean of the **target**. Use `SimpleImputer` inside the Pipeline. Small effect for age/fare; line 23 is part of finding 1.

### 6. [MEDIUM] MLA102 StandardScaler on all rows (line 30)
- Real leak in principle, but **harmless here**: random forests do not depend on feature scale. Fix it for correctness (inside the Pipeline), lowest priority.

## Checked and dismissed
- MLA206 (no `stratify`, line 32): valid hint. Add `stratify=y` when you split before SMOTE.

## Data split check
`split_check.py` on a random split of `titanic3.csv`: no duplicate rows across the split, class shares within 2.4 points, **but `boat` alone predicts survival (category purity 0.98)**. Good news: this notebook does not use `boat` or `body` (both are only known after the sinking). Keep it that way.

## What was reviewed manually
- Post-outcome features: `boat`, `body` correctly excluded. `home.dest` is known before the trip: fine as a raw feature, leaky only through the target encoding.
- Evaluation unit: passengers who travelled together share tickets. A group split by `ticket` would be stricter; mention it if the model must generalise to new families.
- Metric: F1 is reasonable for a 38% positive class. Also report ROC-AUC from probabilities.

## Re-measured after the fixes
Same model, all fixes applied (`fixed_titanic.py`), same 20 random splits:

| | F1 (mean ± sd, 20 splits) | Audit score |
|---|---|---|
| Original notebook | 0.991 ± 0.005 | 0/100 |
| After fixes | **0.749 ± 0.027** (ROC-AUC 0.871) | 100/100 |

---
Generated with ML Pipeline Auditor 1.0.1: static analysis + data check + manual review. Your training code was not run during the audit; the re-measurement above was run separately.
