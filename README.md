# ML Pipeline Auditor

**Is your model's accuracy real?** An AI-agent skill that reviews machine-learning code and Jupyter notebooks like a senior data scientist. It finds the mistakes that make results look better than they are (data leakage, test-set misuse, wrong metrics, irreproducible runs, PyTorch training bugs), then writes a scored report (0–100) with the exact fix for each problem.

Works with Claude Code, Codex CLI, Cursor, Gemini CLI and other agents that use the `SKILL.md` standard.

**[See the full page with the demo and results ↗](https://m-shahmeer-kz.github.io/ml-pipeline-auditor/)**

> **[Get the skill on Agensi ↗](https://www.agensi.io/creators/muhammad-shahmeer)** ($12, one-time, includes future updates). This repository shows what the skill does and how it was tested. The skill itself (instructions, analysers and reference library) is sold there.

---

## One real example

A typical "Titanic survival" notebook ([`examples/leaky_titanic.py`](examples/leaky_titanic.py)) runs fine and reports an almost perfect score. Then the same model after applying the auditor's fixes ([`examples/fixed_titanic.py`](examples/fixed_titanic.py)) is run on the same 20 random splits of the real data:

| | Reported F1 (20 splits) | Audit score |
|---|---|---|
| Leaky notebook | **0.991** ± 0.005 | **0/100 (F)**: 8 findings |
| After the auditor's fixes | **0.749** ± 0.027 (honest) | **100/100 (A)**: 0 findings |

The 0.99 was not real: target encoding on all rows, SMOTE before the split, a scaler and fill values computed on the test rows, depth tuned on the test set and AUC computed from hard labels. Raw report: [`examples/sample-report.md`](examples/sample-report.md).

## Tested on real code

The analyser was run on 447 files from four well-known open-source ML projects. Every high/medium finding was checked by hand ([`results/manual_review.json`](results/manual_review.json)). Two projects were a **hold-out**: run once, with the analyser frozen, before any tuning.

| Project | Files | Findings checked | Result |
|---|---|---|---|
| scikit-learn official examples | 282 | 6 | 3 correct (incl. one the authors call "cheating a bit" in a comment), 3 intentional demos |
| *Hands-On ML* (3rd ed.) notebooks | 28 | 1 | 1 correct: test set used as validation data in chapter 17 |
| Microsoft *ML-For-Beginners* (hold-out) | 69 | 9 | 6 correct, 3 debatable |
| *Python Data Science Handbook* (hold-out) | 68 | 7 | 6 correct (incl. swapped `classification_report(ypred, ytest)`), 1 false alarm, fixed in v1.0.1 |

**Hold-out: 15 of 16 findings correct or debatable, 1 false alarm.** On 44 small labelled pipelines (29 with 31 planted problems, 15 clean): 31/31 caught, 0 false alarms; notebook-order and requirements rules are covered by separate tests. Full numbers: [`results/results.json`](results/results.json).

## What it checks

**26 code rules** in `.py` and `.ipynb` (cell-aware, magics handled):

| Area | Examples |
|---|---|
| Leakage | feature selection / target encoding / SMOTE / scaling / imputation before the split or outside CV, target column left in X, random split of time series, same patient/user on both sides |
| Validation | test set used for tuning or early stopping, model trained on test data, training-set-only evaluation, threshold tuned on test, no stratification |
| Metrics | AUC/log-loss from hard labels, swapped `metric(y_pred, y_true)`, accuracy as the only metric |
| Reproducibility | missing seeds, notebook cells run out of order or skipped, unpinned dependencies |
| Deep learning | missing `optimizer.zero_grad()`, missing `model.eval()`, softmax/sigmoid before a loss that expects logits |

**Data split checker**: compares the real train and test files: duplicate and near-duplicate rows across the split, shared IDs or groups, test rows older than training rows, single features that predict the target almost perfectly, target and feature drift. On a random split of the real Titanic data it flags `boat` (lifeboat number) as a target proxy: [`examples/split-check-titanic.md`](examples/split-check-titanic.md).

**Plus a manual review checklist** for what scanning cannot see: post-outcome features, preprocessing in SQL, wrong evaluation unit, metric that does not match the goal, and LLM/RAG evaluation contamination (few-shot examples from the test set, prompts tuned on the test set, LLM-as-judge bias).

## How the agent uses it

1. Runs the analyser over your notebooks and scripts.
2. Opens every finding and decides: confirmed, intended, or false alarm. No blind copying.
3. Goes through the manual checklist.
4. Checks your actual train/test files, if you have them.
5. Writes `ML_AUDIT_REPORT.md`: verdict, score, findings ordered by impact on your metric, before/after code fixes, "fix first" list.
6. With your OK, applies the fixes (usually one `Pipeline`) and re-scores.

For teams: SARIF output for GitHub code-scanning alerts ([sample](examples/sample-report.sarif)), `--fail-on high` to block leaky pull requests, JSON for dashboards, and `# mlaudit: ignore[RULE]` for accepted exceptions.

## Good to know

- **Private and local.** Nothing is uploaded. The analyser reads code, it never runs your training.
- **No install.** The code analyser is pure Python standard library (tested on Python 3.9, 3.10, 3.11, 3.12, 3.13, 3.14). The data checker needs only pandas and numpy.
- **Honest limits.** Static analysis can miss leaks that live in SQL, other repositories or the data itself, which is why the skill includes a data checker and a manual checklist. I wrote the labelled test cases myself; the real-world tables above are the independent check.

## What is in this repository

```
examples/   leaky and fixed Titanic pipelines, the real data, sample reports (Markdown, SARIF, split check)
results/    benchmark results and the manual verdict for every real-world finding
docs/       the project page (GitHub Pages)
```

The skill's source (instructions, analysers, 70 tests, reference library) is not public.

---

Made by [Muhammad Shahmeer](https://m-shahmeer-kz.github.io/), Data Science MSc (University of Naples Federico II) and AI engineer.
© 2026 Muhammad Shahmeer. All rights reserved. Example data: Titanic `titanic3` (Vanderbilt Biostatistics).
