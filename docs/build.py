"""Build docs/index.html from results/*.json so every number is a measured one.

Run:  python docs/build.py            (set AGENSI_URL inside to the live listing)
"""
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = json.loads((ROOT / "results" / "results.json").read_text())
REV = json.loads((ROOT / "results" / "manual_review.json").read_text())
REPORT = (ROOT / "examples" / "sample-report.md").read_text()
AGENSI_URL = "https://www.agensi.io/skills/ml-pipeline-auditor-is-your-model-s-accuracy-real"
GH = "https://github.com/m-shahmeer-kz/ml-pipeline-auditor"
e = html.escape

imp = R["impact_titanic"]
c = R["corpus"]
rw = R["real_world"]
hold = R["holdout_v100"]
files_total = sum(v["files_scanned"] for v in rw.values())

buy = (f'<a class="btn solid" href="{AGENSI_URL}" target="_blank" rel="noopener">Get it on Agensi · $12 ↗</a>'
       if AGENSI_URL else '<span class="btn solid off">Coming soon on Agensi</span>')

rows = ""
labels = {"scikit-learn examples": "scikit-learn official examples", "handson-ml3": "Hands-On ML (3rd ed.) notebooks",
          "ML-For-Beginners": "Microsoft ML-For-Beginners", "PythonDataScienceHandbook": "Python Data Science Handbook"}
for k, v in rw.items():
    vd = ", ".join(f"{n} {w}" for w, n in v["verdicts"].items())
    rows += (f"<tr><td>{e(labels.get(k, k))}</td><td>{'hold-out' if v['set'].startswith('hold') else 'development'}</td>"
             f"<td>{v['files_scanned']}</td><td>{v['reviewed']}</td><td>{e(vd)}</td></tr>")

notable = []
for proj, data in REV.items():
    for it in data["reviews"]:
        if it["verdict"] == "correct" and it["rule"] != "MLA403":
            notable.append((proj, it))
notable_html = "".join(
    f"<li><b>{e(p)}</b> · <code>{e(i['file'].split('/')[-1])}</code> · {e(i['rule'])}: {e(i['note'])}</li>" for p, i in notable)

# first three findings of the raw report, as a preview
blocks = REPORT.split("\n## ")
preview = "## " + "\n## ".join(blocks[1:4]) if len(blocks) > 3 else REPORT

RULES = [
    ("Leakage", "Feature selection, target encoding, SMOTE, scaling or imputation fitted before the split or outside CV · target left in X · random split of time series · same entity on both sides"),
    ("Validation", "Test set used for tuning or early stopping · model trained on test data · training-only evaluation · threshold tuned on test · no stratification"),
    ("Metrics", "AUC / log-loss from hard labels · swapped metric(y_pred, y_true) · accuracy as the only metric"),
    ("Reproducibility", "Missing seeds · notebook cells run out of order or skipped · unpinned dependencies"),
    ("Deep learning", "Missing optimizer.zero_grad() · missing model.eval() · softmax/sigmoid before a logits loss"),
]
rules_html = "".join(f'<div class="card"><h3>{a}</h3><p>{b}</p></div>' for a, b in RULES)

page = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ML Pipeline Auditor · is your model's accuracy real?</title>
<meta name="description" content="AI-agent skill that audits ML code and notebooks for data leakage, test-set misuse, wrong metrics and training bugs, with a scored report and fixes.">
<meta property="og:type" content="website">
<meta property="og:title" content="ML Pipeline Auditor · is your model's accuracy real?">
<meta property="og:description" content="AI-agent skill that audits ML code and notebooks for data leakage, test-set misuse, wrong metrics and training bugs. Tested results and sample reports.">
<meta property="og:url" content="https://m-shahmeer-kz.github.io/ml-pipeline-auditor/">
<meta property="og:image" content="https://m-shahmeer-kz.github.io/ml-pipeline-auditor/assets/og.png">
<meta property="og:image:width" content="1200"><meta property="og:image:height" content="627">
<meta name="twitter:card" content="summary_large_image">
<link rel="icon" href="assets/logo.png">
<style>
:root{{--bg:#060607;--card:#111113;--line:#26262b;--text:#f2f2f3;--mute:#9a9aa2;--ok:#7ee2a8;--bad:#ff8a8a}}
*{{box-sizing:border-box}}html{{-webkit-text-size-adjust:100%}}
body{{margin:0;background:var(--bg);color:var(--text);font:15px/1.6 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}}
.wrap{{max-width:1040px;margin:0 auto;padding:40px 20px 70px}}
h1{{font-size:clamp(32px,5.4vw,52px);line-height:1.04;margin:0 0 12px;letter-spacing:-.025em}}
h2{{font-size:clamp(20px,3vw,27px);margin:48px 0 10px;letter-spacing:-.01em}}
p,li{{color:#d6d6da}}.sub{{color:var(--mute);max-width:800px}}a{{color:var(--text)}}
code{{font:13px ui-monospace,SFMono-Regular,Menlo,monospace;background:#17171a;border:1px solid var(--line);border-radius:5px;padding:1px 5px;overflow-wrap:anywhere}}
pre{{background:#0d0d0f;border:1px solid var(--line);border-radius:12px;padding:14px;overflow:auto;max-height:430px;white-space:pre-wrap}}
pre code{{background:none;border:0;padding:0}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:16px 18px}}
.card h3{{margin:0 0 6px;font-size:16px}}.card p{{margin:0;font-size:14px}}
.stat b{{display:block;font-size:30px;line-height:1.1}}.stat span{{color:var(--mute);font-size:13px}}
.vs{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}@media(max-width:640px){{.vs{{grid-template-columns:1fr}}}}
.vs .card b{{font-size:42px;display:block;line-height:1.1}}.bad{{color:var(--bad)}}.ok{{color:var(--ok)}}
.tbl{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;font-size:14px;margin:10px 0}}
th,td{{border-bottom:1px solid var(--line);padding:8px;text-align:left;vertical-align:top}}th{{color:var(--mute);font-weight:500}}
.btns{{display:flex;flex-wrap:wrap;gap:8px;margin:16px 0}}
.btn{{display:inline-block;padding:10px 18px;border:1px solid var(--line);border-radius:999px;text-decoration:none;font-size:14px}}
.btn.solid{{background:var(--text);color:#000;border-color:var(--text);font-weight:600}}.btn.off{{opacity:.75;cursor:default}}
.steps{{counter-reset:s;list-style:none;padding:0}}.steps li{{counter-increment:s;margin:8px 0;padding-left:34px;position:relative}}
.steps li:before{{content:counter(s);position:absolute;left:0;top:1px;width:24px;height:24px;border-radius:50%;border:1px solid var(--line);text-align:center;font-size:13px;line-height:23px;color:var(--mute)}}
</style></head>
<body><div class="wrap">
<p class="sub" style="letter-spacing:.12em;text-transform:uppercase;font-size:12px">AI-agent skill · Claude Code · Codex · Cursor · Gemini CLI</p>
<h1>Is your model's accuracy real?</h1>
<p class="sub">ML Pipeline Auditor reviews machine-learning code and Jupyter notebooks like a senior data scientist.
It finds data leakage, test-set misuse, wrong metrics, irreproducible runs and PyTorch training bugs, then writes a
scored report (0–100) with the exact fix for each problem. Local and private: nothing is uploaded, your training is never run.</p>
<div class="btns">{buy}<a class="btn" href="#demo">See the demo</a><a class="btn" href="#tested">Tested on real code</a><a class="btn" href="{GH}">GitHub ↗</a></div>

<h2 id="demo">A 0.99 that was not real</h2>
<p class="sub">A typical Titanic notebook vs the same model after the auditor's fixes, on the same 20 random splits of the real data.</p>
<div class="vs">
 <div class="card"><span class="sub">Leaky notebook reports</span><b class="bad">F1 {imp['leaky']['f1_mean']:.3f}</b>
  <p>± {imp['leaky']['f1_sd']:.3f} · audit score <b style="display:inline;font-size:15px">{imp['audit_leaky']}/100</b> · {len(imp['leaky_rules'])} kinds of problem: target encoding and SMOTE before the split, scaler and fill values from test rows, depth tuned on the test set, AUC from labels</p></div>
 <div class="card"><span class="sub">After the fixes (honest)</span><b class="ok">F1 {imp['fixed']['f1_mean']:.3f}</b>
  <p>± {imp['fixed']['f1_sd']:.3f} · ROC-AUC {imp['fixed']['auc_mean']:.3f} · audit score <b style="display:inline;font-size:15px">{imp['audit_fixed']}/100</b> · one Pipeline, tuning by cross-validation, test set used once</p></div>
</div>
<p>Start of the raw report for the leaky notebook (the agent then confirms each finding and writes the full report):</p>
<pre><code>{e(preview)}</code></pre>

<h2 id="tested">Tested on real code</h2>
<div class="grid">
 <div class="card stat"><b>{files_total}</b><span>files from 4 well-known open-source ML projects</span></div>
 <div class="card stat"><b>{hold['correct_or_debatable']} / {hold['reviewed_high_medium']}</b><span>hold-out findings correct or debatable (code never seen before)</span></div>
 <div class="card stat"><b>{c['caught']} / {c['planted_problems']}</b><span>planted problems caught in {c['cases']} labelled pipelines</span></div>
 <div class="card stat"><b>{c['clean_cases_with_any_finding']} / {c['clean_cases']}</b><span>clean pipelines wrongly flagged</span></div>
</div>
<div class="tbl"><table><tr><th>Project</th><th>Set</th><th>Files</th><th>Findings checked by hand</th><th>Verdict</th></tr>{rows}</table></div>
<p>Real problems it found in famous code:</p><ul>{notable_html}</ul>
<p class="sub">Hold-out projects were run once with the analyser frozen. Its one false alarm (a K-Means example) was fixed afterwards in v1.0.1.
I wrote the labelled cases myself, so the real-world table is the independent check.</p>

<h2>What it checks</h2>
<div class="grid">{rules_html}
 <div class="card"><h3>Your train/test data</h3><p>Duplicate and near-duplicate rows across the split · shared IDs or groups · test rows older than training rows · single features that predict the target · drift</p></div>
 <div class="card"><h3>Manual checklist</h3><p>Post-outcome features, preprocessing in SQL, wrong evaluation unit, metric vs goal, LLM/RAG evaluation contamination and LLM-as-judge bias</p></div>
</div>

<h2>How your agent uses it</h2>
<ol class="steps">
<li>Runs the analyser over your notebooks and scripts (26 rules, cell-aware).</li>
<li>Opens every finding and decides: confirmed, intended or false alarm. No blind copying.</li>
<li>Works through the manual checklist for what scanning cannot see.</li>
<li>Checks your actual train/test files when you have them.</li>
<li>Writes <code>ML_AUDIT_REPORT.md</code>: verdict, score, findings ordered by impact, before/after code fixes.</li>
<li>With your OK, applies the fixes and re-scores.</li>
</ol>

<h2>For teams</h2>
<div class="grid">
 <div class="card"><h3>Pull-request gate</h3><p><code>--fail-on high</code> blocks merges that add leakage.</p></div>
 <div class="card"><h3>Code-scanning alerts</h3><p>SARIF 2.1.0 output shows findings on the changed lines in GitHub.</p></div>
 <div class="card"><h3>No install</h3><p>Pure Python standard library, tested on Python 3.9 to 3.14.</p></div>
 <div class="card"><h3>Exceptions</h3><p><code># mlaudit: ignore[MLA102] reason</code> for accepted cases.</p></div>
</div>

<div class="btns" style="margin-top:34px">{buy}<a class="btn" href="{GH}">GitHub ↗</a></div>
<p class="sub">Made by <a href="https://m-shahmeer-kz.github.io/">Muhammad Shahmeer</a> · Data Science MSc, University of Naples Federico II · © 2026, all rights reserved.</p>
</div></body></html>
"""
(ROOT / "docs" / "index.html").write_text(page)
print("wrote docs/index.html", len(page))
