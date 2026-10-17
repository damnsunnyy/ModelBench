"""Portable HTML and terminal views for benchmark results."""
from html import escape
import json


def encode_json(result):
    return json.dumps(result, indent=2, allow_nan=False)


def _metric(value):
    return "—" if value is None else f"{value:.3f}"


def html_report(result):
    task = result["task"]
    metric = result["primary_metric"]
    title = "Classification" if task == "classification" else "Regression"
    rows = []
    for model in result["results"]:
        if model["error"]:
            rows.append(f'<tr class="failed"><td>{escape(model["model"])}</td><td colspan="5">Model unavailable: {escape(model["error"])}</td></tr>')
            continue
        metrics = model["metrics"]
        primary = metrics.get(metric)
        width = max(0, min(100, primary * 100 if primary is not None else 0))
        cv = "—" if model["cv_mean"] is None else f"{model['cv_mean']:.3f} ± {model['cv_std']:.3f}"
        rows.append(f'''<tr><td><span class="rank">{model["rank"] or "—"}</span>{escape(model["model"])}</td><td><div class="metric-cell"><b>{_metric(primary)}</b><i><span style="width:{width:.1f}%"></span></i></div></td><td>{cv}</td><td>{_metric(model["fit_seconds"]) }s</td><td>{_metric(model["predict_seconds"]) }s</td><td>{" · ".join(f"{escape(key)} {_metric(value)}" for key,value in metrics.items())}</td></tr>''')
    profile = result["profile"]
    importance_rows = "".join(f'<div class="importance-row"><span>{escape(item["feature"])}</span><i><b style="width:{item["weight"]*100:.1f}%"></b></i><code>{item["weight"]:.3f}</code></div>' for item in result["feature_importance"])
    importance = f'<section class="panel"><div class="eyebrow">MODEL SIGNAL · {escape(result["best_model"])}</div><h2>Largest learned weights</h2><p class="muted">Built-in impurity importance for tree models or absolute coefficient share for logistic/ridge models. These are not causal effects.</p>{importance_rows}</section>' if importance_rows else ''
    confusion_html = ""
    matrix = result["confusion_matrix"]
    if matrix:
        labels = matrix["labels"]
        head = "".join(f"<th>{escape(label)}</th>" for label in labels)
        body = "".join(f"<tr><th>{escape(label)}</th>{''.join(f'<td>{value}</td>' for value in values)}</tr>" for label, values in zip(labels, matrix["matrix"]))
        confusion_html = f'<section class="panel"><div class="eyebrow">HOLDOUT · {result["test_rows"]} ROWS</div><h2>Confusion matrix</h2><p class="muted">Rows are actual labels; columns are predictions.</p><div class="table-scroll"><table class="confusion"><thead><tr><th>ACTUAL ↓ / PREDICTED →</th>{head}</tr></thead><tbody>{body}</tbody></table></div></section>'
    profile_items = [("Rows",profile["rows"]),("Features",profile["features"]),("Numeric",profile["numeric_features"]),("Categorical",profile["categorical_features"]),("Missing feature cells",profile["missing_feature_cells"]),("Unlabeled rows removed",profile["target_missing_rows_removed"])]
    profile_html = "".join(f'<div><span>{escape(str(label))}</span><b>{value}</b></div>' for label,value in profile_items)
    metric_labels = {"f1_macro":"Macro F1", "r2":"R²", "accuracy":"Accuracy", "balanced_accuracy":"Balanced accuracy", "mae":"MAE", "rmse":"RMSE"}
    main_score = next((row["metrics"].get(metric) for row in result["results"] if row["model"]==result["best_model"]),None)
    primary_label = metric_labels.get(metric,metric)
    cv_note = f"Cross-validation ({result['cv_folds']} folds) is run only on training rows." if result["cv_folds"] else "Enable --cv 5 to compare training-fold mean and spread."
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="color-scheme" content="dark"><title>ModelBench · {escape(result["dataset"])}</title><style>
:root{{--bg:#101411;--panel:#171e18;--line:#2b372d;--text:#edf1e8;--muted:#a4b09f;--acid:#c2ef66;--blue:#87ccd2;--orange:#f0bd68;--red:#f18b7f}}*{{box-sizing:border-box}}body{{margin:0;background:radial-gradient(ellipse at 80% -10%,#c2ef6610,transparent 40%),var(--bg);color:var(--text);font:14px/1.55 system-ui,sans-serif}}header,main,footer{{max-width:1140px;margin:auto;padding-left:24px;padding-right:24px}}header{{height:63px;display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid var(--line);color:var(--muted);font:10px ui-monospace,monospace;letter-spacing:.1em}}header b,.eyebrow{{color:var(--acid)}}main{{padding-top:55px;padding-bottom:55px}}.hero{{display:flex;justify-content:space-between;align-items:end;gap:20px;margin-bottom:28px}}.eyebrow{{font:9px ui-monospace,monospace;letter-spacing:.13em;margin:0 0 10px}}h1{{font-size:clamp(38px,7vw,68px);line-height:1;letter-spacing:-.07em;margin:0}}h1 em{{font:italic 400 1.05em Georgia,serif;color:var(--acid)}}h2{{font-size:18px;letter-spacing:-.04em;margin:5px 0 13px}}.dataset{{color:var(--muted);font:10px ui-monospace,monospace;margin-top:12px}}.best{{min-width:200px;padding:16px 18px;border:1px solid #c2ef6650;background:#c2ef660a}}.best span{{display:block;color:var(--muted);font:8px ui-monospace,monospace;letter-spacing:.1em}}.best b{{display:block;color:var(--acid);font-size:19px;margin-top:5px}}.best strong{{font:10px ui-monospace,monospace;color:#dce5d7}}.profile{{display:grid;grid-template-columns:repeat(6,1fr);gap:9px;margin:0 0 15px}}.profile>div,.panel{{border:1px solid var(--line);background:var(--panel)}}.profile>div{{padding:14px}}.profile span{{display:block;color:var(--muted);font:8px ui-monospace,monospace}}.profile b{{display:block;font-size:20px;margin-top:4px}}.panel{{padding:22px;margin-top:14px}}.muted,.note{{color:var(--muted);font-size:12px}}.leaderboard{{width:100%;border-collapse:collapse;margin-top:22px;font-size:12px}}.leaderboard th,.leaderboard td{{padding:13px 10px;text-align:left;border-bottom:1px solid var(--line);vertical-align:middle}}.leaderboard th{{color:#9eac99;font:8px ui-monospace,monospace;letter-spacing:.08em}}.leaderboard tr:first-child td:first-child{{color:var(--acid)}}.leaderboard tr.failed td{{color:var(--red)}}.rank{{display:inline-grid;place-items:center;width:20px;height:20px;margin-right:8px;border:1px solid #ffffff20;border-radius:50%;color:var(--muted);font:9px ui-monospace,monospace}}.metric-cell{{display:flex;align-items:center;gap:8px}}.metric-cell>b{{font:11px ui-monospace,monospace}}.metric-cell>i,.importance-row>i{{display:block;width:70px;height:4px;background:#ffffff16;border-radius:5px;overflow:hidden}}.metric-cell>i span,.importance-row>i b{{display:block;height:100%;background:var(--acid)}}.importance-row{{display:grid;grid-template-columns:minmax(130px,1fr) 2fr 45px;gap:13px;align-items:center;padding:9px 0;border-bottom:1px solid #ffffff10;font-size:11px}}.importance-row>i{{width:100%}}.importance-row code{{color:var(--acid);text-align:right;font:9px ui-monospace,monospace}}.table-scroll{{overflow-x:auto}}.confusion{{border-collapse:collapse;font:10px ui-monospace,monospace}}.confusion th,.confusion td{{padding:9px 12px;text-align:center;border:1px solid var(--line)}}.confusion th{{color:var(--muted)}}.confusion td{{background:#c2ef6610}}.note{{margin:15px 0 0;font-size:10px}}footer{{padding-top:18px;padding-bottom:22px;border-top:1px solid var(--line);color:#83917f;font:9px ui-monospace,monospace;display:flex;justify-content:space-between}}@media(max-width:780px){{.profile{{grid-template-columns:repeat(3,1fr)}}.leaderboard{{min-width:800px}}.leader-wrap{{overflow-x:auto}}}}@media(max-width:520px){{header,main,footer{{padding-left:16px;padding-right:16px}}main{{padding-top:36px}}.hero{{align-items:start;flex-direction:column}}.profile{{grid-template-columns:repeat(2,1fr)}}.best{{width:100%}}.panel{{padding:16px}}footer span:last-child{{display:none}}}}
</style></head><body><header><span><b>▧ MODELBENCH</b> / TABULAR LAB</span><span>LOCAL-FIRST · REPRODUCIBLE</span></header><main><section class="hero"><div><p class="eyebrow">{escape(title.upper())} · SEE HOW MODELS COMPARE</p><h1>Find a better<br><em>baseline.</em></h1><p class="dataset">{escape(result["dataset"])} · TARGET: {escape(result["target"])}</p></div><div class="best"><span>TOP HOLDOUT MODEL</span><b>{escape(result["best_model"])}</b><strong>{primary_label} · {_metric(main_score)}</strong></div></section><section class="profile">{profile_html}</section><section class="panel"><div class="eyebrow">01 / MODEL LEADERBOARD</div><h2>Same split. Same preprocessing. Honest comparison.</h2><p class="muted">Models are ranked by {escape(primary_label)} on {result["test_rows"]} held-out rows. All preprocessing is fitted inside each pipeline on training data only.</p><div class="leader-wrap"><table class="leaderboard"><thead><tr><th>MODEL</th><th>{escape(primary_label).upper()}</th><th>TRAIN CV</th><th>FIT</th><th>PREDICT</th><th>OTHER METRICS</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div><p class="note">{cv_note} Model selection uses the holdout leaderboard; treat results as a comparison on this split, not an unbiased estimate after selection. Change --seed or use cross-validation before making decisions.</p></section>{importance}{confusion_html}<footer><span>MODELBENCH · A SMALL, EXPLAINABLE COMPARISON</span><span>DATA NEVER LEAVES YOUR MACHINE</span></footer></main></body></html>'''


def console_report(result):
    print(f"\nModelBench · {result['dataset']} · {result['task']} · target={result['target']}")
    print(f"{result['train_rows']} training rows · {result['test_rows']} held-out rows · ranked by {result['primary_metric']}")
    for row in result["results"]:
        if row["error"]:
            print(f"  — {row['model']}: unavailable ({row['error']})")
            continue
        score = row["metrics"][result["primary_metric"]]
        cv = f" · CV {row['cv_mean']:.3f} ± {row['cv_std']:.3f}" if row["cv_mean"] is not None else ""
        metrics = " · ".join(f"{key}={value:.3f}" for key,value in row["metrics"].items())
        print(f"  {row['rank']:>2}. {row['model']:<22} {result['primary_metric']}={score:.3f}{cv} · {metrics}")
    print(f"\nTop model on this holdout: {result['best_model']}")
    print("Treat this leaderboard as a split comparison, not an unbiased estimate after selecting a winner.")
