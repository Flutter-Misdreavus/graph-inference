"""Generate an interactive HTML report from Graph-as-Code experiment logs."""

import glob
import json
import os
from collections import Counter, defaultdict
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def load_logs():
    """Load all JSONL log files and return a list of (run_name, records) tuples."""
    runs = []
    for path in sorted(glob.glob("logs/*.jsonl")):
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        # Infer run name from filename, e.g. deepseek_chat_run_...
        basename = os.path.splitext(os.path.basename(path))[0]
        name = basename.replace("_run_", " ").replace("_", " ").title()
        runs.append((name, records))
    return runs


def load_category_names():
    """Return mapping from label_id to arxiv category name."""
    df_nodes = pd.read_csv("Datas/Arxiv.csv")
    mapping = dict(df_nodes[["label_id", "category"]].drop_duplicates().sort_values("label_id").values)
    return mapping


def compute_run_metrics(records):
    """Compute overall metrics for a run."""
    total = len(records)
    correct = sum(1 for r in records if r["correct"])
    accuracy = correct / total if total else 0
    times = [r["elapsed_sec"] for r in records]
    avg_time = sum(times) / len(times) if times else 0
    return {
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "avg_time": avg_time,
    }


def compute_per_class_metrics(records):
    """Compute per-class accuracy."""
    stats = defaultdict(lambda: {"total": 0, "correct": 0})
    for r in records:
        tid = r["true_label"]
        stats[tid]["total"] += 1
        if r["correct"]:
            stats[tid]["correct"] += 1
    result = {}
    for tid, s in stats.items():
        result[tid] = {"accuracy": s["correct"] / s["total"], "total": s["total"], "correct": s["correct"]}
    return result


def build_confusion_matrix(records):
    """Build a sparse confusion matrix from misclassified records."""
    cm = defaultdict(int)
    for r in records:
        if not r["correct"] and r["predicted_label"] is not None:
            cm[(r["true_label"], r["predicted_label"])] += 1
    return cm


def chart_overview_kpis(runs, cat_names):
    """KPI cards at the top."""
    fig = make_subplots(
        rows=1, cols=len(runs),
        subplot_titles=[f"{name}" for name, _ in runs],
        specs=[[{"type": "indicator"}] * len(runs)],
    )
    colors = ["#4C72B0", "#DD8452"]
    for i, (name, records) in enumerate(runs):
        m = compute_run_metrics(records)
        fig.add_trace(
            go.Indicator(
                mode="number+delta",
                value=m["accuracy"] * 100,
                number={"suffix": "%", "font": {"size": 48}},
                delta={"reference": 74.4, "relative": False, "valueformat": ".1f", "suffix": "% 论文基准"},
                title={"text": f"准确率<br><span style='font-size:14px'>平均耗时: {m['avg_time']:.1f}s | 节点数: {m['total']}</span>"},
                domain={"row": 0, "column": i},
            ),
            row=1, col=i + 1,
        )
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=60, b=20), showlegend=False)
    return fig.to_html(full_html=False, include_plotlyjs=False)


def chart_per_class_accuracy(runs, cat_names):
    """Grouped horizontal bar chart of per-class accuracy."""
    all_labels = sorted(cat_names.keys())
    y_labels = [cat_names.get(lid, str(lid)) for lid in all_labels]

    fig = go.Figure()
    colors = ["#4C72B0", "#DD8452"]
    for idx, (name, records) in enumerate(runs):
        per_class = compute_per_class_metrics(records)
        accs = [per_class.get(lid, {"accuracy": 0})["accuracy"] * 100 for lid in all_labels]
        fig.add_trace(go.Bar(
            name=name,
            y=y_labels,
            x=accs,
            orientation="h",
            marker_color=colors[idx % len(colors)],
            hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
        ))

    fig.update_layout(
        barmode="group",
        title="各类别准确率对比",
        xaxis_title="准确率 (%)",
        yaxis_title="",
        height=1200,
        margin=dict(l=120, r=20, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)


def chart_confusion_heatmap(runs, cat_names):
    """Confusion heatmap per run (only showing misclassifications)."""
    all_labels = sorted(cat_names.keys())
    y_labels = [cat_names.get(lid, str(lid)) for lid in all_labels]
    x_labels = y_labels

    html_parts = []
    for name, records in runs:
        cm = build_confusion_matrix(records)
        if not cm:
            continue
        # Build full matrix
        z = [[0] * len(all_labels) for _ in range(len(all_labels))]
        for (t, p), c in cm.items():
            if t in all_labels and p in all_labels:
                ti = all_labels.index(t)
                pi = all_labels.index(p)
                z[ti][pi] = c

        fig = go.Figure(data=go.Heatmap(
            z=z,
            x=x_labels,
            y=y_labels,
            colorscale="YlOrRd",
            hoverongaps=False,
            hovertemplate="真实: %{y}<br>预测: %{x}<br>数量: %{z}<extra></extra>",
        ))
        fig.update_layout(
            title=f"混淆热力图 — {name} (仅误判)",
            xaxis_title="预测标签",
            yaxis_title="真实标签",
            height=800,
            width=900,
            margin=dict(l=120, r=20, t=60, b=100),
        )
        html_parts.append(fig.to_html(full_html=False, include_plotlyjs=False))

    return "<br>".join(html_parts)


def chart_time_distribution(runs):
    """Box plot comparing elapsed time by model and correctness."""
    fig = go.Figure()
    colors = {"correct": "#55A868", "wrong": "#C44E52"}
    for name, records in runs:
        for status in ["correct", "wrong"]:
            vals = [r["elapsed_sec"] for r in records if (r["correct"] and status == "correct") or (not r["correct"] and status == "wrong")]
            label = f"{name} — {'正确' if status == 'correct' else '错误'}"
            fig.add_trace(go.Box(
                y=vals,
                name=label,
                marker_color=colors[status],
                boxmean=True,
            ))
    fig.update_layout(
        title="预测耗时分布",
        yaxis_title="耗时 (秒)",
        height=500,
        margin=dict(l=60, r=20, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)


def chart_top_confusions(runs, cat_names):
    """Top confusion patterns across all runs."""
    combined = Counter()
    for _name, records in runs:
        cm = build_confusion_matrix(records)
        combined.update(cm)

    top = combined.most_common(20)
    if not top:
        return "<p>未发现误判样本。</p>"

    y_labels = [f"{cat_names.get(t, str(t))} → {cat_names.get(p, str(p))}" for (t, p), _ in top]
    counts = [c for _, c in top]

    fig = go.Figure(go.Bar(
        x=counts,
        y=y_labels,
        orientation="h",
        marker_color="#C44E52",
        hovertemplate="%{y}: %{x} 个节点<extra></extra>",
    ))
    fig.update_layout(
        title="Top 20 误判模式 (真实 → 预测)",
        xaxis_title="数量",
        yaxis_title="",
        height=600,
        margin=dict(l=200, r=20, t=50, b=40),
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)


def chart_class_frequency(runs, cat_names):
    """Bar chart showing how many test nodes each class had (should be identical across runs)."""
    all_labels = sorted(cat_names.keys())
    counts = []
    for lid in all_labels:
        c = sum(1 for _name, records in runs for r in records if r["true_label"] == lid)
        counts.append(c)

    fig = go.Figure(go.Bar(
        x=[cat_names[lid] for lid in all_labels],
        y=counts,
        marker_color="#8172B3",
        hovertemplate="%{x}: %{y} 个节点<extra></extra>",
    ))
    fig.update_layout(
        title="测试集各类别样本分布",
        xaxis_title="类别",
        yaxis_title="数量",
        height=450,
        margin=dict(l=60, r=20, t=50, b=120),
        xaxis_tickangle=-45,
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)


def generate_html(runs, cat_names):
    """Assemble all charts into a single HTML page."""
    kpi_html = chart_overview_kpis(runs, cat_names)
    class_acc_html = chart_per_class_accuracy(runs, cat_names)
    confusion_html = chart_confusion_heatmap(runs, cat_names)
    time_html = chart_time_distribution(runs)
    conf_html = chart_top_confusions(runs, cat_names)
    freq_html = chart_class_frequency(runs, cat_names)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Graph-as-Code 实验报表</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  :root {{
    --bg: #f8f9fa;
    --card: #ffffff;
    --text: #212529;
    --muted: #6c757d;
    --border: #dee2e6;
  }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    background: var(--bg);
    color: var(--text);
    margin: 0;
    padding: 0;
  }}
  header {{
    background: var(--card);
    border-bottom: 1px solid var(--border);
    padding: 2rem 1rem;
    text-align: center;
  }}
  header h1 {{ margin: 0 0 0.5rem; font-size: 1.75rem; }}
  header p {{ margin: 0; color: var(--muted); }}
  .container {{
    max-width: 1200px;
    margin: 0 auto;
    padding: 1rem;
  }}
  .chart-card {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
  }}
  .chart-card h2 {{
    margin: 0 0 1rem;
    font-size: 1.25rem;
    color: var(--text);
  }}
  footer {{
    text-align: center;
    padding: 2rem 1rem;
    color: var(--muted);
    font-size: 0.875rem;
  }}
</style>
</head>
<body>
<header>
  <h1>Graph-as-Code 实验报表</h1>
  <p>OGB Arxiv 节点分类 &middot; 生成时间 {timestamp}</p>
</header>
<div class="container">
  <div class="chart-card">
    <h2>概览</h2>
    {kpi_html}
  </div>
  <div class="chart-card">
    <h2>各类别准确率</h2>
    {class_acc_html}
  </div>
  <div class="chart-card">
    <h2>混淆热力图</h2>
    {confusion_html}
  </div>
  <div class="chart-card">
    <h2>预测耗时分布</h2>
    {time_html}
  </div>
  <div class="chart-card">
    <h2>主要误判模式</h2>
    {conf_html}
  </div>
  <div class="chart-card">
    <h2>测试集类别分布</h2>
    {freq_html}
  </div>
</div>
<footer>
  由 generate_report.py 生成 &middot; {timestamp}
</footer>
</body>
</html>"""
    return html


def main():
    runs = load_logs()
    if not runs:
        print("No log files found in logs/")
        return

    cat_names = load_category_names()

    print(f"已加载 {len(runs)} 个运行记录:")
    for name, records in runs:
        m = compute_run_metrics(records)
        print(f"  {name}: {m['correct']}/{m['total']} = {m['accuracy']:.2%}, 平均耗时 {m['avg_time']:.1f}s")

    html = generate_html(runs, cat_names)
    output_path = "report.html"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\n报表已保存至 {output_path}")


if __name__ == "__main__":
    main()
