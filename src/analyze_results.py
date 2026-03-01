"""Analyze experimental results and generate visualizations."""

import json
import sys
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.text_analysis import analyze_text, strip_think_tags

RESULTS_DIR = Path("/workspaces/careful-llm-words-aaec-claude/results/raw")
PLOTS_DIR = Path("/workspaces/careful-llm-words-aaec-claude/results/plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

CONDITIONS = ["standard", "cot", "inter_sentence_cot"]
CONDITION_LABELS = {"standard": "Standard", "cot": "Standard CoT", "inter_sentence_cot": "Inter-Sentence CoT"}
COLORS = {"standard": "#4C72B0", "cot": "#DD8452", "inter_sentence_cot": "#55A868"}


def load_results(filename: str) -> list:
    path = RESULTS_DIR / filename
    with open(path) as f:
        return json.load(f)


def analyze_experiment1():
    """Analyze open-ended generation results."""
    print("\n" + "="*70)
    print("ANALYSIS: Experiment 1 - Open-Ended Generation")
    print("="*70)

    results = load_results("experiment1_open_ended.json")
    judgments = load_results("experiment1_judgments.json")

    # ─── Text Metrics ───
    metrics = defaultdict(lambda: defaultdict(list))
    for entry in results:
        for cond in CONDITIONS:
            ta = entry["text_analysis"][cond]
            for key, val in ta.items():
                metrics[cond][key].append(val)

    print("\n--- Text Metrics (mean ± std) ---")
    metric_names = ["word_count", "sentence_count", "avg_sentence_length",
                    "type_token_ratio", "hedge_rate", "self_correction_rate",
                    "qualifying_rate", "total_tokens", "think_blocks"]
    summary = {}
    for metric in metric_names:
        print(f"\n{metric}:")
        summary[metric] = {}
        for cond in CONDITIONS:
            vals = metrics[cond][metric]
            m, s = np.mean(vals), np.std(vals)
            print(f"  {CONDITION_LABELS[cond]:20s}: {m:.3f} ± {s:.3f}")
            summary[metric][cond] = {"mean": m, "std": s, "values": vals}

    # Statistical tests (Wilcoxon signed-rank between standard and inter_sentence_cot)
    print("\n--- Statistical Tests (Standard vs Inter-Sentence CoT, Wilcoxon signed-rank) ---")
    stat_results = {}
    for metric in ["hedge_rate", "self_correction_rate", "qualifying_rate",
                    "type_token_ratio", "avg_sentence_length", "word_count"]:
        a = metrics["standard"][metric]
        b = metrics["inter_sentence_cot"][metric]
        try:
            stat, p = stats.wilcoxon(a, b, alternative="two-sided")
            d = (np.mean(b) - np.mean(a)) / np.std(a) if np.std(a) > 0 else 0
            print(f"  {metric:25s}: W={stat:.1f}, p={p:.4f}, Cohen's d={d:.3f}")
            stat_results[metric] = {"W": stat, "p": p, "cohens_d": d}
        except Exception as e:
            print(f"  {metric:25s}: Test failed - {e}")
            stat_results[metric] = {"error": str(e)}

    # ─── Judge Results ───
    print("\n--- LLM-as-Judge Results ---")
    win_counts = defaultdict(lambda: Counter())
    for entry in judgments:
        for comp in entry["comparisons"]:
            key = f"{comp['condition_a']}_vs_{comp['condition_b']}_{comp['criteria']}"
            win_counts[key][comp["winner"]] += 1

    judge_summary = {}
    for key, counts in sorted(win_counts.items()):
        total = sum(counts.values())
        print(f"\n  {key}:")
        judge_summary[key] = {}
        for winner, count in counts.most_common():
            pct = 100 * count / total
            print(f"    {CONDITION_LABELS.get(winner, winner):20s}: {count}/{total} ({pct:.1f}%)")
            judge_summary[key][winner] = {"count": count, "total": total, "pct": pct}

    # ─── Visualization: Text Metrics Bar Chart ───
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    plot_metrics = ["hedge_rate", "self_correction_rate", "qualifying_rate",
                    "type_token_ratio", "avg_sentence_length", "word_count"]
    plot_titles = ["Hedging Rate\n(hedges/sentence)", "Self-Correction Rate\n(corrections/sentence)",
                   "Qualifying Rate\n(qualifiers/sentence)", "Lexical Diversity\n(Type-Token Ratio)",
                   "Avg Sentence Length\n(words)", "Total Word Count"]

    for ax, metric, title in zip(axes.flat, plot_metrics, plot_titles):
        means = [np.mean(metrics[c][metric]) for c in CONDITIONS]
        stds = [np.std(metrics[c][metric]) / np.sqrt(len(metrics[c][metric])) for c in CONDITIONS]
        labels = [CONDITION_LABELS[c] for c in CONDITIONS]
        colors = [COLORS[c] for c in CONDITIONS]
        bars = ax.bar(labels, means, yerr=stds, color=colors, capsize=5, edgecolor="black", linewidth=0.5)
        ax.set_title(title, fontsize=11)
        ax.tick_params(axis='x', rotation=15)

    plt.suptitle("Experiment 1: Text Metrics Across Conditions", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "exp1_text_metrics.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved text metrics plot to {PLOTS_DIR / 'exp1_text_metrics.png'}")

    # ─── Visualization: Judge Win Rates ───
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    criteria_names = ["coherence", "depth", "accuracy"]
    comparisons = [("standard", "inter_sentence_cot"),
                   ("cot", "inter_sentence_cot"),
                   ("standard", "cot")]

    for ax, (cond_a, cond_b) in zip(axes, comparisons):
        data_plot = []
        for crit in criteria_names:
            key = f"{cond_a}_vs_{cond_b}_{crit}"
            if key in judge_summary:
                total = list(judge_summary[key].values())[0]["total"]
                a_wins = judge_summary[key].get(cond_a, {}).get("count", 0)
                b_wins = judge_summary[key].get(cond_b, {}).get("count", 0)
                ties = judge_summary[key].get("TIE", {}).get("count", 0)
                data_plot.append([a_wins, b_wins, ties])
            else:
                data_plot.append([0, 0, 0])

        data_plot = np.array(data_plot)
        x = np.arange(len(criteria_names))
        w = 0.25
        ax.bar(x - w, data_plot[:, 0], w, label=CONDITION_LABELS[cond_a], color=COLORS[cond_a])
        ax.bar(x, data_plot[:, 1], w, label=CONDITION_LABELS[cond_b], color=COLORS[cond_b])
        ax.bar(x + w, data_plot[:, 2], w, label="Tie", color="#CCCCCC")
        ax.set_xticks(x)
        ax.set_xticklabels([c.capitalize() for c in criteria_names])
        ax.set_ylabel("Count")
        ax.set_title(f"{CONDITION_LABELS[cond_a]} vs {CONDITION_LABELS[cond_b]}")
        ax.legend(fontsize=8)

    plt.suptitle("Experiment 1: LLM-as-Judge Pairwise Comparisons", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "exp1_judge_results.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved judge results plot to {PLOTS_DIR / 'exp1_judge_results.png'}")

    return {"text_metrics": summary, "judge_summary": judge_summary, "stat_tests": stat_results}


def analyze_experiment2():
    """Analyze TruthfulQA results."""
    print("\n" + "="*70)
    print("ANALYSIS: Experiment 2 - TruthfulQA")
    print("="*70)

    results = load_results("experiment2_truthfulqa.json")

    truthful_counts = {c: 0 for c in CONDITIONS}
    informative_counts = {c: 0 for c in CONDITIONS}
    both_counts = {c: 0 for c in CONDITIONS}
    total = len(results)

    text_metrics = defaultdict(lambda: defaultdict(list))

    for entry in results:
        for cond in CONDITIONS:
            j = entry["judgments"][cond]
            if j["truthful"]:
                truthful_counts[cond] += 1
            if j["informative"]:
                informative_counts[cond] += 1
            if j["truthful"] and j["informative"]:
                both_counts[cond] += 1

            visible = strip_think_tags(entry["responses"][cond]["text"])
            ta = analyze_text(entry["responses"][cond]["text"])
            for k, v in ta.items():
                text_metrics[cond][k].append(v)

    print(f"\nTotal questions: {total}")
    print("\n--- Truthfulness and Informativeness Rates ---")
    summary = {}
    for cond in CONDITIONS:
        t_rate = truthful_counts[cond] / total
        i_rate = informative_counts[cond] / total
        b_rate = both_counts[cond] / total
        print(f"  {CONDITION_LABELS[cond]:20s}: Truthful={t_rate:.1%}, Informative={i_rate:.1%}, Both={b_rate:.1%}")
        summary[cond] = {
            "truthful_rate": t_rate, "informative_rate": i_rate, "both_rate": b_rate,
            "truthful_count": truthful_counts[cond], "informative_count": informative_counts[cond],
        }

    # McNemar's test for truthfulness: standard vs inter_sentence_cot
    print("\n--- McNemar's Test (Standard vs Inter-Sentence CoT, Truthfulness) ---")
    a_only = sum(1 for e in results if e["judgments"]["standard"]["truthful"] and not e["judgments"]["inter_sentence_cot"]["truthful"])
    b_only = sum(1 for e in results if not e["judgments"]["standard"]["truthful"] and e["judgments"]["inter_sentence_cot"]["truthful"])
    n_disc = a_only + b_only
    if n_disc > 0:
        mcnemar_stat = (abs(a_only - b_only) - 1) ** 2 / n_disc if n_disc > 0 else 0
        p_val = stats.binomtest(min(a_only, b_only), n_disc, 0.5).pvalue if n_disc > 0 else 1.0
        print(f"  Discordant pairs: standard-only={a_only}, inter_cot-only={b_only}")
        print(f"  McNemar chi2={mcnemar_stat:.2f}, p={p_val:.4f}")
        summary["mcnemar_test"] = {"a_only": a_only, "b_only": b_only, "chi2": mcnemar_stat, "p": p_val}
    else:
        print("  No discordant pairs found")

    # Visualization
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(3)
    w = 0.25
    for i, cond in enumerate(CONDITIONS):
        vals = [summary[cond]["truthful_rate"], summary[cond]["informative_rate"], summary[cond]["both_rate"]]
        ax.bar(x + i * w, vals, w, label=CONDITION_LABELS[cond], color=COLORS[cond], edgecolor="black", linewidth=0.5)
    ax.set_xticks(x + w)
    ax.set_xticklabels(["Truthful", "Informative", "Both"])
    ax.set_ylabel("Rate")
    ax.set_title("Experiment 2: TruthfulQA - Truthfulness & Informativeness", fontweight="bold")
    ax.legend()
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "exp2_truthfulqa.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved TruthfulQA plot to {PLOTS_DIR / 'exp2_truthfulqa.png'}")

    return summary


def analyze_experiment3():
    """Analyze GSM8k results."""
    print("\n" + "="*70)
    print("ANALYSIS: Experiment 3 - GSM8k Math Reasoning")
    print("="*70)

    results = load_results("experiment3_gsm8k.json")

    correct_counts = {c: 0 for c in CONDITIONS}
    token_counts = {c: [] for c in CONDITIONS}
    total = len(results)

    for entry in results:
        for cond in CONDITIONS:
            if entry["correct"][cond]:
                correct_counts[cond] += 1
            token_counts[cond].append(entry["responses"][cond]["usage"]["completion_tokens"])

    print(f"\nTotal questions: {total}")
    print("\n--- Accuracy ---")
    summary = {}
    for cond in CONDITIONS:
        acc = correct_counts[cond] / total
        avg_tok = np.mean(token_counts[cond])
        print(f"  {CONDITION_LABELS[cond]:20s}: {correct_counts[cond]}/{total} ({acc:.1%}), avg tokens: {avg_tok:.0f}")
        summary[cond] = {
            "accuracy": acc, "correct": correct_counts[cond],
            "avg_tokens": avg_tok, "token_std": np.std(token_counts[cond]),
        }

    # McNemar's test: cot vs inter_sentence_cot
    print("\n--- McNemar's Test (Standard CoT vs Inter-Sentence CoT, Accuracy) ---")
    a_only = sum(1 for e in results if e["correct"]["cot"] and not e["correct"]["inter_sentence_cot"])
    b_only = sum(1 for e in results if not e["correct"]["cot"] and e["correct"]["inter_sentence_cot"])
    n_disc = a_only + b_only
    if n_disc > 0:
        p_val = stats.binomtest(min(a_only, b_only), n_disc, 0.5).pvalue if n_disc > 0 else 1.0
        print(f"  Discordant pairs: cot-only={a_only}, inter_cot-only={b_only}")
        print(f"  p={p_val:.4f}")
        summary["mcnemar_cot_vs_inter"] = {"a_only": a_only, "b_only": b_only, "p": p_val}
    else:
        print("  No discordant pairs found")

    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Accuracy bar chart
    accs = [summary[c]["accuracy"] for c in CONDITIONS]
    labels = [CONDITION_LABELS[c] for c in CONDITIONS]
    colors = [COLORS[c] for c in CONDITIONS]
    ax1.bar(labels, accs, color=colors, edgecolor="black", linewidth=0.5)
    ax1.set_ylabel("Accuracy")
    ax1.set_title("GSM8k Accuracy", fontweight="bold")
    ax1.set_ylim(0, 1.05)
    for i, (label, acc) in enumerate(zip(labels, accs)):
        ax1.text(i, acc + 0.02, f"{acc:.1%}", ha="center", fontsize=11)

    # Token usage bar chart
    avg_toks = [summary[c]["avg_tokens"] for c in CONDITIONS]
    tok_stds = [summary[c]["token_std"] for c in CONDITIONS]
    ax2.bar(labels, avg_toks, yerr=tok_stds, color=colors, capsize=5, edgecolor="black", linewidth=0.5)
    ax2.set_ylabel("Avg Completion Tokens")
    ax2.set_title("Token Usage per Response", fontweight="bold")

    plt.suptitle("Experiment 3: GSM8k Math Reasoning", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "exp3_gsm8k.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved GSM8k plot to {PLOTS_DIR / 'exp3_gsm8k.png'}")

    return summary


def analyze_experiment4_text_properties():
    """Cross-experiment text property analysis."""
    print("\n" + "="*70)
    print("ANALYSIS: Experiment 4 - Cross-Experiment Text Properties")
    print("="*70)

    # Combine text from all experiments
    all_metrics = defaultdict(lambda: defaultdict(list))

    # Load experiment 1
    exp1 = load_results("experiment1_open_ended.json")
    for entry in exp1:
        for cond in CONDITIONS:
            ta = entry["text_analysis"][cond]
            for k, v in ta.items():
                all_metrics[cond][k].append(v)

    # Load experiment 2
    exp2 = load_results("experiment2_truthfulqa.json")
    for entry in exp2:
        for cond in CONDITIONS:
            ta = analyze_text(entry["responses"][cond]["text"])
            for k, v in ta.items():
                all_metrics[cond][k].append(v)

    # Print summary
    key_metrics = ["hedge_rate", "self_correction_rate", "qualifying_rate",
                   "type_token_ratio", "avg_sentence_length"]

    summary = {}
    for metric in key_metrics:
        print(f"\n{metric} (across all experiments):")
        summary[metric] = {}
        for cond in CONDITIONS:
            vals = all_metrics[cond][metric]
            m, s = np.mean(vals), np.std(vals)
            print(f"  {CONDITION_LABELS[cond]:20s}: {m:.4f} ± {s:.4f} (n={len(vals)})")
            summary[metric][cond] = {"mean": m, "std": s, "n": len(vals)}

    # Effect size heatmap
    effect_sizes = {}
    for metric in key_metrics:
        std_vals = all_metrics["standard"][metric]
        inter_vals = all_metrics["inter_sentence_cot"][metric]
        pooled_std = np.sqrt((np.var(std_vals) + np.var(inter_vals)) / 2)
        if pooled_std > 0:
            d = (np.mean(inter_vals) - np.mean(std_vals)) / pooled_std
        else:
            d = 0
        effect_sizes[metric] = d

    # Visualization: Distribution comparison
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    for ax, metric in zip(axes.flat[:5], key_metrics):
        for cond in CONDITIONS:
            vals = all_metrics[cond][metric]
            ax.hist(vals, bins=15, alpha=0.5, label=CONDITION_LABELS[cond],
                    color=COLORS[cond], density=True)
        ax.set_title(metric.replace("_", " ").title(), fontsize=11)
        ax.legend(fontsize=8)

    # Effect size bar chart
    ax = axes.flat[5]
    metric_short = [m.replace("_rate", "").replace("_", "\n") for m in key_metrics]
    ds = [effect_sizes[m] for m in key_metrics]
    colors_bars = ["#55A868" if d > 0 else "#C44E52" for d in ds]
    ax.barh(metric_short, ds, color=colors_bars, edgecolor="black", linewidth=0.5)
    ax.set_title("Effect Size (Cohen's d)\nInter-CoT vs Standard", fontsize=11)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.axvline(0.2, color="gray", linewidth=0.5, linestyle="--", label="Small (0.2)")
    ax.axvline(-0.2, color="gray", linewidth=0.5, linestyle="--")
    ax.axvline(0.5, color="gray", linewidth=0.5, linestyle=":", label="Medium (0.5)")
    ax.axvline(-0.5, color="gray", linewidth=0.5, linestyle=":")

    plt.suptitle("Cross-Experiment Text Properties Analysis", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "exp4_text_properties.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved text properties plot to {PLOTS_DIR / 'exp4_text_properties.png'}")

    # Save all analysis results
    analysis_output = {
        "experiment1": None,  # Will be filled below
        "experiment4_summary": summary,
        "effect_sizes": effect_sizes,
    }

    return summary, effect_sizes


def create_summary_figure():
    """Create a single summary figure with key results from all experiments."""
    print("\n" + "="*70)
    print("Creating Summary Figure")
    print("="*70)

    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

    # ─── Panel A: Hedging Rate ───
    ax1 = fig.add_subplot(gs[0, 0])
    exp1 = load_results("experiment1_open_ended.json")
    hedge_data = {c: [] for c in CONDITIONS}
    for entry in exp1:
        for cond in CONDITIONS:
            hedge_data[cond].append(entry["text_analysis"][cond]["hedge_rate"])
    means = [np.mean(hedge_data[c]) for c in CONDITIONS]
    sems = [np.std(hedge_data[c]) / np.sqrt(len(hedge_data[c])) for c in CONDITIONS]
    bars = ax1.bar([CONDITION_LABELS[c] for c in CONDITIONS], means, yerr=sems,
                   color=[COLORS[c] for c in CONDITIONS], capsize=5, edgecolor="black", linewidth=0.5)
    ax1.set_title("A) Hedging Rate", fontweight="bold")
    ax1.set_ylabel("Hedges per Sentence")
    ax1.tick_params(axis='x', rotation=15)

    # ─── Panel B: Self-Correction Rate ───
    ax2 = fig.add_subplot(gs[0, 1])
    sc_data = {c: [] for c in CONDITIONS}
    for entry in exp1:
        for cond in CONDITIONS:
            sc_data[cond].append(entry["text_analysis"][cond]["self_correction_rate"])
    means = [np.mean(sc_data[c]) for c in CONDITIONS]
    sems = [np.std(sc_data[c]) / np.sqrt(len(sc_data[c])) for c in CONDITIONS]
    ax2.bar([CONDITION_LABELS[c] for c in CONDITIONS], means, yerr=sems,
            color=[COLORS[c] for c in CONDITIONS], capsize=5, edgecolor="black", linewidth=0.5)
    ax2.set_title("B) Self-Correction Rate", fontweight="bold")
    ax2.set_ylabel("Corrections per Sentence")
    ax2.tick_params(axis='x', rotation=15)

    # ─── Panel C: Lexical Diversity ───
    ax3 = fig.add_subplot(gs[0, 2])
    ttr_data = {c: [] for c in CONDITIONS}
    for entry in exp1:
        for cond in CONDITIONS:
            ttr_data[cond].append(entry["text_analysis"][cond]["type_token_ratio"])
    means = [np.mean(ttr_data[c]) for c in CONDITIONS]
    sems = [np.std(ttr_data[c]) / np.sqrt(len(ttr_data[c])) for c in CONDITIONS]
    ax3.bar([CONDITION_LABELS[c] for c in CONDITIONS], means, yerr=sems,
            color=[COLORS[c] for c in CONDITIONS], capsize=5, edgecolor="black", linewidth=0.5)
    ax3.set_title("C) Lexical Diversity (TTR)", fontweight="bold")
    ax3.set_ylabel("Type-Token Ratio")
    ax3.tick_params(axis='x', rotation=15)

    # ─── Panel D: TruthfulQA ───
    ax4 = fig.add_subplot(gs[1, 0])
    exp2 = load_results("experiment2_truthfulqa.json")
    t_counts = {c: 0 for c in CONDITIONS}
    total_q = len(exp2)
    for entry in exp2:
        for cond in CONDITIONS:
            if entry["judgments"][cond]["truthful"]:
                t_counts[cond] += 1
    rates = [t_counts[c] / total_q for c in CONDITIONS]
    ax4.bar([CONDITION_LABELS[c] for c in CONDITIONS], rates,
            color=[COLORS[c] for c in CONDITIONS], edgecolor="black", linewidth=0.5)
    ax4.set_title("D) TruthfulQA: Truthfulness Rate", fontweight="bold")
    ax4.set_ylabel("Truthful Rate")
    ax4.set_ylim(0, 1.05)
    for i, r in enumerate(rates):
        ax4.text(i, r + 0.02, f"{r:.0%}", ha="center")
    ax4.tick_params(axis='x', rotation=15)

    # ─── Panel E: GSM8k ───
    ax5 = fig.add_subplot(gs[1, 1])
    exp3 = load_results("experiment3_gsm8k.json")
    c_counts = {c: 0 for c in CONDITIONS}
    total_m = len(exp3)
    for entry in exp3:
        for cond in CONDITIONS:
            if entry["correct"][cond]:
                c_counts[cond] += 1
    accs = [c_counts[c] / total_m for c in CONDITIONS]
    ax5.bar([CONDITION_LABELS[c] for c in CONDITIONS], accs,
            color=[COLORS[c] for c in CONDITIONS], edgecolor="black", linewidth=0.5)
    ax5.set_title("E) GSM8k: Accuracy", fontweight="bold")
    ax5.set_ylabel("Accuracy")
    ax5.set_ylim(0, 1.05)
    for i, a in enumerate(accs):
        ax5.text(i, a + 0.02, f"{a:.0%}", ha="center")
    ax5.tick_params(axis='x', rotation=15)

    # ─── Panel F: Token Usage ───
    ax6 = fig.add_subplot(gs[1, 2])
    tok_data = {c: [] for c in CONDITIONS}
    for entry in exp1:
        for cond in CONDITIONS:
            tok_data[cond].append(entry["text_analysis"][cond]["total_tokens"])
    means = [np.mean(tok_data[c]) for c in CONDITIONS]
    sems = [np.std(tok_data[c]) / np.sqrt(len(tok_data[c])) for c in CONDITIONS]
    ax6.bar([CONDITION_LABELS[c] for c in CONDITIONS], means, yerr=sems,
            color=[COLORS[c] for c in CONDITIONS], capsize=5, edgecolor="black", linewidth=0.5)
    ax6.set_title("F) Token Usage (Open-Ended)", fontweight="bold")
    ax6.set_ylabel("Completion Tokens")
    ax6.tick_params(axis='x', rotation=15)

    # ─── Panel G-I: Judge Win Rates ───
    judgments = load_results("experiment1_judgments.json")
    comparisons_to_plot = [
        ("standard", "inter_sentence_cot", "G"),
        ("cot", "inter_sentence_cot", "H"),
        ("standard", "cot", "I"),
    ]
    for idx, (ca, cb, panel_label) in enumerate(comparisons_to_plot):
        ax = fig.add_subplot(gs[2, idx])
        criteria = ["coherence", "depth", "accuracy"]
        a_wins, b_wins, ties = [], [], []
        for crit in criteria:
            aw = bw = tw = 0
            for entry in judgments:
                for comp in entry["comparisons"]:
                    if comp["condition_a"] == ca and comp["condition_b"] == cb and comp["criteria"] == crit:
                        if comp["winner"] == ca:
                            aw += 1
                        elif comp["winner"] == cb:
                            bw += 1
                        else:
                            tw += 1
            total_j = aw + bw + tw
            a_wins.append(aw / total_j * 100 if total_j else 0)
            b_wins.append(bw / total_j * 100 if total_j else 0)
            ties.append(tw / total_j * 100 if total_j else 0)

        x = np.arange(len(criteria))
        w = 0.25
        ax.bar(x - w, a_wins, w, label=CONDITION_LABELS[ca], color=COLORS[ca])
        ax.bar(x, b_wins, w, label=CONDITION_LABELS[cb], color=COLORS[cb])
        ax.bar(x + w, ties, w, label="Tie", color="#CCCCCC")
        ax.set_xticks(x)
        ax.set_xticklabels([c.capitalize() for c in criteria])
        ax.set_ylabel("Win Rate (%)")
        ax.set_title(f"{panel_label}) {CONDITION_LABELS[ca]} vs {CONDITION_LABELS[cb]}", fontweight="bold")
        ax.legend(fontsize=7)

    plt.suptitle("An LLM That's Careful With Its Words: Summary Results",
                 fontsize=16, fontweight="bold", y=1.01)
    plt.savefig(PLOTS_DIR / "summary_figure.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved summary figure to {PLOTS_DIR / 'summary_figure.png'}")


def main():
    print("Starting analysis...")
    exp1_analysis = analyze_experiment1()
    exp2_analysis = analyze_experiment2()
    exp3_analysis = analyze_experiment3()
    exp4_summary, effect_sizes = analyze_experiment4_text_properties()
    create_summary_figure()

    # Save all analysis results
    all_results = {
        "experiment1": exp1_analysis,
        "experiment2": exp2_analysis,
        "experiment3": exp3_analysis,
        "experiment4": {"summary": exp4_summary, "effect_sizes": effect_sizes},
    }
    out_path = RESULTS_DIR.parent / "analysis_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved all analysis results to {out_path}")


if __name__ == "__main__":
    main()
