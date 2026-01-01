"""Standalone figure-generation script for the ACL paper (v6 redesign).

Pulls numbers only from committed real result files:
  - results/final/results.json                        (locked synthetic n=60 run)
  - results/final/robustness_second_model.json         (post-hoc gpt-5.4-mini check)
  - results/final/mab_secondary_analysis_expanded.json (post-hoc real-MAB check, n=50 cap)

v6 change: redesigned for print quality and statistical legibility rather
than just fitting a panel size. Concretely: (1) serif font family matching
the paper's Times body text, so figures don't visually clash with the
surrounding prose; (2) SD/SEM error bars added everywhere the underlying
per-scenario or per-row data supports them, since a bare mean without
variance is not something an ACL reader should have to take on faith; (3) a
light horizontal gridline so bar heights can be read without hunting for the
axis; (4) Figure 3 panel (a) redesigned from a grouped bar chart into a
slope/dumbbell chart (synthetic -> real MAB per strategy), which is a more
legible way to show "value inverts under this condition change" than
side-by-side bars, and reads correctly even in grayscale.

Writes PNGs into paper/ for \\includegraphics. This is the only copy: an
identical duplicate at paper/make_figures.py was removed, since two copies of a
figure generator is how a figure in the paper stops matching the script that
claims to produce it. Not part of the branchmem/ package; run manually (or
re-run to regenerate) with:
    source .venv/bin/activate && python scripts/make_figures.py
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Print-quality defaults, tuned for panels placed at ~half \columnwidth.
# Serif family matches the paper's Times body text so figures read as part
# of the document rather than a pasted-in screenshot.
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9.5,
    "axes.titlesize": 9.5,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    # No "tight" bbox here on purpose: a tight bbox recomputes the saved
    # image's extent from whatever the content (including legends placed
    # outside the axes) happens to occupy, which silently changes the
    # aspect ratio panel to panel. Every panel below reserves its own
    # margins with subplots_adjust/tight_layout instead, so the figsize
    # declared per panel is the actual saved aspect ratio -- required for
    # every subfigure to render as the same horizontal rectangle in print.
    "savefig.pad_inches": 0.02,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.axisbelow": True,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": "#DDDDDD",
    "grid.linewidth": 0.6,
    "errorbar.capsize": 2.5,
})

# Colorblind-friendly, print-safe palette (Okabe-Ito). The proposed method
# (ThreeWayLLM) always gets PALETTE[-1] (a distinct red) and a black outline
# across every figure, so a reader can find "our method" at a glance without
# re-reading each legend.
PALETTE = ["#0072B2", "#E69F00", "#009E73", "#56B4E9", "#009E73"]
PROPOSED_COLOR = "#D55E00"
MAB_COLOR = "#CC79A7"

# Every subfigure panel across every figure uses this same width and height
# (inches), so every panel is the same horizontal-rectangle shape (landscape,
# not the tall/portrait shape used before) and the same size, regardless of
# what content it holds. 1.9:1.2 is a clear landscape ratio (~1.6:1) while
# still tall enough for axis labels and a compact legend to stay legible.
SUBFIG_WIDTH = 1.9
SUBFIG_HEIGHT = 1.05
# fig_synthetic_vs_mab_a is the one exception: it has five categorical rows
# (one per strategy) that need vertical room to stay legible, so it gets a
# slightly taller variant -- still clearly landscape (1.9:1.25 ~= 1.5:1).
SUBFIG_HEIGHT_TALL = 1.25

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "final"
# Explicit, not "next to this script": the figures belong to the paper, and
# deriving the path from __file__ silently relocated them when the script moved.
OUT = ROOT / "paper"
OUT.mkdir(parents=True, exist_ok=True)

STRATEGIES = [
    "last_writer_wins",
    "naive_concat",
    "branch_discard_always_b",
    "branch_discard_fewer_updates",
    "three_way_llm",
]
STRATEGY_LABELS = {
    "last_writer_wins": "LWW",
    "naive_concat": "NaiveConcat",
    "branch_discard_always_b": "Discard(A)",
    "branch_discard_fewer_updates": "Discard(F)",
    "three_way_llm": "ThreeWayLLM",
}
# Shorter labels for the tighter horizontal-rectangle legends only (full
# names stay in captions/prose); keeps a 3-column legend row from
# overflowing the panel's fixed canvas width.
STRATEGY_LABELS_SHORT = {
    "last_writer_wins": "LWW",
    "naive_concat": "NConcat",
    "branch_discard_always_b": "Disc(A)",
    "branch_discard_fewer_updates": "Disc(F)",
    "three_way_llm": "3WayLLM",
}
STRATEGY_COLORS = {
    "last_writer_wins": PALETTE[0],
    "naive_concat": PALETTE[1],
    "branch_discard_always_b": PALETTE[2],
    "branch_discard_fewer_updates": PALETTE[3],
    "three_way_llm": PROPOSED_COLOR,
    "confidence_rule": "#444444",
}
STRATEGY_LABELS_SHORT_EXT = {
    "last_writer_wins": "LWW",
    "naive_concat": "NConcat",
    "branch_discard_always_b": "Disc(A)",
    "branch_discard_fewer_updates": "Disc(F)",
    "confidence_rule": "ConfRule",
    "three_way_llm": "3WayLLM",
}
# Figure 1 promotes ConfidenceRuleMerge (deterministic, zero-LLM-cost,
# post-hoc) into the main category-breakdown comparison alongside the
# original locked-run strategies, rather than showing it only in the
# ablation table -- it is the benchmark's strongest baseline.
CATEGORIES = ["orthogonal", "resolvable", "ambiguous"]
CATEGORY_LABELS = ["Orthog.", "Resolv.", "Ambig."]


def load(name):
    with open(RESULTS / name) as f:
        return json.load(f)


def sem(values: list[float]) -> float:
    """Standard error of the mean; 0 for n<2 rather than raising."""
    if len(values) < 2:
        return 0.0
    return statistics.stdev(values) / (len(values) ** 0.5)


# ---------------------------------------------------------------------------
# Figure 1, panel (a): category breakdown grouped bar chart, synthetic n=60,
# with SEM error bars so the reader can see spread, not just the mean.
# ---------------------------------------------------------------------------
def fig_category_breakdown_a():
    results = load("results.json")
    raw = {s: {c: [] for c in CATEGORIES} for s in STRATEGIES}
    for sc in results["per_scenario"]:
        cabs = sc["category_accuracy_by_strategy"]
        for s in STRATEGIES:
            for c in CATEGORIES:
                if c in cabs[s]:
                    raw[s][c].append(cabs[s][c])
    agg = {s: {c: statistics.mean(v) for c, v in cats.items()} for s, cats in raw.items()}
    err = {s: {c: sem(v) for c, v in cats.items()} for s, cats in raw.items()}

    # Promote ConfidenceRuleMerge (post-hoc, zero-LLM-cost, deterministic)
    # into the main comparison, positioned just before ThreeWayLLM since it
    # is the strategy that actually challenges it.
    extra = load("category_breakdown_extra_strategies.json")
    agg["confidence_rule"] = extra["mean_accuracy_by_category"]["confidence_rule"]
    plot_order = [s for s in STRATEGIES if s != "three_way_llm"] + ["confidence_rule", "three_way_llm"]

    x = np.arange(len(CATEGORIES))
    width = 0.14
    fig, ax = plt.subplots(figsize=(SUBFIG_WIDTH, SUBFIG_HEIGHT))
    for i, s in enumerate(plot_order):
        vals = [agg[s][c] for c in CATEGORIES]
        errs = [err[s][c] for c in CATEGORIES] if s in err else [0, 0, 0]
        offset = (i - (len(plot_order) - 1) / 2) * width
        is_proposed = s == "three_way_llm"
        bars = ax.bar(
            x + offset, vals, width, yerr=errs, label=STRATEGY_LABELS_SHORT_EXT[s],
            color=STRATEGY_COLORS[s],
            edgecolor="black" if is_proposed else "none",
            linewidth=1.1 if is_proposed else 0,
            error_kw=dict(elinewidth=0.7, ecolor="#333333"),
        )
    ax.set_xticks(x)
    ax.set_xticklabels(CATEGORY_LABELS)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.12)
    ax.set_yticks([0, 0.5, 1.0])
    # Two-row legend below the plot; margins reserved explicitly (rather
    # than tight_layout/tight bbox) so the saved PNG keeps the fixed
    # landscape figsize instead of growing taller to fit the legend.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=3,
              frameon=False, handlelength=0.8, columnspacing=0.5,
              handletextpad=0.25, fontsize=6.0)
    fig.subplots_adjust(left=0.20, right=0.96, top=0.97, bottom=0.44)
    fig.savefig(OUT / "category_breakdown_a.png")
    plt.close(fig)
    return agg


# ---------------------------------------------------------------------------
# Figure 1, panel (b): per-category gap, ThreeWayLLM minus best baseline
# ---------------------------------------------------------------------------
def fig_category_breakdown_b(agg):
    baselines = [s for s in STRATEGIES if s != "three_way_llm"]
    gaps, mechanical = [], []
    for c in CATEGORIES:
        best_baseline = max(agg[s][c] for s in baselines)
        gaps.append(agg["three_way_llm"][c] - best_baseline)
        # Ambiguous is mechanically guaranteed: no baseline can emit
        # flagged_unresolved at all, so every baseline scores exactly 0 there
        # by construction, not by being outreasoned.
        mechanical.append(c == "ambiguous")

    fig, ax = plt.subplots(figsize=(SUBFIG_WIDTH, SUBFIG_HEIGHT))
    colors = ["#B0B0B0" if m else PROPOSED_COLOR for m in mechanical]
    bars = ax.bar(CATEGORY_LABELS, gaps, color=colors,
                   edgecolor="black", linewidth=0.7, width=0.55)
    for b, g, m in zip(bars, gaps, mechanical):
        label = f"{g:+.2f}" + ("*" if m else "")
        va = "bottom" if g >= 0 else "top"
        offset = 0.03 if g >= 0 else -0.03
        ax.text(b.get_x() + b.get_width() / 2, g + offset, label, ha="center",
                va=va, fontsize=7.5, fontweight="bold" if not m else "normal")
    ax.set_ylabel("Gap vs. baseline", fontsize=8)
    ax.set_ylim(-0.08, 1.05)
    ax.axhline(0, color="black", linewidth=0.9)
    ax.grid(axis="y", alpha=0.5)
    fig.subplots_adjust(left=0.22, right=0.98, top=0.95, bottom=0.20)
    fig.savefig(OUT / "category_breakdown_b.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2, panel (a): robustness check, paired per-scenario scatter
# ---------------------------------------------------------------------------
def _robustness_data():
    results = load("results.json")
    robustness = load("robustness_second_model.json")
    cross_vendor = load("cross_vendor_rq1.json")
    nano_by_id = {
        sc["scenario_id"]: sc["accuracy_by_strategy"]["three_way_llm"]
        for sc in results["per_scenario"]
        if sc["divergence_span"] == 4.0
    }
    mini_by_id = {r["scenario_id"]: r["accuracy"] for r in robustness["per_scenario"]}
    haiku_by_id = {r["scenario_id"]: r["accuracy"] for r in cross_vendor["per_scenario"]}
    ids = [sid for sid in mini_by_id if sid in nano_by_id and sid in haiku_by_id]
    nano_vals = [nano_by_id[sid] for sid in ids]
    mini_vals = [mini_by_id[sid] for sid in ids]
    haiku_vals = [haiku_by_id[sid] for sid in ids]
    return robustness, nano_vals, mini_vals, haiku_vals


def fig_robustness_a(nano_vals, mini_vals, haiku_vals):
    fig, ax = plt.subplots(figsize=(SUBFIG_WIDTH, SUBFIG_HEIGHT))
    ax.plot([0.5, 1.02], [0.5, 1.02], "--", color="#999999", linewidth=1.0, zorder=1)
    # Jitter + count-weighted marker size so overlapping identical points
    # (common at 1.0/1.0) are still visible as denser rather than vanishing.
    rng = np.random.RandomState(0)
    jitter = rng.uniform(-0.008, 0.008, size=(len(nano_vals), 2))
    ax.scatter(np.array(nano_vals) + jitter[:, 0], np.array(mini_vals) + jitter[:, 1],
               alpha=0.75, s=24, color=PROPOSED_COLOR, edgecolor="white",
               linewidth=0.6, zorder=2, label="mini (same vendor)")
    jitter2 = rng.uniform(-0.008, 0.008, size=(len(nano_vals), 2))
    ax.scatter(np.array(nano_vals) + jitter2[:, 0], np.array(haiku_vals) + jitter2[:, 1],
               alpha=0.75, s=24, color=PALETTE[2], edgecolor="white",
               linewidth=0.6, zorder=2, marker="^", label="haiku (2nd vendor)")
    ax.set_xlim(0.55, 1.03)
    ax.set_ylim(0.55, 1.03)
    ax.set_xticks([0.6, 0.8, 1.0])
    ax.set_yticks([0.6, 0.8, 1.0])
    ax.set_xlabel("nano accuracy")
    ax.set_ylabel("other model acc.")
    ax.legend(loc="lower right", fontsize=5.5, frameon=False, handletextpad=0.3,
              borderaxespad=0.1)
    # No equal-aspect constraint here: a strict equal aspect on a landscape
    # canvas would just shrink the plot into a centered square and waste the
    # extra width, so the scatter fills the full horizontal-rectangle panel.
    fig.subplots_adjust(left=0.22, right=0.97, top=0.96, bottom=0.32)
    fig.savefig(OUT / "robustness_a.png")
    plt.close(fig)


def fig_robustness_b(robustness, nano_vals, mini_vals, haiku_vals):
    fig, ax = plt.subplots(figsize=(SUBFIG_WIDTH, SUBFIG_HEIGHT))
    means = [statistics.mean(nano_vals), statistics.mean(mini_vals), statistics.mean(haiku_vals)]
    errs = [sem(nano_vals), sem(mini_vals), sem(haiku_vals)]
    bars = ax.bar(
        ["nano\n(locked)", "mini\n(post-hoc)", "haiku\n(2nd vendor)"], means, yerr=errs,
        color=[PALETTE[0], PROPOSED_COLOR, PALETTE[2]], width=0.55,
        edgecolor="black", linewidth=0.7,
        error_kw=dict(elinewidth=0.8, ecolor="#333333"),
    )
    ax.set_ylim(0, 1.25)
    ax.set_ylabel("Mean accuracy")
    for b, m in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, m + 0.09, f"{m:.3f}",
                ha="center", fontsize=7.5, fontweight="bold")
    ax.tick_params(axis="x", labelsize=6.5)
    fig.subplots_adjust(left=0.22, right=0.97, top=0.96, bottom=0.30)
    fig.savefig(OUT / "robustness_b.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3, panel (a): synthetic vs. MAB, redesigned as a slope/dumbbell
# chart -- one line per strategy from its synthetic accuracy to its real-MAB
# accuracy. This reads the "does the ranking survive the condition change"
# story directly from the line directions, which a grouped bar chart forces
# the reader to reconstruct by eye; it also degrades gracefully in grayscale.
# ---------------------------------------------------------------------------
def _synth_mab_data():
    """The two runs do not share a strategy set, so the slope chart pairs only
    what both measured.

    The synthetic run carries `branch_discard_fewer_updates`; the reconciled
    MAB run (seed 2026, 270 conflicts -- addendum A12) carries `confidence_rule`
    in its place. Indexing one by the other's keys is what made this script die
    at this figure and never reach the two abstention figures the paper
    actually includes. A strategy measured on one side only cannot be drawn as
    a line between two sides, so it is dropped here rather than faked.
    """
    results = load("results.json")
    mab = load("mab_secondary_analysis_expanded.json")
    mab_overall = mab["mean_accuracy_by_strategy"]
    synth_by_strategy = {
        s: statistics.mean(sc["accuracy_by_strategy"][s] for sc in results["per_scenario"])
        for s in STRATEGIES
        if all(s in sc["accuracy_by_strategy"] for sc in results["per_scenario"])
    }
    shared = [s for s in STRATEGIES if s in synth_by_strategy and s in mab_overall]
    assert "three_way_llm" in shared, (
        "the proposed strategy is missing from one of the two runs; the "
        "synthetic-vs-real comparison has no subject")
    return {s: synth_by_strategy[s] for s in shared}, {s: mab_overall[s] for s in shared}


def fig_synthetic_vs_mab_a(synth_overall, mab_overall):
    fig, ax = plt.subplots(figsize=(SUBFIG_WIDTH, SUBFIG_HEIGHT_TALL))
    order = sorted(synth_overall, key=lambda s: synth_overall[s])
    y = np.arange(len(order))

    for i, s in enumerate(order):
        sv, mv = synth_overall[s], mab_overall[s]
        is_proposed = s == "three_way_llm"
        line_color = PROPOSED_COLOR if is_proposed else "#999999"
        ax.plot([sv, mv], [i, i], color=line_color,
                linewidth=2.0 if is_proposed else 1.1, zorder=2,
                solid_capstyle="round")
        ax.scatter([sv], [i], color=PALETTE[0], s=32, zorder=3,
                   edgecolor="black" if is_proposed else "white", linewidth=0.7)
        ax.scatter([mv], [i], color=MAB_COLOR, s=32, zorder=3,
                   edgecolor="black" if is_proposed else "white", linewidth=0.7)

    ax.set_yticks(y)
    ax.set_yticklabels([STRATEGY_LABELS_SHORT[s] for s in order], fontsize=7.5)
    ax.set_xlabel("Mean accuracy", labelpad=2)
    ax.set_xlim(-0.03, 1.05)
    ax.set_xticks([0, 0.5, 1.0])
    ax.grid(axis="x", alpha=0.5)
    ax.grid(axis="y", visible=False)
    # Emphasize the one line that reverses direction.
    ax.text(0.02, len(order) - 1 + 0.55, "flips", fontsize=7, color=PROPOSED_COLOR,
            fontweight="bold", ha="left")
    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=PALETTE[0],
                   markersize=5, label="Synthetic"),
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=MAB_COLOR,
                   markersize=5, label="Real MAB"),
    ]
    ax.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, -0.48),
              ncol=2, frameon=False, handletextpad=0.3, columnspacing=1.2,
              fontsize=6.5)
    fig.subplots_adjust(left=0.31, right=0.97, top=0.85, bottom=0.42)
    fig.savefig(OUT / "synthetic_vs_mab_a.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3, panel (b): ThreeWayLLM isolated, the single number that inverts
# ---------------------------------------------------------------------------
def fig_synthetic_vs_mab_b(synth_overall, mab_overall):
    fig, ax = plt.subplots(figsize=(SUBFIG_WIDTH, SUBFIG_HEIGHT))
    vals = [synth_overall["three_way_llm"], mab_overall["three_way_llm"]]
    bars = ax.bar(["Synth.", "Real MAB"], vals,
                   color=[PALETTE[0], MAB_COLOR], width=0.5,
                   edgecolor="black", linewidth=0.8)
    ax.text(bars[0].get_x() + bars[0].get_width() / 2, vals[0] + 0.05, f"{vals[0]:.3f}",
            ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    ax.text(bars[1].get_x() + bars[1].get_width() / 2 + 0.32, vals[1] + 0.01, f"{vals[1]:.3f}",
            ha="left", va="bottom", fontsize=8.5, fontweight="bold")
    ax.annotate(
        "", xy=(1, vals[1] + 0.06), xytext=(0.32, vals[0] - 0.02),
        arrowprops=dict(arrowstyle="->", color=PROPOSED_COLOR, linewidth=1.4,
                        connectionstyle="arc3,rad=0.15"),
    )
    ax.text(0.5, 0.62, f"$-${vals[0]-vals[1]:.2f}", ha="center", color=PROPOSED_COLOR,
            fontsize=8.5, fontweight="bold", transform=ax.transAxes)
    ax.set_ylabel("3WayLLM acc.", fontsize=8)
    ax.set_ylim(0, 1.28)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_xlim(-0.5, 1.7)
    ax.grid(axis="y", alpha=0.5)
    fig.subplots_adjust(left=0.24, right=0.97, top=0.95, bottom=0.19)
    fig.savefig(OUT / "synthetic_vs_mab_b.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4, panel (a): commit rate vs. conditional-on-commit accuracy,
# synthetic vs. real MAB -- shows abstention behavior is consistent (high
# accuracy when committed) even though raw accuracy differs wildly.
# ---------------------------------------------------------------------------
def fig_abstention_coverage():
    report = load("abstention_metrics_report.json")
    synth = report["synthetic_resolvable_and_ambiguous"]
    mab = report["mab_expanded"]

    labels = ["Synthetic", "Real MAB"]
    commit = [synth["commit_rate"], mab["commit_rate"]]
    cond_acc = [synth["conditional_on_commit_accuracy"], mab["conditional_on_commit_accuracy"] or 0.0]

    x = np.arange(len(labels))
    width = 0.32
    fig, ax = plt.subplots(figsize=(SUBFIG_WIDTH, SUBFIG_HEIGHT))
    b1 = ax.bar(x - width / 2, commit, width, label="Commit rate", color=PALETTE[0],
                edgecolor="black", linewidth=0.6)
    b2 = ax.bar(x + width / 2, cond_acc, width, label="Cond. acc.", color=PROPOSED_COLOR,
                edgecolor="black", linewidth=0.6)
    for b, v in zip(b1, commit):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.03, f"{v:.2f}", ha="center", fontsize=7)
    for b, v, m in zip(b2, cond_acc, [True, mab["conditional_on_commit_accuracy"] is not None]):
        label = f"{v:.2f}" if m else "undef."
        ax.text(b.get_x() + b.get_width() / 2, v + 0.03, label, ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Rate")
    ax.set_ylim(0, 1.18)
    ax.set_yticks([0, 0.5, 1.0])
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=2,
              frameon=False, handlelength=0.9, fontsize=7)
    fig.subplots_adjust(left=0.24, right=0.97, top=0.96, bottom=0.32)
    fig.savefig(OUT / "abstention_coverage.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4, panel (b): expected utility under three wrong-commit cost
# regimes, synthetic vs. real MAB -- makes the safety argument for
# abstention concrete rather than abstract.
# ---------------------------------------------------------------------------
def fig_abstention_utility():
    report = load("abstention_metrics_report.json")
    synth = report["synthetic_resolvable_and_ambiguous"]["expected_utility"]
    mab = report["mab_expanded"]["expected_utility"]
    cost_labels = ["cost_-2", "cost_-5", "cost_-10"]
    x_labels = ["$-2$", "$-5$", "$-10$"]

    x = np.arange(len(cost_labels))
    width = 0.32
    fig, ax = plt.subplots(figsize=(SUBFIG_WIDTH, SUBFIG_HEIGHT))
    synth_vals = [synth[c] for c in cost_labels]
    mab_vals = [mab[c] for c in cost_labels]
    ax.bar(x - width / 2, synth_vals, width, label="Synthetic", color=PALETTE[0],
           edgecolor="black", linewidth=0.6)
    ax.bar(x + width / 2, mab_vals, width, label="Real MAB", color=MAB_COLOR,
           edgecolor="black", linewidth=0.6)
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("Wrong-commit cost", fontsize=8)
    ax.set_ylabel("Expected utility", fontsize=8)
    ax.set_ylim(-1.15, 0.45)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.28), ncol=2,
              frameon=False, fontsize=6.5, handlelength=0.9, columnspacing=1.0)
    fig.subplots_adjust(left=0.30, right=0.97, top=0.82, bottom=0.34)
    fig.savefig(OUT / "abstention_utility.png")
    plt.close(fig)


def fig_noise_sensitivity():
    data = load("noise_sensitivity.json")["per_noise"]
    noises = [row["noise"] for row in data]
    series = {
        "three_way_llm": (PROPOSED_COLOR, "o", "ThreeWayLLM"),
        "confidence_rule": (PALETTE[2], "s", "ConfidenceRule"),
        "last_writer_wins": (PALETTE[0], "^", "LWW (naive)"),
    }
    fig, ax = plt.subplots(figsize=(2.6, 1.7))
    for name, (color, marker, label) in series.items():
        ys = [row["resolvable_mean_accuracy"][name] for row in data]
        ax.plot(noises, ys, marker=marker, color=color, linewidth=1.3, markersize=4,
                label=label, markeredgecolor="white", markeredgewidth=0.5)
    ax.axhline(0.5, ls=":", color="#999999", linewidth=0.9)
    ax.axvline(0.15, ls="--", color="#bbbbbb", linewidth=0.9)
    ax.text(0.155, 0.28, "locked", fontsize=6, color="#888888", rotation=90, va="bottom")
    ax.set_xlabel("source-signal noise", fontsize=8)
    ax.set_ylabel("resolvable accuracy", fontsize=8)
    ax.set_ylim(0.2, 1.05)
    ax.set_xticks([0.0, 0.15, 0.3, 0.5])
    ax.legend(loc="lower left", fontsize=6.0, frameon=False)
    fig.subplots_adjust(left=0.16, right=0.97, top=0.96, bottom=0.20)
    fig.savefig(OUT / "noise_sensitivity.png")
    plt.close(fig)


def fig_signal_coverage():
    data = load("signal_coverage.json")["per_coverage"]
    covs = [r["coverage"] for r in data]
    acc = [r["three_way_llm"]["resolvable_accuracy"] for r in data]
    commit = [r["three_way_llm"]["commit_rate"] for r in data]
    rule = [r["confidence_rule"]["resolvable_accuracy"] for r in data]
    fig, ax = plt.subplots(figsize=(2.7, 1.7))
    ax.plot(covs, acc, marker="o", color=PROPOSED_COLOR, linewidth=1.3, markersize=4,
            label="3way accuracy", markeredgecolor="white", markeredgewidth=0.5)
    ax.plot(covs, rule, marker="s", color=PALETTE[2], linewidth=1.1, markersize=3.5,
            label="rule accuracy", markeredgecolor="white", markeredgewidth=0.5)
    ax.plot(covs, commit, marker="^", color=PALETTE[0], linewidth=1.1, markersize=3.5,
            ls="--", label="3way commit rate", markeredgecolor="white", markeredgewidth=0.5)
    ax.plot([0, 1], [0, 1], ":", color="#cccccc", linewidth=0.9, zorder=0)
    ax.set_xlabel("signal coverage (frac. conflicts w/ signal)", fontsize=7)
    ax.set_ylabel("resolvable acc. / commit", fontsize=7)
    ax.set_ylim(-0.03, 1.05)
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.legend(loc="upper left", fontsize=5.8, frameon=False)
    fig.subplots_adjust(left=0.15, right=0.97, top=0.96, bottom=0.22)
    fig.savefig(OUT / "signal_coverage.png")
    plt.close(fig)


def fig_risk_coverage():
    pts = load("risk_coverage_frontier.json")["points"]
    labels = {
        "last_writer_wins": ("LWW", PALETTE[0]),
        "naive_concat": ("NaiveConcat", PALETTE[1]),
        "branch_discard_always_b": ("Discard", PALETTE[3]),
        "confidence_rule": ("ConfRule", PALETTE[2]),
        "three_way_llm": ("ThreeWayLLM", PROPOSED_COLOR),
    }
    fig, ax = plt.subplots(figsize=(2.6, 1.7))
    for name, (lab, color) in labels.items():
        p = pts[name]
        if p["risk"] is None:
            continue
        ax.scatter([p["coverage"]], [p["risk"]], s=42, color=color, edgecolor="black",
                   linewidth=0.7, zorder=3, label=lab)
    ax.set_xlabel("coverage (commit rate)", fontsize=8)
    ax.set_ylabel("risk (1 - cond. acc.)", fontsize=8)
    ax.set_xlim(0.4, 1.05)
    ax.set_ylim(-0.02, 0.78)
    ax.legend(loc="center left", fontsize=6.0, frameon=False, handletextpad=0.1)
    fig.subplots_adjust(left=0.17, right=0.97, top=0.96, bottom=0.20)
    fig.savefig(OUT / "risk_coverage.png")
    plt.close(fig)


if __name__ == "__main__":
    agg = fig_category_breakdown_a()
    fig_category_breakdown_b(agg)
    fig_noise_sensitivity()
    fig_risk_coverage()
    fig_signal_coverage()

    robustness, nano_vals, mini_vals, haiku_vals = _robustness_data()
    fig_robustness_a(nano_vals, mini_vals, haiku_vals)
    fig_robustness_b(robustness, nano_vals, mini_vals, haiku_vals)

    synth_overall, mab_overall = _synth_mab_data()
    fig_synthetic_vs_mab_a(synth_overall, mab_overall)
    fig_synthetic_vs_mab_b(synth_overall, mab_overall)

    fig_abstention_coverage()
    fig_abstention_utility()

    print("Wrote:")
    for p in [
        "category_breakdown_a.png", "category_breakdown_b.png",
        "robustness_a.png", "robustness_b.png",
        "synthetic_vs_mab_a.png", "synthetic_vs_mab_b.png",
        "abstention_coverage.png", "abstention_utility.png",
        "noise_sensitivity.png", "risk_coverage.png",
    ]:
        print(" ", OUT / p)
