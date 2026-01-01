#!/usr/bin/env python3
"""Experiment #2: capability-scaling of calibrated selective resolution
(post-hoc, exploratory; NOT part of ANALYSIS_PLAN.md's confirmatory tests).

Claim under test: on the no-metadata `semantic_resolvable` regime (n=150,
seed 3001) -- where the deterministic rule resolves nothing -- the LLM's
*calibration* (does it commit only when right?) is governed by model
capability. A frontier model should be a wrong-commit-free selective resolver;
weaker models that commit at all should do so with substantial error. This
turns an anecdote ("gpt-5.4 happens to work") into a trend across the model
ladder.

Uniform apples-to-apples: every model is the single-call ThreeWayLLMMerge with
the same seed/n/temperature/max_tokens the original runs used, so all calls hit
the existing content-hash cache (zero new API). A model whose calls are not
cached is skipped with a note rather than silently spending.

Metrics per model (abstention lens): commit rate, conditional-on-commit
accuracy, wrong-commit rate.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from branchmem.benchmark.downstream_tasks import generate_downstream_questions
from branchmem.benchmark.semantic_resolvable_generator import generate_semantic_resolvable_scenarios
from branchmem.eval.abstention_metrics import compute_abstention_metrics
from branchmem.evaluation.metrics import score_downstream
from branchmem.llm.base import build_backend
from branchmem.merge.three_way_llm import ThreeWayLLMMerge
from branchmem.utils.seeding import set_all_seeds

OUT = Path(__file__).resolve().parents[1] / "results" / "final"

# (label, capability_rank, backend config). Rank is a coarse ordering used only
# for plotting; cross-vendor comparability is discussed, not asserted as exact.
MODELS = [
    ("llama-3.1-8b", 1, {"backend": "openai_compatible", "model": "llama3.1:8b",
                         "base_url": os.environ.get("PROBE_BASE_URL") or "http://localhost:11434/v1"}),
    ("gpt-5.4-nano", 2, {"backend": "openai_compatible", "model": "gpt-5.4-nano"}),
    ("claude-haiku-4-5", 3, {"backend": "anthropic", "model": "claude-haiku-4-5-20251001"}),
    ("gpt-5.4", 4, {"backend": "openai_compatible", "model": "gpt-5.4"}),
]
COMMON = {"cache_dir": "llm_cache", "temperature": 0.0, "max_tokens": 8000}


def decomposition_for(cfg):
    set_all_seeds(3001)
    scenarios = generate_semantic_resolvable_scenarios(150, seed=3001)
    backend = build_backend({**cfg, **COMMON})
    strat = ThreeWayLLMMerge(backend=backend)
    detail = []
    for s in scenarios:
        qs = generate_downstream_questions(s)
        res = strat.merge(s.ancestor, s.branch_a, s.branch_b, use_cache=True)
        detail.extend(score_downstream(qs, res).detail)
    cats = tuple(sorted({d["category"] for d in detail}))
    m = compute_abstention_metrics(detail, categories_in_scope=cats)
    acc = sum(d["correct"] for d in detail) / len(detail)
    return {
        "n_keys": m.n_questions, "accuracy": acc,
        "commit_rate": m.commit_rate,
        "conditional_on_commit_accuracy": m.conditional_on_commit_accuracy,
        "wrong_commit_rate": m.wrong_commit_rate,
        "n_committed": m.n_committed, "n_wrong_committed": m.n_wrong_committed,
    }


def main() -> None:
    results = {}
    for label, rank, cfg in MODELS:
        try:
            d = decomposition_for(cfg)
            d["capability_rank"] = rank
            results[label] = d
            print(f"{label:18} acc={d['accuracy']:.3f}  commit={d['commit_rate']:.3f}  "
                  f"cond_acc={d['conditional_on_commit_accuracy']}  wrong_commit={d['wrong_commit_rate']:.3f}  "
                  f"({d['n_committed']} commits, {d['n_wrong_committed']} wrong)")
        except Exception as exc:  # cache miss (needs a live key) or backend unavailable
            results[label] = {"skipped": True, "reason": type(exc).__name__ + ": " + str(exc)[:120], "capability_rank": rank}
            print(f"{label:18} SKIPPED ({type(exc).__name__}: {str(exc)[:80]})")

    payload = {
        "note": (
            "Experiment #2 (post-hoc, exploratory; not in ANALYSIS_PLAN.md). Single-call "
            "ThreeWayLLMMerge abstention decomposition on semantic_resolvable (seed 3001, n=150) "
            "across the model ladder. Uniform temp=0.0/max_tokens=8000; all calls served from the "
            "content-hash cache. Capability governs CALIBRATION: only the frontier reaches a "
            "wrong-commit-free selective-resolution regime."
        ),
        "rule_baseline": {"accuracy": 0.0, "commit_rate": 0.0, "note": "equal confidence -> always flags"},
        "models": results,
    }
    (OUT / "capability_scaling.json").write_text(json.dumps(payload, indent=2))
    print("\nWrote", OUT / "capability_scaling.json")

    # Optional figure (skip cleanly if matplotlib unavailable).
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ok = [(lbl, d) for lbl, d in results.items() if not d.get("skipped")]
        ok.sort(key=lambda t: t[1]["capability_rank"])
        labels = [lbl for lbl, _ in ok]
        cond = [d["conditional_on_commit_accuracy"] or 0.0 for _, d in ok]
        wrong = [d["wrong_commit_rate"] for _, d in ok]
        commit = [d["commit_rate"] for _, d in ok]
        x = range(len(labels))
        fig, ax = plt.subplots(figsize=(6.2, 3.8))
        ax.plot(x, cond, "o-", label="conditional-on-commit accuracy", color="#2a7")
        ax.plot(x, wrong, "s--", label="wrong-commit rate", color="#c33")
        ax.plot(x, commit, "^:", label="commit rate", color="#579")
        ax.axhline(1.0, color="#2a7", lw=0.5, alpha=0.4)
        ax.set_xticks(list(x)); ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
        ax.set_ylim(-0.03, 1.05); ax.set_ylabel("rate")
        ax.set_title("Selective resolution on semantic_resolvable scales with capability", fontsize=9)
        ax.legend(fontsize=7, loc="center left")
        fig.tight_layout()
        fig.savefig(OUT / "capability_scaling.png", dpi=150)
        print("Wrote", OUT / "capability_scaling.png")
    except Exception as exc:
        print("Figure skipped:", type(exc).__name__, str(exc)[:80])


if __name__ == "__main__":
    main()
