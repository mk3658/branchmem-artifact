#!/usr/bin/env python3
"""Experiment #1: HybridMerge dominance (post-hoc, exploratory; NOT part of
ANALYSIS_PLAN.md's confirmatory tests).

Question. The paper's two central results pull apart: where a provenance
signal exists the deterministic ConfidenceRuleMerge matches/beats the LLM;
where it does not (semantic_resolvable) only a capable LLM resolves anything,
and does so without wrong commits. Does a single *deployable* policy --
ConfidenceRuleMerge with escalation-on-abstention to a frontier
ThreeWayLLMMerge (branchmem/merge/hybrid.py) -- dominate both parents across
the union of regimes, or does escalating the rule's abstentions backfire on
genuinely ambiguous conflicts the rule correctly flagged?

Two evaluation regions, both reusing the frontier gpt-5.4 content-hash cache
(zero new API when those calls are already cached):
  (A) the RQ1 confirmatory scenarios (seed 2026, n=60): mixed orthogonal /
      resolvable / ambiguous, i.e. the signal-present + must-abstain regimes;
  (B) semantic_resolvable (seed 3001, n=150): the no-metadata regime.

Reports, per strategy and region: category accuracy, and the abstention
decomposition (commit rate, conditional-on-commit accuracy, wrong-commit
rate). The dominance claim is checked, not assumed: we print, per category,
whether Hybrid >= max(rule, LLM).
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from branchmem.benchmark.downstream_tasks import generate_downstream_questions
from branchmem.benchmark.scenario_generator import ScenarioConfig, ScenarioGenerator
from branchmem.benchmark.semantic_resolvable_generator import generate_semantic_resolvable_scenarios
from branchmem.eval.abstention_metrics import compute_abstention_metrics
from branchmem.evaluation.metrics import score_downstream
from branchmem.llm.base import build_backend
from branchmem.merge.confidence_rule import ConfidenceRuleMerge
from branchmem.merge.hybrid import HybridMerge
from branchmem.merge.three_way_llm import ThreeWayLLMMerge
from branchmem.utils.seeding import set_all_seeds

MODEL = "gpt-5.4"
OUT = Path(__file__).resolve().parents[1] / "results" / "final"


def locked_scenarios(n_scenarios=60, seed=2026, spans=(4.0, 10.0, 20.0)):
    """Reproduce the LOCKED confirmatory construction exactly: n/len(spans)
    scenarios per divergence span, each span using sub-seed `seed + i`
    (branchmem/evaluation/runner.py). Using a single span here instead would
    silently produce a different scenario set and non-comparable numbers."""
    gen = ScenarioGenerator()
    n_per = n_scenarios // len(spans)
    rem = n_scenarios - n_per * len(spans)
    out = []
    for i, span in enumerate(spans):
        k = n_per + (1 if i < rem else 0)
        out.extend(gen.generate(k, ScenarioConfig(divergence_span=span), seed=seed + i))
    return out


def _pool(scenarios, strategies):
    """Run each strategy over the scenarios; return pooled per-question detail
    lists keyed by strategy name."""
    detail = defaultdict(list)
    for s in scenarios:
        qs = generate_downstream_questions(s)
        for name, strat in strategies.items():
            res = strat.merge(s.ancestor, s.branch_a, s.branch_b)
            detail[name].extend(score_downstream(qs, res).detail)
    return detail


def _accuracy_by_category(detail_rows):
    tot = defaultdict(int)
    cor = defaultdict(int)
    for d in detail_rows:
        tot[d["category"]] += 1
        cor[d["category"]] += int(d["correct"])
    return {c: cor[c] / tot[c] for c in sorted(tot)}, dict(tot)


def _region(scenarios, strategies, scope):
    detail = _pool(scenarios, strategies)
    out = {}
    for name, rows in detail.items():
        acc, totals = _accuracy_by_category(rows)
        overall = sum(d["correct"] for d in rows) / len(rows)
        m = compute_abstention_metrics(rows, categories_in_scope=scope)
        out[name] = {
            "overall_accuracy": overall,
            "accuracy_by_category": acc,
            "n_by_category": totals,
            "commit_rate": m.commit_rate,
            "conditional_on_commit_accuracy": m.conditional_on_commit_accuracy,
            "wrong_commit_rate": m.wrong_commit_rate,
            "n_committed": m.n_committed,
            "n_wrong_committed": m.n_wrong_committed,
        }
    return out


def _dominance(region, categories):
    """Per category: is hybrid >= max(rule, llm)? Return a checklist."""
    checks = {}
    for c in categories:
        r = region["confidence_rule"]["accuracy_by_category"].get(c)
        l = region["three_way_llm"]["accuracy_by_category"].get(c)
        h = region["hybrid"]["accuracy_by_category"].get(c)
        if h is None:
            continue
        best_parent = max(v for v in (r, l) if v is not None)
        checks[c] = {"rule": r, "llm": l, "hybrid": h,
                     "hybrid_ge_best_parent": h >= best_parent - 1e-9,
                     "hybrid_strictly_beats_a_parent": (r is not None and h > r + 1e-9) or (l is not None and h > l + 1e-9)}
    return checks


def main() -> None:
    backend = build_backend({"backend": "openai_compatible", "model": MODEL, "cache_dir": "llm_cache",
                             "temperature": 0.0, "max_tokens": 8000})
    strategies = {
        "confidence_rule": ConfidenceRuleMerge(),
        "three_way_llm": ThreeWayLLMMerge(backend=backend),
        "hybrid": HybridMerge(backend=backend),
    }

    # Region A: RQ1 confirmatory scenarios. The generator sub-seeds per scenario
    # Uses the LOCKED mixed-divergence-span construction (20 each at spans
    # 4.0/10.0/20.0, sub-seeds 2026/2027/2028) so numbers are directly
    # comparable to the confirmatory tables.
    set_all_seeds(2026)
    rq1_scen = locked_scenarios(60, seed=2026)
    region_a = _region(rq1_scen, strategies, scope=("resolvable", "ambiguous"))

    # Region B: semantic_resolvable (no-metadata regime, n=150).
    set_all_seeds(3001)
    sem_scen = generate_semantic_resolvable_scenarios(150, seed=3001)
    sem_cats = sorted({q.category for s in sem_scen for q in generate_downstream_questions(s)})
    region_b = _region(sem_scen, strategies, scope=tuple(sem_cats))

    dom_a = _dominance(region_a, [c for c in ("resolvable", "ambiguous") if c in region_a["hybrid"]["accuracy_by_category"]])
    dom_b = _dominance(region_b, sem_cats)

    result = {
        "note": (
            "Experiment #1 (post-hoc, exploratory; not in ANALYSIS_PLAN.md). HybridMerge = "
            "ConfidenceRuleMerge with escalation-on-abstention to frontier ThreeWayLLMMerge "
            "(gpt-5.4). Region A = RQ1 confirmatory scenarios (seed 2026, n=60); "
            "Region B = semantic_resolvable (seed 3001, n=150). Reuses the content-hash "
            "cache; zero new API when the frontier calls are already cached."
        ),
        "model": MODEL,
        "region_a_rq1_confirmatory": {"n_scenarios": len(rq1_scen), "by_strategy": region_a, "dominance": dom_a},
        "region_b_semantic_resolvable": {"n_scenarios": len(sem_scen), "categories": sem_cats,
                                          "by_strategy": region_b, "dominance": dom_b},
    }
    (OUT / "hybrid_merge.json").write_text(json.dumps(result, indent=2))

    # Console summary
    def show(title, region, dom):
        print(f"\n=== {title} ===")
        for name in ("confidence_rule", "three_way_llm", "hybrid"):
            r = region[name]
            print(f"  {name:16} overall={r['overall_accuracy']:.3f}  "
                  f"commit={r['commit_rate']:.3f}  wrong_commit={r['wrong_commit_rate']:.3f}  "
                  f"cond_acc={r['conditional_on_commit_accuracy']}")
            print(f"                   by_cat={ {k: round(v,3) for k,v in r['accuracy_by_category'].items()} }")
        print("  dominance (hybrid >= best parent?):")
        for c, ch in dom.items():
            print(f"    {c:20} rule={ch['rule']}  llm={ch['llm']}  hybrid={ch['hybrid']}  "
                  f"ge_best={ch['hybrid_ge_best_parent']}  strictly_beats_a_parent={ch['hybrid_strictly_beats_a_parent']}")

    show("Region A: RQ1 confirmatory (n=60)", region_a, dom_a)
    show("Region B: semantic_resolvable (n=150)", region_b, dom_b)
    print("\nWrote", OUT / "hybrid_merge.json")


if __name__ == "__main__":
    main()
