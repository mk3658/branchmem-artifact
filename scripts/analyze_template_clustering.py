#!/usr/bin/env python3
"""Template-family-clustered inference (post-hoc, exploratory).

Conflicts generated from a small template pool are not independent, so a
per-conflict interval overstates precision for claims that should generalize to
*unseen conflict types*. This script quantifies that rather than only noting it:

  1. per-family accuracy and its spread;
  2. leave-one-template-family-out (LOFO) accuracy, i.e. how much the headline
     number depends on any single family;
  3. a cluster bootstrap that resamples whole FAMILIES (not conflicts), giving
     an interval for generalization to new families, next to the naive
     per-conflict bootstrap for contrast.

Zero new API: every merge call is served from the content-hash cache.
"""
from __future__ import annotations

import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

from branchmem.benchmark.downstream_tasks import generate_downstream_questions
from branchmem.benchmark.misleading_provenance_generator import (
    generate_misleading_provenance_scenarios,
)
from branchmem.benchmark.semantic_resolvable_generator import (
    generate_semantic_resolvable_scenarios,
)
from branchmem.evaluation.metrics import score_downstream
from branchmem.llm.base import build_backend
from branchmem.merge.three_way_llm import ThreeWayLLMMerge

OUT = Path(__file__).resolve().parents[1] / "results" / "final"
COMMON = {"cache_dir": "llm_cache", "temperature": 0.0, "max_tokens": 8000}


def per_item(scenarios, strat, category="resolvable"):
    """[(family, correct01)] over resolvable questions."""
    rows = []
    for s in scenarios:
        fam = s.metadata.get("template_predicate", "?")
        sc = score_downstream(generate_downstream_questions(s),
                              strat.merge(s.ancestor, s.branch_a, s.branch_b))
        for d in sc.detail:
            if d["category"] == category:
                rows.append((fam, int(d["correct"])))
    return rows


def cluster_stats(rows, n_boot=4000, seed=11):
    rng = random.Random(seed)
    vals = [v for _, v in rows]
    by_fam = defaultdict(list)
    for f, v in rows:
        by_fam[f].append(v)
    fams = sorted(by_fam)
    point = sum(vals) / len(vals)

    fam_acc = {f: sum(v) / len(v) for f, v in by_fam.items()}

    # leave-one-family-out
    lofo = {}
    for f in fams:
        kept = [v for g, v in rows if g != f]
        lofo[f] = sum(kept) / len(kept) if kept else float("nan")

    def q(a, p):
        a = sorted(a)
        return a[min(int(p * len(a)), len(a) - 1)]

    naive = []
    for _ in range(n_boot):
        samp = [vals[rng.randrange(len(vals))] for _ in vals]
        naive.append(sum(samp) / len(samp))

    clustered = []
    for _ in range(n_boot):
        pool = []
        for _ in range(len(fams)):
            pool.extend(by_fam[fams[rng.randrange(len(fams))]])
        clustered.append(sum(pool) / len(pool))

    return {
        "n_conflicts": len(vals), "n_families": len(fams), "point_accuracy": point,
        "per_family_accuracy": fam_acc,
        "family_accuracy_sd": statistics.pstdev(list(fam_acc.values())) if len(fam_acc) > 1 else 0.0,
        "family_accuracy_min": min(fam_acc.values()), "family_accuracy_max": max(fam_acc.values()),
        "lofo_min": min(lofo.values()), "lofo_max": max(lofo.values()),
        "lofo_most_influential_family": min(lofo, key=lambda f: abs(lofo[f] - point)) if lofo else None,
        "lofo_largest_swing_family": max(lofo, key=lambda f: abs(lofo[f] - point)) if lofo else None,
        "lofo_largest_swing": max(abs(v - point) for v in lofo.values()) if lofo else 0.0,
        "naive_ci95": [q(naive, 0.025), q(naive, 0.975)],
        "clustered_ci95": [q(clustered, 0.025), q(clustered, 0.975)],
        "naive_width": q(naive, 0.975) - q(naive, 0.025),
        "clustered_width": q(clustered, 0.975) - q(clustered, 0.025),
    }


def main() -> None:
    frontier = build_backend({"backend": "openai_compatible", "model": "gpt-5.4", **COMMON})
    strat = ThreeWayLLMMerge(backend=frontier)

    conds = {
        "semantic_resolvable_gpt54": generate_semantic_resolvable_scenarios(150, seed=3001),
        "misleading_provenance_gpt54": generate_misleading_provenance_scenarios(60, seed=5150),
    }

    out = {}
    for name, scen in conds.items():
        rows = per_item(scen, strat)
        st = cluster_stats(rows)
        out[name] = st
        print(f"\n=== {name} ===")
        print(f"  {st['n_conflicts']} conflicts from {st['n_families']} template families")
        print(f"  point accuracy      : {st['point_accuracy']:.3f}")
        print(f"  per-family accuracy : min {st['family_accuracy_min']:.3f}, "
              f"max {st['family_accuracy_max']:.3f}, SD {st['family_accuracy_sd']:.3f}")
        print(f"  LOFO range          : [{st['lofo_min']:.3f}, {st['lofo_max']:.3f}]  "
              f"largest swing {st['lofo_largest_swing']:.3f} (drop '{st['lofo_largest_swing_family']}')")
        print(f"  naive     95% CI    : [{st['naive_ci95'][0]:.3f}, {st['naive_ci95'][1]:.3f}]  width {st['naive_width']:.3f}")
        print(f"  CLUSTERED 95% CI    : [{st['clustered_ci95'][0]:.3f}, {st['clustered_ci95'][1]:.3f}]  width {st['clustered_width']:.3f}")
        if st["naive_width"] > 0:
            print(f"  width inflation     : {st['clustered_width']/st['naive_width']:.2f}x")

    payload = {
        "note": ("Post-hoc, exploratory (not in ANALYSIS_PLAN.md). Template-family-clustered "
                 "inference: per-family accuracy, leave-one-family-out, and a cluster bootstrap "
                 "that resamples whole template families (the correct unit for generalization to "
                 "unseen conflict types) against the naive per-conflict bootstrap. Zero new API."),
        "conditions": out,
    }
    (OUT / "template_clustering.json").write_text(json.dumps(payload, indent=2))
    print("\nWrote", OUT / "template_clustering.json")


if __name__ == "__main__":
    main()
