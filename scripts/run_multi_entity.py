#!/usr/bin/env python3
"""Post-hoc, exploratory (see ANALYSIS_PLAN_ADDENDUM.md). NOT confirmatory.

Addresses two common concerns about the locked run at once: (i) small sample
(n=60) and (ii) single-entity-per-scenario (never tests cross-entity/cross-key
consistency). This runs a LARGER, MULTI-ENTITY replication of RQ1: 120 scenarios
of 3 entities each, with more conflicts per scenario, on a disjoint seed (7001),
with gpt-5.4-nano. If the RQ1 ranking (three-way and the deterministic rule beat
naive baselines; rule matches/beats the LLM) holds here, the locked finding is
not an artifact of n=60 or of single-entity scenarios.
"""
from __future__ import annotations

import json
from pathlib import Path

from branchmem.benchmark.downstream_tasks import generate_downstream_questions
from branchmem.benchmark.scenario_generator import ScenarioConfig, ScenarioGenerator
from branchmem.evaluation.metrics import score_downstream
from branchmem.evaluation.stats import paired_comparison
from branchmem.llm.base import build_backend
from branchmem.merge.confidence_rule import ConfidenceRuleMerge
from branchmem.merge.last_writer_wins import LastWriterWins
from branchmem.merge.naive_concat import NaiveConcat
from branchmem.merge.three_way_llm import ThreeWayLLMMerge
from branchmem.utils.seeding import set_all_seeds

SEED = 7001
N = 120
OUT = Path(__file__).resolve().parents[1] / "results" / "final"


def main() -> None:
    set_all_seeds(SEED)
    cfg = ScenarioConfig(n_entities=3, n_ancestor_facts=8, n_orthogonal_a=3, n_orthogonal_b=3,
                         n_resolvable=4, n_ambiguous=2)
    gen = ScenarioGenerator()
    scenarios = gen.generate(N, cfg, seed=SEED)
    backend = build_backend({"backend": "openai_compatible", "model": "gpt-5.4-nano",
                             "cache_dir": "llm_cache", "temperature": 0.0, "max_tokens": 2048})
    strategies = {
        "three_way_llm": ThreeWayLLMMerge(backend=backend),
        "confidence_rule": ConfidenceRuleMerge(),
        "last_writer_wins": LastWriterWins(),
        "naive_concat": NaiveConcat(),
    }
    acc = {n: [] for n in strategies}
    # True cross-key contamination check: for each CONFLICT key, a committed
    # (non-flagged) three-way resolution must equal one of that key's own two
    # branch candidate values -- never a value belonging to a different key or
    # entity. Counts genuine fabrication/leakage, not untouched-ancestor facts.
    contamination = 0
    committed_conflict = 0
    for i, s in enumerate(scenarios):
        q = generate_downstream_questions(s)
        fl = {f.fact_id: f for f in s.branch_a.facts + s.branch_b.facts}
        candidates = {}  # (entity,predicate) -> {value_a, value_b}
        for p in s.conflict_pairs:
            fa, fb = fl[p.fact_a_id], fl[p.fact_b_id]
            candidates[(fa.entity, fa.predicate)] = {fa.value, fb.value}
        for n, strat in strategies.items():
            sc = strat.merge(s.ancestor, s.branch_a, s.branch_b)
            acc[n].append(score_downstream(q, sc).accuracy)
            if n == "three_way_llm":
                for rf in sc.resulting_facts:
                    key = (rf.fact.entity, rf.fact.predicate)
                    if key in candidates and rf.resolution.value != "flagged_unresolved":
                        committed_conflict += 1
                        if rf.fact.value not in candidates[key]:
                            contamination += 1
        if (i + 1) % 20 == 0:
            m = lambda v: sum(v) / len(v)
            print(f"[{i+1}/{N}] three_way={m(acc['three_way_llm']):.3f} rule={m(acc['confidence_rule']):.3f} "
                  f"lww={m(acc['last_writer_wins']):.3f} concat={m(acc['naive_concat']):.3f}", flush=True)

    mean = lambda v: sum(v) / len(v)
    result = {
        "note": ("post-hoc, exploratory larger + multi-entity RQ1 replication (3 entities/scenario, "
                 "disjoint seed 7001, gpt-5.4-nano). Tests whether the locked RQ1 ranking survives a "
                 "3x-larger, multi-entity construction. NOT part of ANALYSIS_PLAN.md confirmatory tests."),
        "n_scenarios": N, "n_entities": 3, "seed": SEED,
        "mean_accuracy": {n: round(mean(v), 4) for n, v in acc.items()},
        "committed_conflict_resolutions": committed_conflict, "cross_key_contamination": contamination,
        "comparisons": {},
    }
    for base in ("last_writer_wins", "naive_concat", "confidence_rule"):
        r = paired_comparison("three_way_llm", acc["three_way_llm"], base, acc[base], seed=SEED)
        result["comparisons"][f"three_way_vs_{base}"] = {
            "mean_diff": round(r.mean_diff, 4), "ci": [round(r.ci_low, 4), round(r.ci_high, 4)], "p": r.p_value,
        }
    (OUT / "multi_entity.json").write_text(json.dumps(result, indent=2))
    print("\n" + json.dumps(result["mean_accuracy"], indent=2))
    print("committed conflict resolutions:", committed_conflict, "| cross-key contamination:", contamination)
    print("Wrote", OUT / "multi_entity.json")


if __name__ == "__main__":
    main()
