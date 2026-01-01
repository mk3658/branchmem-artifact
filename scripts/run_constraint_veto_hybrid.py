#!/usr/bin/env python3
"""Semantic-then-provenance hybrid vs its two parents (post-hoc, exploratory).

The NLI gate survives inverted provenance and is weak when provenance is the
only signal; the confidence rule is the mirror image. This script measures
whether checking for an explicit ancestor contradiction BEFORE applying the
provenance decision beats either parent on the coverage/risk trade-off, and
what it costs.

Evaluated across all three provenance regimes on identical items, plus the
wall-clock cost of a merge decision for each policy. The NLI models run locally
on CPU; the LLM numbers come from cache, so their latency is quoted from the
recorded cost profile rather than re-measured here.

Zero new API: every LLM merge is served from the content-hash cache.

Usage: OPENAI_API_KEY=cache-only-sentinel PYTHONPATH=. python scripts/run_constraint_veto_hybrid.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from branchmem.benchmark.downstream_tasks import generate_downstream_questions
from branchmem.benchmark.misleading_provenance_generator import (
    MisleadingProvenanceConfig,
    generate_misleading_provenance_scenarios,
)
from branchmem.benchmark.semantic_resolvable_generator import (
    generate_semantic_resolvable_scenarios,
)
from branchmem.conflict.nli_detector import NLIConflictDetector
from branchmem.eval.abstention_metrics import compute_abstention_metrics
from branchmem.evaluation.metrics import score_downstream
from branchmem.llm.base import build_backend
from branchmem.memory.schemas import MergeResult, Resolution, ResolvedFact
from branchmem.merge.base import MergeStrategy, current_facts_by_key, find_collisions
from branchmem.merge.confidence_rule import ConfidenceRuleMerge
from branchmem.merge.constraint_veto import ConstraintVetoMerge
from branchmem.merge.three_way_llm import ThreeWayLLMMerge

OUT = Path(__file__).resolve().parents[1] / "results" / "final"
SEED = 5150
N = 60

_INFORMATIVE = MisleadingProvenanceConfig(
    misleading_source="inference",
    misleading_confidence=0.55,
    truthful_source="user",
    truthful_confidence=0.95,
)


class NLIGatedMerge(MergeStrategy):
    """Relative-contradiction gate: keep whichever value the ancestor
    contradicts less. Never reads source or confidence. Reproduced here from
    scripts/run_operational_baselines.py so both parents are in one place."""

    name = "nli_gated"

    def __init__(self, detector: NLIConflictDetector | None = None, margin: float = 0.10) -> None:
        self.detector = detector or NLIConflictDetector()
        self.margin = margin

    def merge(self, ancestor, branch_a, branch_b) -> MergeResult:
        anc = current_facts_by_key(ancestor)
        a_facts = current_facts_by_key(branch_a)
        b_facts = current_facts_by_key(branch_b)
        collisions = find_collisions(branch_a, branch_b)
        resolved = []
        for key in sorted((set(a_facts) | set(b_facts)) - set(collisions)):
            f = a_facts.get(key) or b_facts.get(key)
            resolved.append(ResolvedFact(fact=f, resolution=Resolution.KEPT, source_branch_ids=[f.branch_id]))
        for key, (fa, fb) in collisions.items():
            anc_fact = anc.get(key)
            if anc_fact is None:
                resolved.append(ResolvedFact(fact=fa, resolution=Resolution.FLAGGED_UNRESOLVED,
                                             source_branch_ids=[branch_a.branch_id, branch_b.branch_id],
                                             justification="no ancestor constraint"))
                continue
            prem = f"{anc_fact.entity} {anc_fact.value}."
            ca = self.detector._predict([(prem, f"{fa.entity} {fa.value}.")])[0]["contradiction"]
            cb = self.detector._predict([(prem, f"{fb.entity} {fb.value}.")])[0]["contradiction"]
            if abs(ca - cb) < self.margin:
                resolved.append(ResolvedFact(fact=fa, resolution=Resolution.FLAGGED_UNRESOLVED,
                                             source_branch_ids=[branch_a.branch_id, branch_b.branch_id],
                                             justification="contradiction scores too close"))
            else:
                win = fa if ca < cb else fb
                resolved.append(ResolvedFact(fact=win, resolution=Resolution.KEPT,
                                             source_branch_ids=[win.branch_id],
                                             justification="less contradicted by ancestor"))
        return MergeResult(strategy_name=self.name, resulting_facts=resolved)


def evaluate(scenarios, strat) -> dict:
    rows, t0 = [], time.perf_counter()
    for s in scenarios:
        qs = generate_downstream_questions(s)
        rows.extend(score_downstream(qs, strat.merge(s.ancestor, s.branch_a, s.branch_b)).detail)
    elapsed = time.perf_counter() - t0
    cats = tuple(sorted({d["category"] for d in rows}))
    m = compute_abstention_metrics(rows, categories_in_scope=cats)
    return {
        "accuracy": sum(d["correct"] for d in rows) / len(rows),
        "coverage": m.commit_rate,
        "conditional_risk": (
            None if m.n_committed == 0 else m.n_wrong_committed / m.n_committed
        ),
        "wrong_commit_rate": m.wrong_commit_rate,
        "n_committed": m.n_committed,
        "n_wrong_committed": m.n_wrong_committed,
        "n_flagged": m.n_flagged,
        "seconds_per_conflict": elapsed / max(1, len(rows)),
    }


def main() -> None:
    backend = build_backend(
        {
            "backend": "openai_compatible",
            "model": "gpt-5.4",
            "cache_dir": "llm_cache",
            "temperature": 0.0,
            "max_tokens": 8000,
        }
    )
    detector = NLIConflictDetector()

    regimes = {
        "informative": generate_misleading_provenance_scenarios(N, seed=SEED, config=_INFORMATIVE),
        "absent": generate_semantic_resolvable_scenarios(N, seed=3001),
        "misleading": generate_misleading_provenance_scenarios(N, seed=SEED),
    }

    results: dict[str, dict] = {}
    veto_stats: dict[str, dict] = {}
    for regime, scenarios in regimes.items():
        veto = ConstraintVetoMerge(detector=detector)
        policies = {
            "ConfidenceRule (provenance only)": ConfidenceRuleMerge(),
            "NliGated (semantics only)": NLIGatedMerge(detector=detector),
            "ConstraintVeto (semantics then provenance)": veto,
            "ThreeWayLLM (frontier)": ThreeWayLLMMerge(backend=backend),
        }
        results[regime] = {name: evaluate(scenarios, p) for name, p in policies.items()}
        veto_stats[regime] = {
            "n_vetoes": veto.n_vetoes,
            "n_provenance": veto.n_provenance,
            "n_abstained": veto.n_abstained,
        }
        print(f"\n=== {regime} (n={N}) ===")
        print(f"{'policy':44} {'acc':>6} {'cov':>6} {'risk':>7} {'s/conf':>8}")
        for name, r in results[regime].items():
            cr = "   n/a" if r["conditional_risk"] is None else f"{r['conditional_risk']:6.3f}"
            print(
                f"{name:44} {r['accuracy']:6.3f} {r['coverage']:6.3f} {cr} "
                f"{r['seconds_per_conflict']:8.4f}"
            )
        v = veto_stats[regime]
        print(f"  veto fired {v['n_vetoes']}, provenance decided {v['n_provenance']}, "
              f"abstained {v['n_abstained']}")

    payload = {
        "note": (
            "Post-hoc, exploratory (not in ANALYSIS_PLAN.md). Semantic-then-provenance hybrid "
            "(ConstraintVetoMerge) against its two parents and the frontier LLM across three "
            "provenance regimes on identical items. The veto fires only when one branch value's "
            "contradiction against the ancestor clears an absolute threshold while the other "
            "stays below it; otherwise the provenance rule decides. LLM merges are cache-served, "
            "so their seconds/conflict is cache-read latency, not API latency."
        ),
        "n_per_regime": N,
        "seed": SEED,
        "veto_threshold": 0.60,
        "clear_threshold": 0.30,
        "by_regime": results,
        "veto_routing": veto_stats,
    }
    (OUT / "constraint_veto_hybrid.json").write_text(json.dumps(payload, indent=2))
    print("\nWrote", OUT / "constraint_veto_hybrid.json")


if __name__ == "__main__":
    main()
