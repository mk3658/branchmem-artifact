#!/usr/bin/env python3
"""Experiment: MISLEADING provenance (post-hoc, exploratory; NOT in ANALYSIS_PLAN.md).

The benchmark's central diagnostic gap: in every prior category the provenance
field either points at the answer (`resolvable`, so a one-line rule wins) or is
absent (`semantic_resolvable`, so the rule can only abstain). Neither separates
FIELD-READING from SEMANTIC REASONING, because the field never points the wrong
way.

Here it does. Each item pairs a strict ancestor constraint with a provenance
signal favouring the value that CONTRADICTS it (Section: misleading_provenance).
Ground truth is the semantically compatible value, so:
  - every confidence-following strategy is wrong BY CONSTRUCTION (0.000, via a
    confident wrong commit, not an abstention);
  - only a strategy that overrides the field using ancestor semantics can score
    above 0.

Informative either way. Override => LLM merging supplies something no
field-reading rule can. Follow-the-field => LLM merging here is
provenance-following, not semantic reasoning, a sharper negative than "a rule
matches it".

Also evaluates two deterministic baselines that a rule-based deployment would
plausibly use instead of a bare confidence comparison:
  - ConfidenceMarginMerge: commit only if the confidence gap exceeds a margin,
    else abstain (tests whether margin-gating rescues the rule);
  - a "both-retained" reader (NaiveConcat) for reference.

Usage: OPENAI_API_KEY=... PYTHONPATH=. python scripts/run_misleading_provenance.py [n]
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

from branchmem.benchmark.downstream_tasks import generate_downstream_questions
from branchmem.benchmark.misleading_provenance_generator import (
    generate_misleading_provenance_scenarios,
)
from branchmem.eval.abstention_metrics import compute_abstention_metrics
from branchmem.evaluation.metrics import score_downstream
from branchmem.llm.base import build_backend
from branchmem.memory.schemas import MergeResult, Resolution, ResolvedFact
from branchmem.merge.base import MergeStrategy, current_facts_by_key, find_collisions
from branchmem.merge.confidence_rule import ConfidenceRuleMerge
from branchmem.merge.naive_concat import NaiveConcat
from branchmem.merge.three_way_llm import ThreeWayLLMMerge
from branchmem.merge.two_way_llm import TwoWayLLMMerge

OUT = Path(__file__).resolve().parents[1] / "results" / "final"
SEED = 5150


class ConfidenceMarginMerge(MergeStrategy):
    """Deterministic rule with a margin: commit only when the confidence gap is
    large enough to be trusted, otherwise abstain. A natural hardening of
    ConfidenceRuleMerge that a cautious deployment would use."""

    def __init__(self, margin: float = 0.3) -> None:
        self.margin = margin
        self.name = f"confidence_margin_{margin}"

    def merge(self, ancestor, branch_a, branch_b) -> MergeResult:
        a_facts = current_facts_by_key(branch_a)
        b_facts = current_facts_by_key(branch_b)
        collisions = find_collisions(branch_a, branch_b)
        resolved: list[ResolvedFact] = []
        for key in sorted((set(a_facts) | set(b_facts)) - set(collisions)):
            fa, fb = a_facts.get(key), b_facts.get(key)
            fact = fa or fb
            resolved.append(ResolvedFact(fact=fact, resolution=Resolution.KEPT,
                                         source_branch_ids=[fact.branch_id]))
        for key, (fa, fb) in collisions.items():
            gap = abs(fa.confidence - fb.confidence)
            if gap >= self.margin:
                win = fa if fa.confidence > fb.confidence else fb
                resolved.append(ResolvedFact(fact=win, resolution=Resolution.KEPT,
                                             source_branch_ids=[win.branch_id],
                                             justification=f"confidence gap {gap:.2f} >= {self.margin}"))
            else:
                resolved.append(ResolvedFact(fact=fa, resolution=Resolution.FLAGGED_UNRESOLVED,
                                             source_branch_ids=[branch_a.branch_id, branch_b.branch_id],
                                             justification=f"confidence gap {gap:.2f} < {self.margin}"))
        return MergeResult(strategy_name=self.name, resulting_facts=resolved)


def evaluate(scenarios, strategies):
    detail = defaultdict(list)
    for s in scenarios:
        qs = generate_downstream_questions(s)
        for name, strat in strategies.items():
            res = strat.merge(s.ancestor, s.branch_a, s.branch_b)
            detail[name].extend(score_downstream(qs, res).detail)
    out = {}
    for name, rows in detail.items():
        cats = tuple(sorted({d["category"] for d in rows}))
        m = compute_abstention_metrics(rows, categories_in_scope=cats)
        out[name] = {
            "accuracy": sum(d["correct"] for d in rows) / len(rows),
            "n": len(rows),
            "commit_rate": m.commit_rate,
            "conditional_on_commit_accuracy": m.conditional_on_commit_accuracy,
            "wrong_commit_rate": m.wrong_commit_rate,
            "n_committed": m.n_committed,
            "n_correct_committed": m.n_correct_committed,
            "n_wrong_committed": m.n_wrong_committed,
            "n_flagged": m.n_flagged,
        }
    return out


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    scenarios = generate_misleading_provenance_scenarios(n, seed=SEED)

    common = {"cache_dir": "llm_cache", "temperature": 0.0, "max_tokens": 8000}
    results = {}
    for label, model in (("gpt-5.4", "gpt-5.4"), ("gpt-5.4-nano", "gpt-5.4-nano")):
        backend = build_backend({"backend": "openai_compatible", "model": model, **common})
        strategies = {
            "confidence_rule": ConfidenceRuleMerge(),
            "confidence_margin_0.3": ConfidenceMarginMerge(margin=0.3),
            "naive_concat": NaiveConcat(),
            "two_way_llm": TwoWayLLMMerge(backend=backend),
            "three_way_llm": ThreeWayLLMMerge(backend=backend),
        }
        results[label] = evaluate(scenarios, strategies)
        print(f"\n=== solver: {label}  (n={n} misleading-provenance conflicts) ===")
        for sname, r in results[label].items():
            print(f"  {sname:22} acc={r['accuracy']:.3f}  commit={r['commit_rate']:.3f}  "
                  f"cond_acc={r['conditional_on_commit_accuracy']}  "
                  f"wrong_commit={r['wrong_commit_rate']:.3f}  "
                  f"({r['n_committed']} commits, {r['n_wrong_committed']} wrong, {r['n_flagged']} flagged)")

    payload = {
        "note": (
            "Post-hoc, exploratory (not in ANALYSIS_PLAN.md). MISLEADING provenance: a strict "
            "ancestor constraint paired with a source/confidence signal favouring the value that "
            "CONTRADICTS it. Ground truth is the semantically compatible value, so every "
            "confidence-following strategy is wrong by construction (0.000 via confident wrong "
            "commits). Tests whether LLM merging overrides a misleading field using ancestor "
            "semantics, i.e. separates field-reading from semantic reasoning. Content held fixed "
            "vs the realism check; only the provenance assignment changes."
        ),
        "seed": SEED, "n_scenarios": n,
        "deterministic_rule_is_wrong_by_construction": True,
        "by_solver": results,
    }
    (OUT / "misleading_provenance.json").write_text(json.dumps(payload, indent=2))
    print("\nWrote", OUT / "misleading_provenance.json")


if __name__ == "__main__":
    main()
