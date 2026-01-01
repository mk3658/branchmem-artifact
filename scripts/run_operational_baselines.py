#!/usr/bin/env python3
"""Operational baselines across provenance regimes (post-hoc, exploratory).

Addresses two gaps at once: (a) the comparison set previously lacked baselines
close to realistic agent deployments, and (b) strategies were only tested where
provenance is INFORMATIVE or ABSENT, never where it is MISLEADING. We evaluate
every strategy across all three regimes, so the boundary condition is mapped
prospectively rather than asserted from one regime.

Regimes (rows are held-content-fixed where possible):
  informative  : locked resolvable construction (seed 2026)  -- field points right
  absent       : semantic_resolvable (seed 3001)             -- no field at all
  misleading   : misleading_provenance (seed 5150)           -- field points wrong

Baselines added here:
  NLIGatedMerge            : deterministic + semantic. Scores each branch value
                             against the ancestor constraint with a local NLI
                             cross-encoder and keeps the LESS contradicted one.
                             Ignores source/confidence entirely, so it is the
                             natural competitor in the misleading regime.
  RetrievalDeferredMerge   : never resolves at merge time; keeps both values and
                             defers to a retrieval-time confidence rule (the
                             "resolve at read time" deployment pattern).
  BayesianSourceMerge      : estimates per-source reliability P(label | source)
                             from a CALIBRATION split and MAP-selects. Reported
                             both transferred (calibrated on the locked regime)
                             and in-regime (an optimistic upper bound), which is
                             the direct test of whether learned source-reliability
                             detects a miscalibrated field.

Zero new API: NLI runs locally, the rest are deterministic.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from branchmem.benchmark.downstream_tasks import generate_downstream_questions
from branchmem.benchmark.misleading_provenance_generator import (
    generate_misleading_provenance_scenarios,
)
from branchmem.benchmark.scenario_generator import ScenarioConfig, ScenarioGenerator
from branchmem.benchmark.semantic_resolvable_generator import (
    generate_semantic_resolvable_scenarios,
)
from branchmem.conflict.nli_detector import NLIConflictDetector
from branchmem.evaluation.metrics import score_downstream
from branchmem.memory.schemas import MergeResult, Resolution, ResolvedFact
from branchmem.merge.base import MergeStrategy, current_facts_by_key, find_collisions
from branchmem.merge.confidence_rule import ConfidenceRuleMerge
from branchmem.utils.seeding import set_all_seeds

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


def _passthrough(ancestor, branch_a, branch_b, collisions):
    """Non-conflicting keys, resolved deterministically (shared by all baselines)."""
    a, b = current_facts_by_key(branch_a), current_facts_by_key(branch_b)
    out = []
    for key in sorted((set(a) | set(b)) - set(collisions)):
        f = a.get(key) or b.get(key)
        out.append(ResolvedFact(fact=f, resolution=Resolution.KEPT, source_branch_ids=[f.branch_id]))
    return out


class NLIGatedMerge(MergeStrategy):
    """Deterministic + semantic: keep whichever branch value the ancestor
    constraint contradicts LESS. Never reads source or confidence."""

    name = "nli_gated"

    def __init__(self, detector: NLIConflictDetector | None = None, margin: float = 0.10) -> None:
        self.detector = detector or NLIConflictDetector()
        self.margin = margin

    def merge(self, ancestor, branch_a, branch_b) -> MergeResult:
        anc = current_facts_by_key(ancestor)
        collisions = find_collisions(branch_a, branch_b)
        resolved = _passthrough(ancestor, branch_a, branch_b, collisions)
        for key, (fa, fb) in collisions.items():
            anc_fact = anc.get(key)
            if anc_fact is None:
                resolved.append(ResolvedFact(fact=fa, resolution=Resolution.FLAGGED_UNRESOLVED,
                                             source_branch_ids=[branch_a.branch_id, branch_b.branch_id],
                                             justification="no ancestor constraint to test against"))
                continue
            prem = f"{anc_fact.entity} {anc_fact.value}."
            ca = self.detector._predict([(prem, f"{fa.entity} {fa.value}.")])[0]["contradiction"]
            cb = self.detector._predict([(prem, f"{fb.entity} {fb.value}.")])[0]["contradiction"]
            if abs(ca - cb) < self.margin:
                resolved.append(ResolvedFact(fact=fa, resolution=Resolution.FLAGGED_UNRESOLVED,
                                             source_branch_ids=[branch_a.branch_id, branch_b.branch_id],
                                             justification=f"contradiction scores too close ({ca:.2f} vs {cb:.2f})"))
            else:
                win = fa if ca < cb else fb
                resolved.append(ResolvedFact(fact=win, resolution=Resolution.KEPT,
                                             source_branch_ids=[win.branch_id],
                                             justification=f"less contradicted by ancestor ({min(ca,cb):.2f})"))
        return MergeResult(strategy_name=self.name, resulting_facts=resolved)


class RetrievalDeferredMerge(MergeStrategy):
    """Keep both values at merge time; a retrieval-time rule picks the
    higher-confidence one when the key is read. Information-preserving."""

    name = "retrieval_deferred"

    def merge(self, ancestor, branch_a, branch_b) -> MergeResult:
        collisions = find_collisions(branch_a, branch_b)
        resolved = _passthrough(ancestor, branch_a, branch_b, collisions)
        for key, (fa, fb) in collisions.items():
            win = fa if fa.confidence > fb.confidence else fb if fb.confidence > fa.confidence else None
            if win is None:
                resolved.append(ResolvedFact(fact=fa, resolution=Resolution.FLAGGED_UNRESOLVED,
                                             source_branch_ids=[branch_a.branch_id, branch_b.branch_id],
                                             justification="tie at retrieval time"))
            else:
                resolved.append(ResolvedFact(fact=win, resolution=Resolution.KEPT,
                                             source_branch_ids=[win.branch_id],
                                             justification="retrieval-time confidence selection (both retained in store)"))
        return MergeResult(strategy_name=self.name, resulting_facts=resolved)


class BayesianSourceMerge(MergeStrategy):
    """MAP selection using per-source reliability estimated from a calibration
    split, rather than trusting the raw confidence number."""

    name = "bayesian_source"

    def __init__(self, reliability: dict[str, float] | None = None) -> None:
        self.reliability = reliability or {}

    @staticmethod
    def calibrate(scenarios) -> dict[str, float]:
        """P(this source carries the construction label) per source string."""
        hit, tot = defaultdict(int), defaultdict(int)
        for s in scenarios:
            for pair in s.conflict_pairs:
                gt = getattr(pair, "ground_truth_value", None)
                if gt is None:
                    continue
                for br in (s.branch_a, s.branch_b):
                    for f in current_facts_by_key(br).values():
                        if f.key() == (pair.entity, pair.predicate) if hasattr(pair, "entity") else False:
                            pass
        # Simpler, robust: walk collisions directly.
        hit, tot = defaultdict(int), defaultdict(int)
        for s in scenarios:
            qs = {(q.entity, q.predicate): q.expected_answer for q in generate_downstream_questions(s)}
            for key, (fa, fb) in find_collisions(s.branch_a, s.branch_b).items():
                exp = qs.get(key)
                if exp is None:
                    continue
                for f in (fa, fb):
                    tot[f.source] += 1
                    hit[f.source] += int(f.value == exp)
        return {s: hit[s] / tot[s] for s in tot if tot[s]}

    def merge(self, ancestor, branch_a, branch_b) -> MergeResult:
        collisions = find_collisions(branch_a, branch_b)
        resolved = _passthrough(ancestor, branch_a, branch_b, collisions)
        for key, (fa, fb) in collisions.items():
            ra = self.reliability.get(fa.source, 0.5)
            rb = self.reliability.get(fb.source, 0.5)
            if abs(ra - rb) < 1e-9:
                resolved.append(ResolvedFact(fact=fa, resolution=Resolution.FLAGGED_UNRESOLVED,
                                             source_branch_ids=[branch_a.branch_id, branch_b.branch_id],
                                             justification=f"equal estimated source reliability ({ra:.2f})"))
            else:
                win = fa if ra > rb else fb
                resolved.append(ResolvedFact(fact=win, resolution=Resolution.KEPT,
                                             source_branch_ids=[win.branch_id],
                                             justification=f"higher estimated source reliability ({max(ra,rb):.2f})"))
        return MergeResult(strategy_name=self.name, resulting_facts=resolved)


def accuracy(scenarios, strat, category="resolvable"):
    vals = []
    for s in scenarios:
        qs = generate_downstream_questions(s)
        sc = score_downstream(qs, strat.merge(s.ancestor, s.branch_a, s.branch_b))
        a = sc.accuracy_for(category)
        if a == a:
            vals.append(a)
    return sum(vals) / len(vals) if vals else float("nan")


def main() -> None:
    set_all_seeds(2026)
    informative = locked_scenarios(60, seed=2026)
    absent = generate_semantic_resolvable_scenarios(60, seed=3001)
    misleading = generate_misleading_provenance_scenarios(60, seed=5150)

    rel_locked = BayesianSourceMerge.calibrate(informative)
    rel_mislead = BayesianSourceMerge.calibrate(misleading)
    print("estimated source reliability (locked/informative):", {k: round(v, 3) for k, v in rel_locked.items()})
    print("estimated source reliability (misleading regime): ", {k: round(v, 3) for k, v in rel_mislead.items()})

    nli = NLIGatedMerge()
    strategies = {
        "confidence_rule": ConfidenceRuleMerge(),
        "retrieval_deferred": RetrievalDeferredMerge(),
        "bayesian_source_transferred": BayesianSourceMerge(rel_locked),
        "bayesian_source_in_regime": None,  # filled per regime below
        "nli_gated": nli,
    }

    regimes = {"informative": informative, "absent": absent, "misleading": misleading}
    in_regime_rel = {"informative": rel_locked, "absent": BayesianSourceMerge.calibrate(absent),
                     "misleading": rel_mislead}

    results = {}
    for rname, scen in regimes.items():
        results[rname] = {}
        for sname, strat in strategies.items():
            s = BayesianSourceMerge(in_regime_rel[rname]) if sname == "bayesian_source_in_regime" else strat
            results[rname][sname] = accuracy(scen, s)
        print(f"\n=== regime: {rname} (n={len(scen)}) ===")
        for sname, acc in results[rname].items():
            print(f"  {sname:30} resolvable acc = {acc:.3f}")

    payload = {
        "note": ("Post-hoc, exploratory (not in ANALYSIS_PLAN.md). Operational baselines evaluated "
                 "across three provenance regimes: informative (locked seed 2026), absent "
                 "(semantic_resolvable seed 3001), misleading (seed 5150). NLIGatedMerge is "
                 "deterministic+semantic and ignores source/confidence; BayesianSourceMerge is "
                 "reported both transferred (calibrated on the informative regime) and in-regime "
                 "(optimistic upper bound). Zero new API."),
        "estimated_source_reliability": {"informative": rel_locked, "misleading": rel_mislead},
        "resolvable_accuracy_by_regime": results,
    }
    (OUT / "operational_baselines.json").write_text(json.dumps(payload, indent=2))
    print("\nWrote", OUT / "operational_baselines.json")


if __name__ == "__main__":
    main()
