#!/usr/bin/env python3
"""Does the provenance-inversion failure need a HARD constraint? (post-hoc)

The paper's diagnostic puts a strict ancestor constraint ("has a documented
anaphylactic allergy to shrimp") against provenance that favours the value
contradicting it. A fair objection is that the result may be an artifact of
strict-constraint semantics: perhaps the merger only ignores a constraint
because the constraint is phrased as a logical impossibility, and ordinary
revisable preferences would behave differently.

This script holds content fixed and varies ONLY the modal force of the
ancestor. `data/defeasible_ancestor_templates.json` copies every predicate and
both branch values verbatim from the strict bank and rewrites the ancestor as a
preference or stale observation ("mentioned a while ago that they had been
cutting back on shellfish"). Strict and defeasible items therefore differ in
one respect.

The measure is label-free, which matters: with a defeasible ancestor there is
no logically correct answer to score against, so accuracy would be
meaningless. Instead each item is merged TWICE with identical content and
opposite provenance -- once with the strong source/confidence on the
ancestor-consistent value, once on the ancestor-inconsistent value -- and we
ask whether the decision moved:

  flip rate = P(committed value differs between the two arms | commits in both)

A merger that reasons over the ancestor has flip rate near 0: the content did
not change, so neither should its answer. A merger that reads the provenance
field has flip rate near 1. `ConfidenceRuleMerge` is included as the trivial
upper reference (flip rate 1 by construction).

Usage: PYTHONPATH=. python scripts/run_ancestor_strength_ablation.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from branchmem.benchmark.misleading_provenance_generator import (
    MisleadingProvenanceConfig,
    generate_misleading_provenance_scenarios,
)
from branchmem.llm.base import build_backend
from branchmem.memory.schemas import Resolution
from branchmem.merge.base import find_collisions
from branchmem.merge.confidence_rule import ConfidenceRuleMerge
from branchmem.merge.three_way_llm import ThreeWayLLMMerge

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "final"
STRICT_BANK = ROOT / "data" / "realistic_semantic_templates.json"
DEFEASIBLE_BANK = ROOT / "data" / "defeasible_ancestor_templates.json"

SEED = 5150          # same seed as the cached misleading-provenance runs
N = int(os.environ.get("ANCESTOR_STRENGTH_N", "60"))

# Provenance pointing AT the ancestor-inconsistent value (the paper's condition).
MISLEADING = MisleadingProvenanceConfig()
# Provenance pointing AT the ancestor-consistent value (matched control).
INFORMATIVE = MisleadingProvenanceConfig(
    misleading_source="inference", misleading_confidence=0.55,
    truthful_source="user", truthful_confidence=0.95,
)


def decide(strategy, scenario) -> dict:
    """The single conflict's outcome: what was committed, and what the field said."""
    collisions = find_collisions(scenario.branch_a, scenario.branch_b)
    (key, (fa, fb)), = collisions.items()
    result = strategy.merge(scenario.ancestor, scenario.branch_a, scenario.branch_b)
    committed, resolution = None, None
    for rf in result.resulting_facts:
        if rf.fact.key() == key:
            resolution = rf.resolution
            committed = None if rf.resolution == Resolution.FLAGGED_UNRESOLVED else rf.fact.value
    favoured = fa.value if fa.confidence > fb.confidence else (
        fb.value if fb.confidence > fa.confidence else None)
    truth = scenario.conflict_pairs[0].ground_truth_value
    return {
        "scenario_id": scenario.scenario_id,
        "committed": committed,
        "abstained": resolution == Resolution.FLAGGED_UNRESOLVED,
        "provenance_favoured": favoured,
        "ancestor_consistent": truth,
        "follows_provenance": committed is not None and committed == favoured,
        "ancestor_consistent_choice": committed is not None and committed == truth,
    }


def arm(strategy, bank: Path, config: MisleadingProvenanceConfig) -> list[dict]:
    scenarios = generate_misleading_provenance_scenarios(
        N, seed=SEED, config=config, templates_path=bank
    )
    return [decide(strategy, s) for s in scenarios]


def summarise(mis: list[dict], inf: list[dict]) -> dict:
    both = [(m, i) for m, i in zip(mis, inf) if not m["abstained"] and not i["abstained"]]
    flips = sum(1 for m, i in both if m["committed"] != i["committed"])
    return {
        "n": len(mis),
        "n_commit_both_arms": len(both),
        "flip_rate": None if not both else round(flips / len(both), 3),
        "misleading_arm": {
            "abstain_rate": round(sum(m["abstained"] for m in mis) / len(mis), 3),
            "follows_provenance": round(sum(m["follows_provenance"] for m in mis) / len(mis), 3),
            "ancestor_consistent": round(
                sum(m["ancestor_consistent_choice"] for m in mis) / len(mis), 3),
        },
        "informative_arm": {
            "abstain_rate": round(sum(i["abstained"] for i in inf) / len(inf), 3),
            "follows_provenance": round(sum(i["follows_provenance"] for i in inf) / len(inf), 3),
            "ancestor_consistent": round(
                sum(i["ancestor_consistent_choice"] for i in inf) / len(inf), 3),
        },
    }


def main() -> None:
    models = ["gpt-5.4", "gpt-5.4-nano"]
    banks = {"strict_constraint": STRICT_BANK, "defeasible_preference": DEFEASIBLE_BANK}
    out: dict[str, dict] = {}

    for bank_name, bank in banks.items():
        out[bank_name] = {}
        rule = ConfidenceRuleMerge()
        out[bank_name]["ConfidenceRule"] = summarise(
            arm(rule, bank, MISLEADING), arm(rule, bank, INFORMATIVE)
        )
        for model in models:
            backend = build_backend({"backend": "openai_compatible", "model": model,
                                     "cache_dir": "llm_cache", "temperature": 0.0,
                                     "max_tokens": 8000})
            strat = ThreeWayLLMMerge(backend=backend)
            print(f"{bank_name} / {model}: misleading arm ...")
            mis = arm(strat, bank, MISLEADING)
            print(f"{bank_name} / {model}: informative arm ...")
            inf = arm(strat, bank, INFORMATIVE)
            out[bank_name][f"ThreeWayLLM ({model})"] = summarise(mis, inf)

    for bank_name, policies in out.items():
        print(f"\n=== {bank_name} (n={N}) ===")
        print(f"{'policy':28} {'flip':>6} {'mis:follow':>11} {'mis:abst':>9} "
              f"{'inf:follow':>11} {'inf:abst':>9}")
        for name, s in policies.items():
            fl = "   n/a" if s["flip_rate"] is None else f"{s['flip_rate']:6.3f}"
            print(f"{name:28} {fl} {s['misleading_arm']['follows_provenance']:11.3f} "
                  f"{s['misleading_arm']['abstain_rate']:9.3f} "
                  f"{s['informative_arm']['follows_provenance']:11.3f} "
                  f"{s['informative_arm']['abstain_rate']:9.3f}")

    payload = {
        "note": (
            "Post-hoc, exploratory (not in ANALYSIS_PLAN.md). Matched ancestor-strength "
            "ablation: identical predicates and branch values, ancestor phrased as a strict "
            "constraint vs an ordinary revisable preference. Each item merged twice with "
            "opposite provenance; flip_rate is the share of items whose committed value moves "
            "when only the metadata moves, among items committed in both arms. Accuracy is not "
            "reported for the defeasible bank because a revisable preference does not entail a "
            "correct answer; 'ancestor_consistent' is descriptive, not a correctness claim."
        ),
        "n_per_arm": N,
        "seed": SEED,
        "banks": {"strict_constraint": str(STRICT_BANK.relative_to(ROOT)),
                  "defeasible_preference": str(DEFEASIBLE_BANK.relative_to(ROOT))},
        "results": out,
    }
    (OUT / "ancestor_strength_ablation.json").write_text(json.dumps(payload, indent=2))
    print("\nWrote", OUT / "ancestor_strength_ablation.json")


if __name__ == "__main__":
    main()
