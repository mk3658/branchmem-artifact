#!/usr/bin/env python3
"""One item bank, three provenance assignments, four policies (post-hoc).

Why this script exists. The paper's regime table was assembled from three
different experiments, so each column was a different item population: the
informative column came from the locked `resolvable` construction, the absent
column from the 12-template semantic bank, and the misleading column from the
realistic strict-constraint bank. Worse, one cell (ConstraintVeto, informative)
came from the realistic bank while the cells beside it came from the locked
one, which is not a like-for-like comparison and happened to favour our own
method: on the realistic bank the confidence rule also scores 1.000 there.

This script replaces that with a genuinely matched design. It holds ONE bank
fixed --- the 60 realistic strict-constraint templates of
`data/realistic_semantic_templates.json` --- and varies only what the merger is
shown about where each value came from:

  informative  provenance favours the ancestor-compatible value   (seed 5150)
  absent       source and confidence identical on both branches   (seed 4004)
  misleading   provenance favours the contradicting value         (seed 5150)

The seeds are the ones whose frontier merges are already cached, so the LLM
column is served from cache and this run issues no API calls. The informative
and misleading arms are two settings of the same generator at the same seed and
so share their items exactly; the absent arm is the same templates through the
equalizing generator at its own seed, which the output records rather than
glossing over.

Usage: OPENAI_API_KEY=cache-only-sentinel PYTHONPATH=. python scripts/run_matched_regime_table.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

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
from branchmem.merge.confidence_rule import ConfidenceRuleMerge
from branchmem.merge.constraint_veto import ConstraintVetoMerge
from branchmem.merge.three_way_llm import ThreeWayLLMMerge

from run_constraint_veto_hybrid import NLIGatedMerge
from run_operational_baselines import BayesianSourceMerge

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "final"
BANK = ROOT / "data" / "realistic_semantic_templates.json"
N = 60
SEED_SIGNALLED = 5150     # informative and misleading share this seed and these items
SEED_EQUALIZED = 4004     # the equalizing generator's own instantiation

_INFORMATIVE = MisleadingProvenanceConfig(
    misleading_source="inference", misleading_confidence=0.55,
    truthful_source="user", truthful_confidence=0.95,
)
_MISLEADING = MisleadingProvenanceConfig()   # defaults: strong signal on the violator


def regimes() -> dict[str, list]:
    return {
        "informative": generate_misleading_provenance_scenarios(
            N, seed=SEED_SIGNALLED, config=_INFORMATIVE, templates_path=BANK),
        "absent": generate_semantic_resolvable_scenarios(
            N, seed=SEED_EQUALIZED, templates_path=BANK),
        "misleading": generate_misleading_provenance_scenarios(
            N, seed=SEED_SIGNALLED, config=_MISLEADING, templates_path=BANK),
    }


def evaluate(scenarios, strategy) -> dict:
    rows = []
    for s in scenarios:
        questions = generate_downstream_questions(s)
        rows.extend(score_downstream(
            questions, strategy.merge(s.ancestor, s.branch_a, s.branch_b)).detail)
    rows = [r for r in rows if r["category"] != "orthogonal"]
    m = compute_abstention_metrics(rows, categories_in_scope=tuple(
        sorted({r["category"] for r in rows})))
    return {
        "n": len(rows),
        "accuracy": round(sum(r["correct"] for r in rows) / len(rows), 4),
        "coverage": round(m.commit_rate, 4),
        "conditional_risk": (None if m.n_committed == 0
                             else round(m.n_wrong_committed / m.n_committed, 4)),
        "n_committed": m.n_committed,
        "n_wrong_committed": m.n_wrong_committed,
    }


def main() -> None:
    detector = NLIConflictDetector()
    backend = build_backend({"backend": "openai_compatible", "model": "gpt-5.4",
                             "cache_dir": "llm_cache", "temperature": 0.0,
                             "max_tokens": 8000})
    policies = {
        "ConfidenceRule": lambda: ConfidenceRuleMerge(),
        "NliGated": lambda: NLIGatedMerge(detector=detector),
        "ConstraintVeto": lambda: ConstraintVetoMerge(detector=detector),
        "ThreeWayLLM (frontier)": lambda: ThreeWayLLMMerge(backend=backend),
    }

    scen = regimes()
    results: dict[str, dict] = {}
    for name, make in policies.items():
        results[name] = {}
        for regime, scenarios in scen.items():
            results[name][regime] = evaluate(scenarios, make())
            print(f"{name:24} {regime:12} acc={results[name][regime]['accuracy']:.3f} "
                  f"cov={results[name][regime]['coverage']:.3f} "
                  f"risk={results[name][regime]['conditional_risk']}")

    # Learned source reliability, in two variants a deployment can and cannot
    # have. "transferred" is calibrated on the informative regime and applied
    # everywhere, which is what a deployment could actually do. "in-regime" is
    # calibrated on the regime it is scored on, which requires labels from the
    # very regime whose miscalibration is to be detected: an upper bound, not a
    # policy.
    transferred = BayesianSourceMerge.calibrate(scen["informative"])
    for label, calib in (("BayesianSource (transferred)", lambda r: transferred),
                         ("BayesianSource (in-regime)", lambda r: BayesianSourceMerge.calibrate(scen[r]))):
        results[label] = {}
        for regime, scenarios in scen.items():
            strat = BayesianSourceMerge(reliability=calib(regime))
            results[label][regime] = evaluate(scenarios, strat)
            print(f"{label:28} {regime:12} acc={results[label][regime]['accuracy']:.3f}")
    results["_estimated_source_reliability"] = {
        "transferred_from_informative": transferred,
        **{f"in_regime_{r}": BayesianSourceMerge.calibrate(sc) for r, sc in scen.items()},
    }

    print(f"\n{'policy':28} {'informative':>12} {'absent':>9} {'misleading':>11}")
    for name, r in results.items():
        if name.startswith("_"):
            continue
        print(f"{name:28} {r['informative']['accuracy']:12.3f} "
              f"{r['absent']['accuracy']:9.3f} {r['misleading']['accuracy']:11.3f}")

    payload = {
        "note": (
            "Post-hoc, exploratory (not in ANALYSIS_PLAN.md). MATCHED regime table: one item "
            "bank (the 60 realistic strict-constraint templates) under three provenance "
            "assignments, with four policies on identical items. Replaces an earlier table whose "
            "three columns were three different populations and one of whose cells was measured "
            "on a different bank from the cells beside it. The informative and misleading arms "
            "are the same generator at the same seed and so share items exactly; the absent arm "
            "is the same templates through the equalizing generator at its own seed. Accuracy is "
            "recovery of the ancestor-compatible value. Zero new API: the LLM column is cache-served."
        ),
        "bank": str(BANK.relative_to(ROOT)),
        "n_per_regime": N,
        "seeds": {"informative": SEED_SIGNALLED, "absent": SEED_EQUALIZED,
                  "misleading": SEED_SIGNALLED},
        "results": results,
    }
    (OUT / "matched_regime_table.json").write_text(json.dumps(payload, indent=2))
    print("\nWrote", OUT / "matched_regime_table.json")


if __name__ == "__main__":
    main()
