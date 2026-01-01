#!/usr/bin/env python3
"""Post-hoc, exploratory: stronger-model check (see ANALYSIS_PLAN_ADDENDUM.md).
NOT part of ANALYSIS_PLAN.md confirmatory tests. Re-runs RQ1 (same 20 locked
scenarios) and the semantic_resolvable category (same 150 scenarios, seed 3001)
with gpt-5.4 (the full frontier-tier OpenAI model), to test whether the RQ1
boundary condition and the semantic_resolvable 0.000 collapse are artifacts of
the cheap gpt-5.4-nano/mini tier or hold for a substantially stronger model.
Same-provider capability step-up (nano/mini -> full), isolating capability
from vendor (the cross-vendor Claude check is separate).
"""
from __future__ import annotations

import json
from pathlib import Path

from branchmem.benchmark.downstream_tasks import generate_downstream_questions
from branchmem.benchmark.scenario_generator import ScenarioConfig, ScenarioGenerator
from branchmem.benchmark.semantic_resolvable_generator import generate_semantic_resolvable_scenarios
from branchmem.evaluation.metrics import score_downstream
from branchmem.llm.base import build_backend
from branchmem.merge.branch_discard import BranchDiscard
from branchmem.merge.confidence_rule import ConfidenceRuleMerge
from branchmem.merge.last_writer_wins import LastWriterWins
from branchmem.merge.naive_concat import NaiveConcat
from branchmem.merge.raw_text_llm import RawTextLLMMerge
from branchmem.merge.three_way_llm import ThreeWayLLMMerge
from branchmem.merge.two_way_llm import TwoWayLLMMerge
from branchmem.utils.seeding import set_all_seeds

MODEL = "gpt-5.4"
OUT = Path(__file__).resolve().parents[1] / "results" / "final"


def run_rq1(backend):
    set_all_seeds(2026)
    gen = ScenarioGenerator()
    scenarios = gen.generate(20, ScenarioConfig(divergence_span=4.0), seed=2026)
    strat = ThreeWayLLMMerge(backend=backend)
    overall, resolvable = [], []
    for s in scenarios:
        q = generate_downstream_questions(s)
        sc = strat.merge(s.ancestor, s.branch_a, s.branch_b)
        score = score_downstream(q, sc)
        overall.append(score.accuracy)
        resolvable.append(score.accuracy_for("resolvable"))
    def mean(v):
        v = [x for x in v if x == x]
        return sum(v) / len(v) if v else float("nan")
    return {"n": 20, "mean_overall": mean(overall), "mean_resolvable": mean(resolvable),
            "reference_nano": {"overall": 0.914, "resolvable_locked": 0.858},
            "reference_mini": {"overall": 0.929}}


def run_semantic_resolvable(backend):
    set_all_seeds(3001)
    scenarios = generate_semantic_resolvable_scenarios(150, seed=3001)
    strategies = {
        "confidence_rule": ConfidenceRuleMerge(),
        "two_way_llm": TwoWayLLMMerge(backend=backend),
        "raw_text_llm": RawTextLLMMerge(backend=backend),
        "three_way_llm": ThreeWayLLMMerge(backend=backend),
    }
    acc = {n: [] for n in strategies}
    for s in scenarios:
        q = generate_downstream_questions(s)
        for n, strat in strategies.items():
            sc = strat.merge(s.ancestor, s.branch_a, s.branch_b)
            acc[n].append(score_downstream(q, sc).accuracy_for("resolvable"))
    def mean(v):
        v = [x for x in v if x == x]
        return sum(v) / len(v) if v else float("nan")
    return {"n": 150, "mean_accuracy": {n: mean(v) for n, v in acc.items()},
            "reference_nano": {"three_way_llm": 0.000, "confidence_rule": 0.000,
                               "two_way_llm": 0.000, "raw_text_llm": 0.000}}


def main() -> None:
    backend = build_backend({"backend": "openai_compatible", "model": MODEL, "cache_dir": "llm_cache",
                             "temperature": 0.0, "max_tokens": 8000})
    print("Running RQ1 with", MODEL, "...")
    rq1 = run_rq1(backend)
    print("RQ1:", json.dumps(rq1, indent=2))
    print("Running semantic_resolvable with", MODEL, "...")
    semres = run_semantic_resolvable(backend)
    print("semantic_resolvable:", json.dumps(semres["mean_accuracy"], indent=2))

    result = {
        "note": (
            "post-hoc, exploratory stronger-model check with gpt-5.4 (full frontier "
            "OpenAI model), same-provider capability step-up from nano/mini. Tests "
            "whether the RQ1 boundary condition and the semantic_resolvable 0.000 "
            "collapse persist at a stronger capability tier. NOT part of "
            "ANALYSIS_PLAN.md confirmatory tests."
        ),
        "model": MODEL,
        "rq1": rq1,
        "semantic_resolvable": semres,
    }
    (OUT / "stronger_model.json").write_text(json.dumps(result, indent=2))
    print("Wrote", OUT / "stronger_model.json")


if __name__ == "__main__":
    main()
