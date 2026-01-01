#!/usr/bin/env python3
"""Post-hoc, exploratory: open-weight-model check (see ANALYSIS_PLAN_ADDENDUM.md).
NOT part of ANALYSIS_PLAN.md confirmatory tests. Re-runs RQ1 (same 20 locked
scenarios, seed 2026) and the semantic_resolvable category (same 150 scenarios,
seed 3001) with an OPEN-WEIGHT model (Llama 3.1 8B Instruct, Apache/Llama
community licence) served locally via Ollama's OpenAI-compatible endpoint.

This closes the "no open-weight model" gap: every prior model (gpt-5.4-nano/
mini/full, claude-haiku-4-5) is proprietary. It tests whether the RQ1
boundary condition (field-reading is capability-invariant) and the
semantic_resolvable collapse (semantic reasoning is capability-dependent yet
unsolved for cheap models) also hold for an openly released model, not just a
vendor API. Runs fully locally at zero API cost; every call is still cached
under llm_cache/ (keyed by model name, so no collision with the API models).

Requires a local Ollama server with the model pulled:
    ollama pull llama3.1:8b
    OPENAI_API_KEY=ollama python scripts/run_open_weight.py
"""
from __future__ import annotations

import json
from pathlib import Path

from branchmem.benchmark.downstream_tasks import generate_downstream_questions
from branchmem.benchmark.scenario_generator import ScenarioConfig, ScenarioGenerator
from branchmem.benchmark.semantic_resolvable_generator import generate_semantic_resolvable_scenarios
from branchmem.evaluation.metrics import score_downstream
from branchmem.llm.base import build_backend
from branchmem.merge.confidence_rule import ConfidenceRuleMerge
from branchmem.merge.raw_text_llm import RawTextLLMMerge
from branchmem.merge.three_way_llm import ThreeWayLLMMerge
from branchmem.merge.two_way_llm import TwoWayLLMMerge
from branchmem.utils.seeding import set_all_seeds

MODEL = "llama3.1:8b"
BASE_URL = "http://localhost:11434/v1"  # Ollama OpenAI-compatible endpoint
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
            "reference_mini": {"overall": 0.929}, "reference_gpt54": {"overall": 0.936}}


def run_semantic_resolvable(backend):
    set_all_seeds(3001)
    scenarios = generate_semantic_resolvable_scenarios(150, seed=3001)
    # Open-weight semantic check focuses on the flagship ThreeWayLLMMerge (the
    # only LLM strategy that receives the ancestor constraint, i.e. the one that
    # *could* resolve semantically) plus the deterministic rule. The TwoWay/
    # RawText ablations were uniformly 0.000 for every model tested and are
    # omitted here to keep the local run tractable.
    strategies = {
        "confidence_rule": ConfidenceRuleMerge(),
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
                               "two_way_llm": 0.000, "raw_text_llm": 0.000},
            "reference_gpt54": {"three_way_llm": 0.467}}


def main() -> None:
    backend = build_backend({"backend": "openai_compatible", "model": MODEL,
                             "base_url": BASE_URL, "cache_dir": "llm_cache",
                             "temperature": 0.0, "max_tokens": 8000})
    print("Running RQ1 with", MODEL, "...")
    rq1 = run_rq1(backend)
    print("RQ1:", json.dumps(rq1, indent=2))
    print("Running semantic_resolvable with", MODEL, "...")
    semres = run_semantic_resolvable(backend)
    print("semantic_resolvable:", json.dumps(semres["mean_accuracy"], indent=2))

    result = {
        "note": (
            "post-hoc, exploratory open-weight-model check with Llama 3.1 8B "
            "Instruct served locally via Ollama (OpenAI-compatible endpoint). "
            "First openly released model in the study; every other model is "
            "proprietary. Tests whether the RQ1 boundary condition and the "
            "semantic_resolvable 0.000 collapse hold for an open-weight model. "
            "Zero API cost (local). NOT part of ANALYSIS_PLAN.md confirmatory tests."
        ),
        "model": MODEL,
        "base_url": BASE_URL,
        "rq1": rq1,
        "semantic_resolvable": semres,
    }
    (OUT / "open_weight.json").write_text(json.dumps(result, indent=2))
    print("Wrote", OUT / "open_weight.json")


if __name__ == "__main__":
    main()
