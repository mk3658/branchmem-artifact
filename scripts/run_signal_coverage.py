#!/usr/bin/env python3
"""Post-hoc, exploratory: signal-coverage sweep (see ANALYSIS_PLAN_ADDENDUM.md).
NOT part of ANALYSIS_PLAN.md confirmatory tests.

Motivation: the paper demonstrates the boundary condition mainly at the two
extremes -- perfect synthetic signal (all resolvable conflicts carry a
source/confidence signal) vs. zero real signal (MemoryAgentBench, no signal on
any conflict). This experiment maps the *transition* by varying the FRACTION of
resolvable conflicts that carry an exploitable provenance signal:
coverage in {0.0, 0.25, 0.5, 0.75, 1.0}. Signal-bearing conflicts get the usual
user/0.95 vs inference/0.55 signal (ground truth = reliable branch, 15% noise);
signal-absent conflicts get equal source/confidence on both branches
(observation/0.7) with an unrecoverable ground truth -- exactly the MAB
condition. Each scenario batches 4 resolvable conflicts, round(coverage*4) of
which carry signal, so a single merge call sees a realistic mix.

We report ThreeWayLLMMerge and ConfidenceRuleMerge resolvable accuracy and
commit rate as a function of coverage. gpt-5.4-nano (the locked model), cheap.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from branchmem.benchmark.downstream_tasks import generate_downstream_questions
from branchmem.benchmark.scenario_generator import Scenario
from branchmem.eval.abstention_metrics import compute_abstention_metrics
from branchmem.evaluation.metrics import score_downstream
from branchmem.memory.branch_sim import BranchSimulator, UpdateSpec
from branchmem.llm.base import build_backend
from branchmem.merge.confidence_rule import ConfidenceRuleMerge
from branchmem.merge.three_way_llm import ThreeWayLLMMerge
from branchmem.utils.seeding import set_all_seeds

SEED = 6001
N_SCENARIOS = 24
N_RESOLVABLE = 4
NOISE = 0.15
MODEL = "gpt-5.4-nano"
COVERAGES = [0.0, 0.25, 0.5, 0.75, 1.0]
OUT = Path(__file__).resolve().parents[1] / "results" / "final"
TEMPLATES = Path(__file__).resolve().parents[1] / "data" / "scenario_templates.json"


def build_scenarios(coverage, seed):
    pools = json.loads(TEMPLATES.read_text())
    entities, objective = pools["entities"], pools["objective_predicates"]
    sim = BranchSimulator()
    n_signal = round(coverage * N_RESOLVABLE)
    scenarios = []
    for i in range(N_SCENARIOS):
        rng = random.Random(f"sigcov:{seed}:{coverage}:{i}")
        entity = rng.choice(entities)
        preds = rng.sample(list(objective.keys()), k=min(N_RESOLVABLE + 2, len(objective)))
        ancestor_facts = [(entity, p, rng.choice(objective[p])) for p in preds[N_RESOLVABLE:]]
        ancestor = sim.create_common_ancestor(facts=ancestor_facts, timestamp=0.0)
        ua, ub = [], []
        for j, p in enumerate(preds[:N_RESOLVABLE]):
            va, vb = rng.sample(objective[p], k=2)
            has_signal = j < n_signal
            if has_signal:
                a_user = rng.random() < 0.5
                sa, ca = ("user", 0.95) if a_user else ("inference", 0.55)
                sb, cb = ("inference", 0.55) if a_user else ("user", 0.95)
                reliable = va if a_user else vb
                unreliable = vb if a_user else va
                gt = unreliable if rng.random() < NOISE else reliable
            else:
                # signal-absent: equal source/confidence, unrecoverable ground truth (MAB-like)
                sa = sb = "observation"
                ca = cb = 0.7
                gt = rng.choice([va, vb])
            ta, tb = 1.0 + rng.uniform(0.1, 10.0), 1.0 + rng.uniform(0.1, 10.0)
            ua.append(UpdateSpec(entity=entity, predicate=p, value=va, timestamp=ta, source=sa,
                                 confidence=ca, divergence_type="resolvable_conflict", ground_truth_value=gt))
            ub.append(UpdateSpec(entity=entity, predicate=p, value=vb, timestamp=tb, source=sb,
                                 confidence=cb, divergence_type="resolvable_conflict", ground_truth_value=gt))
        ba, bb, pairs = sim.diverge(ancestor, ua, ub, fork_point_timestamp=1.0)
        scenarios.append(Scenario(scenario_id=f"sigcov_{seed}_{coverage}_{i:03d}", ancestor=ancestor,
                                  branch_a=ba, branch_b=bb, conflict_pairs=pairs, source="synthetic",
                                  metadata={"category": "signal_coverage", "coverage": coverage}))
    return scenarios


def main() -> None:
    set_all_seeds(SEED)
    backend = build_backend({"backend": "openai_compatible", "model": MODEL, "cache_dir": "llm_cache",
                             "temperature": 0.0, "max_tokens": 2048})
    strategies = {"three_way_llm": ThreeWayLLMMerge(backend=backend),
                  "confidence_rule": ConfidenceRuleMerge()}
    per_coverage = []
    for cov in COVERAGES:
        scenarios = build_scenarios(cov, SEED)
        acc = {n: [] for n in strategies}
        detail = {n: [] for n in strategies}
        for s in scenarios:
            q = generate_downstream_questions(s)
            for n, strat in strategies.items():
                sc = strat.merge(s.ancestor, s.branch_a, s.branch_b)
                score = score_downstream(q, sc)
                acc[n].append(score.accuracy_for("resolvable"))
                detail[n].extend(score.detail)
        def mean(v):
            v = [x for x in v if x == x]
            return sum(v) / len(v) if v else float("nan")
        row = {"coverage": cov}
        for n in strategies:
            m = compute_abstention_metrics(detail[n], categories_in_scope=("resolvable",))
            row[n] = {"resolvable_accuracy": mean(acc[n]), "commit_rate": m.commit_rate}
        per_coverage.append(row)
        print(f"coverage={cov}: 3way acc={row['three_way_llm']['resolvable_accuracy']:.3f} "
              f"commit={row['three_way_llm']['commit_rate']:.3f} | "
              f"rule acc={row['confidence_rule']['resolvable_accuracy']:.3f}")

    result = {
        "note": (
            "post-hoc, exploratory signal-coverage sweep: fraction of resolvable conflicts "
            "carrying an exploitable provenance signal, from 0.0 (MAB-like, no signal) to 1.0 "
            "(all synthetic). Maps the transition between the paper's two extremes. "
            f"n={N_SCENARIOS} scenarios/level, {N_RESOLVABLE} resolvable conflicts each, "
            "gpt-5.4-nano. NOT part of ANALYSIS_PLAN.md confirmatory tests."
        ),
        "run_metadata": {"seed": SEED, "n_scenarios": N_SCENARIOS, "n_resolvable": N_RESOLVABLE,
                         "noise": NOISE, "model": MODEL, "coverages": COVERAGES},
        "per_coverage": per_coverage,
    }
    (OUT / "signal_coverage.json").write_text(json.dumps(result, indent=2))
    print("Wrote", OUT / "signal_coverage.json")


if __name__ == "__main__":
    main()
