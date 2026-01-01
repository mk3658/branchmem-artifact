#!/usr/bin/env python3
"""End-to-end Contextual BranchGuard on OBSERVED conflicts (post-hoc, exploratory).

Section 6.3's calibration/deployment study draws channel reliabilities from four
prescribed Bernoulli masses. One objection is that this demonstrates the
estimator, not the system: the numbers that make subgroup masking appear are
chosen rather than measured. This script removes the Bernoulli generator.

Channels here are conflict *domains*, an attribute of the conflict key that a
merger sees before deciding, and every reliability is ESTIMATED from the
benchmark's own adjudicated construction labels on a calibration split:

  objective_fact   locked resolvable conflicts: provenance points at the target
                   with the generator's source-signal noise
  no_signal        equalized source/confidence: provenance carries nothing
  safety_constraint  strict ancestor constraints with provenance favouring the
                   value that violates them: provenance is ANTI-correlated

The mixture is deliberately uneven, so the rare anti-correlated domain is the
subgroup a pooled certificate can hide. Policies are then scored on a disjoint
deployment split, and where a policy defers to the LLM the outcome used is the
model's ACTUAL cached merge decision, not a simulated draw.

Zero new API: every merge call is served from the content-hash cache.

Usage: PYTHONPATH=. python scripts/run_empirical_channel_branchguard.py
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

from branchmem.benchmark.downstream_tasks import generate_downstream_questions
from branchmem.benchmark.misleading_provenance_generator import (
    generate_misleading_provenance_scenarios,
)
from branchmem.benchmark.semantic_resolvable_generator import (
    generate_semantic_resolvable_scenarios,
)
from branchmem.branchguard.contextual import Action, certify_channel, certify_channels
from branchmem.benchmark.scenario_generator import ScenarioConfig, ScenarioGenerator
from branchmem.llm.base import build_backend
from branchmem.merge.base import current_facts_by_key, find_collisions
from branchmem.merge.three_way_llm import ThreeWayLLMMerge

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "final"
MODEL = "gpt-5.4"
LAMBDA = 0.25
DELTA = 0.05
SPLIT_SEED = 90210

# Locked construction, reused so the objective_fact channel is the same
# population the confirmatory study scored.
LOCKED_SPANS = (4.0, 10.0, 20.0)
LOCKED_PER_SPAN = 20
LOCKED_SEED = 2026


def locked_scenarios():
    gen = ScenarioGenerator()
    out = []
    for i, span in enumerate(LOCKED_SPANS):
        out.extend(
            gen.generate(
                LOCKED_PER_SPAN,
                ScenarioConfig(divergence_span=span),
                seed=LOCKED_SEED + i,
            )
        )
    return out


def _contradiction_features(detector, scenario, key, fa, fb) -> dict:
    """NLI contradiction of each branch value against the ancestor value."""
    if detector is None:
        return {}
    anc = current_facts_by_key(scenario.ancestor).get(key)
    if anc is None:
        return {"anc_contradicts_a": None, "anc_contradicts_b": None}
    premise = f"{anc.entity} {anc.value}."
    score = lambda v: detector._predict([(premise, f"{fa.entity} {v}.")])[0]["contradiction"]  # noqa: E731
    return {"anc_contradicts_a": round(score(fa.value), 4),
            "anc_contradicts_b": round(score(fb.value), 4)}


def conflict_rows(scenarios, channel: str, strat, detector=None) -> list[dict]:
    """One row per conflicting key: what provenance would say, what is true,
    and what the LLM actually did (served from cache)."""
    rows: list[dict] = []
    for s in scenarios:
        questions = {(q.entity, q.predicate): q for q in generate_downstream_questions(s)}
        collisions = find_collisions(s.branch_a, s.branch_b)
        if not collisions:
            continue
        merged = strat.merge(s.ancestor, s.branch_a, s.branch_b)
        merged_by_key = {(f.fact.entity, f.fact.predicate): f for f in merged.resulting_facts}

        for key, (fa, fb) in collisions.items():
            q = questions.get(key)
            if q is None or q.category != "resolvable":
                continue  # only keys with a fact of the matter
            target = q.expected_answer

            # What a provenance-following rule commits to: the higher-confidence
            # branch, with ties broken as an abstention.
            if fa.confidence == fb.confidence:
                provenance_value = None
            else:
                provenance_value = (fa if fa.confidence > fb.confidence else fb).value

            res = merged_by_key.get(key)
            llm_abstained = res is None or res.resolution.name == "FLAGGED_UNRESOLVED"
            llm_value = None if llm_abstained else res.fact.value

            rows.append(
                {
                    "channel": channel,
                    "scenario": s.scenario_id,
                    "key": f"{key[0]}|{key[1]}",
                    # Features visible BEFORE deciding, for the channel router of
                    # scripts/run_channel_governance.py.
                    "predicate": key[1],
                    "has_ancestor": key in current_facts_by_key(s.ancestor),
                    "equal_metadata": fa.source == fb.source and fa.confidence == fb.confidence,
                    # Contradiction of each branch value against the ancestor, a
                    # quantity a merger can COMPUTE at decision time rather than
                    # guess from a predicate name. Used by the semantic router of
                    # scripts/run_routing_aware_certification.py.
                    **_contradiction_features(detector, s, key, fa, fb),
                    "provenance_value": provenance_value,
                    "provenance_correct": (
                        None if provenance_value is None else provenance_value == target
                    ),
                    "inverse_value": (
                        None
                        if provenance_value is None
                        else (fb if provenance_value == fa.value else fa).value
                    ),
                    "llm_abstained": llm_abstained,
                    "llm_correct": (None if llm_abstained else llm_value == target),
                }
            )
    return rows


def score(rows, decide) -> dict:
    """decide(row) -> 'follow' | 'invert' | 'abstain' | 'llm'."""
    committed = wrong = 0
    per_channel = defaultdict(lambda: [0, 0])
    for r in rows:
        act = decide(r)
        if act == "abstain":
            continue
        if act == "llm":
            if r["llm_abstained"]:
                continue
            ok = bool(r["llm_correct"])
        elif act == "follow":
            if r["provenance_value"] is None:
                continue
            ok = bool(r["provenance_correct"])
        elif act == "invert":
            if r["provenance_value"] is None:
                continue
            ok = not bool(r["provenance_correct"])
        else:
            raise ValueError(act)
        committed += 1
        wrong += 0 if ok else 1
        per_channel[r["channel"]][0] += 1
        per_channel[r["channel"]][1] += 0 if ok else 1
    risks = {c: (w / n if n else None) for c, (n, w) in per_channel.items()}
    live = [v for v in risks.values() if v is not None]
    return {
        "coverage": committed / len(rows) if rows else 0.0,
        "n_committed": committed,
        "n_wrong": wrong,
        "conditional_risk": wrong / committed if committed else None,
        "per_channel_risk": risks,
        "worst_channel_risk": max(live) if live else None,
        "unsafe_channel_committed": any(v > LAMBDA for v in live),
        # +1 correct, -4 wrong, 0 abstain, on the same scale as Section 6.3
        "expected_utility": ((committed - wrong) - 4 * wrong) / len(rows) if rows else 0.0,
    }


def main() -> None:
    backend = build_backend(
        {
            "backend": "openai_compatible",
            "model": MODEL,
            "cache_dir": "llm_cache",
            "temperature": 0.0,
            "max_tokens": 8000,
        }
    )
    strat = ThreeWayLLMMerge(backend=backend)

    rows: list[dict] = []
    rows += conflict_rows(locked_scenarios(), "objective_fact", strat)
    rows += conflict_rows(
        generate_semantic_resolvable_scenarios(60, seed=3001), "no_signal", strat
    )
    rows += conflict_rows(
        generate_misleading_provenance_scenarios(60, seed=5150), "safety_constraint", strat
    )

    rng = random.Random(SPLIT_SEED)
    rng.shuffle(rows)
    cut = len(rows) // 2
    calib, deploy = rows[:cut], rows[cut:]

    # Per-channel provenance reliability, estimated from adjudicated labels.
    counts: dict[str, tuple[int, int]] = {}
    for channel in sorted({r["channel"] for r in rows}):
        usable = [r for r in calib if r["channel"] == channel and r["provenance_value"] is not None]
        counts[channel] = (sum(1 for r in usable if r["provenance_correct"]), len(usable))

    contextual = certify_channels(counts, abstention_loss=LAMBDA, delta=DELTA)

    pooled_usable = [r for r in calib if r["provenance_value"] is not None]
    pooled = certify_channel(
        "pooled",
        sum(1 for r in pooled_usable if r["provenance_correct"]),
        len(pooled_usable),
        abstention_loss=LAMBDA,
        alpha=DELTA,
    )

    act_name = {Action.FOLLOW: "follow", Action.INVERT: "invert", Action.ABSTAIN: "abstain"}

    policies = {
        "Always follow provenance": lambda r: "follow",
        "LLM merger (observed)": lambda r: "llm",
        "Pooled BranchGuard": lambda r: act_name[pooled.action],
        "Contextual BranchGuard": lambda r: act_name[contextual[r["channel"]].action],
        "Contextual BranchGuard + LLM fallback": lambda r: (
            "llm"
            if contextual[r["channel"]].action is Action.ABSTAIN
            else act_name[contextual[r["channel"]].action]
        ),
    }

    results = {name: score(deploy, fn) for name, fn in policies.items()}

    # Decompose the composite policy: the part the certificate covers vs the
    # part it does not. Reported separately because they carry different
    # warrants, not different numbers of the same warrant.
    certified_channels = {
        c for c, cert in contextual.items() if cert.action is not Action.ABSTAIN
    }
    certified_rows = [r for r in deploy if r["channel"] in certified_channels]
    fallback_rows = [r for r in deploy if r["channel"] not in certified_channels]
    results["  [certified subset only]"] = score(
        certified_rows, lambda r: act_name[contextual[r["channel"]].action]
    )
    results["  [uncertified fallback only]"] = score(fallback_rows, lambda r: "llm")

    print(f"{len(rows)} resolvable conflicts; calibration {len(calib)}, deployment {len(deploy)}")
    print("\nEstimated provenance reliability (calibration split):")
    for channel, cert in contextual.items():
        print(
            f"  {channel:18} n={cert.n:4d}  rho={cert.empirical_reliability:.3f}  "
            f"CI=[{cert.lower:.3f},{cert.upper:.3f}]  -> {cert.action.value}"
        )
    print(
        f"  {'POOLED':18} n={pooled.n:4d}  rho={pooled.empirical_reliability:.3f}  "
        f"CI=[{pooled.lower:.3f},{pooled.upper:.3f}]  -> {pooled.action.value}"
    )

    print("\nDeployment split:")
    hdr = f"{'policy':40} {'cov':>6} {'risk':>7} {'worst':>7} {'util':>7}  unsafe"
    print(hdr)
    for name, r in results.items():
        cr = "  n/a " if r["conditional_risk"] is None else f"{r['conditional_risk']:.3f}"
        wc = "  n/a " if r["worst_channel_risk"] is None else f"{r['worst_channel_risk']:.3f}"
        print(
            f"{name:40} {r['coverage']:.3f} {cr:>7} {wc:>7} "
            f"{r['expected_utility']:7.3f}  {r['unsafe_channel_committed']}"
        )

    payload = {
        "note": (
            "Post-hoc, exploratory (not in ANALYSIS_PLAN.md). End-to-end Contextual BranchGuard "
            "on observed benchmark conflicts rather than prescribed Bernoulli channels. Channels "
            "are conflict domains visible at merge time; per-channel provenance reliability is "
            "ESTIMATED from adjudicated construction labels on a calibration split and the "
            "certified policy is scored on a disjoint deployment split, using the model's actual "
            "cached merge decisions wherever a policy defers to the LLM. Zero new API."
        ),
        "model": MODEL,
        "abstention_loss": LAMBDA,
        "delta": DELTA,
        "split_seed": SPLIT_SEED,
        "n_conflicts": len(rows),
        "n_calibration": len(calib),
        "n_deployment": len(deploy),
        "channel_certificates": {
            c: {
                "n": cert.n,
                "correct": cert.correct,
                "empirical_reliability": cert.empirical_reliability,
                "lower": cert.lower,
                "upper": cert.upper,
                "action": cert.action.value,
            }
            for c, cert in contextual.items()
        },
        "pooled_certificate": {
            "n": pooled.n,
            "correct": pooled.correct,
            "empirical_reliability": pooled.empirical_reliability,
            "lower": pooled.lower,
            "upper": pooled.upper,
            "action": pooled.action.value,
        },
        "certified_channels": sorted(certified_channels),
        "n_deployment_certified": len(certified_rows),
        "n_deployment_uncertified": len(fallback_rows),
        "deployment_results": results,
    }
    (OUT / "empirical_channel_branchguard.json").write_text(json.dumps(payload, indent=2))
    print("\nWrote", OUT / "empirical_channel_branchguard.json")


if __name__ == "__main__":
    main()
