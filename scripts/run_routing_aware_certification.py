#!/usr/bin/env python3
"""Certifying the router together with the merger (post-hoc, exploratory).

Appendix B.3 showed that a channel certificate is only as good as the channel
assignment: a router that is 92.1% accurate loses the protection entirely,
because the misassigned minority makes the anti-correlated channel
uncertifiable and its conflicts fall through to an uncertified model. That is a
diagnosis, not a remedy. This script asks how much certified coverage can be
kept when the assignment itself is not trusted.

Three policies, all deployed on the same held-out half:

  oracle        channels as declared by the construction. Upper reference; a
                deployment does not have this.
  predicted     the router's single best guess, which is what Appendix B.3
                measured.
  conservative  certification over a CANDIDATE SET. The router's decision rests
                on three features; two of them (whether the two branches carry
                identical metadata, and whether the key has an ancestor value)
                are read off the conflict exactly, while the third (whether the
                predicate name matches a declared safety lexicon) is the guess.
                So instead of trusting that guess we take the set of channels
                consistent with the exact features, and commit only when EVERY
                channel in that set is certified to the same action. Otherwise
                we abstain, or escalate to the uncertified model, reported
                separately.

What this buys, stated precisely. The certificate no longer needs the
assumption "the router assigns the true channel", which is false at a measured
rate of 7.9%. It needs the weaker assumption "the candidate set contains the
true channel", which this script measures directly. Risk is then bounded by the
worst certified channel in the set, so a committed conflict is covered whatever
its true channel is, as long as the set contains it.

Zero new API: every merge outcome is cache-served.

Usage: OPENAI_API_KEY=cache-only-sentinel PYTHONPATH=. python scripts/run_routing_aware_certification.py
"""
from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from branchmem.branchguard.contextual import Action

import run_channel_governance as gov
from branchmem.conflict.nli_detector import NLIConflictDetector

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "final"
LAMBDA = gov.LAMBDA
CHANNELS = ("objective_fact", "no_signal", "safety_constraint")
DELTA_PER_CHANNEL = gov.DELTA / len(CHANNELS)


def candidates(row: dict) -> tuple[str, ...]:
    """Channels consistent with the features the router reads EXACTLY.

    `equal_metadata` and `has_ancestor` are properties of the conflict, not
    predictions. The safety lexicon over predicate names is the guess, so where
    it is the only thing separating two channels we keep both.
    """
    if row["equal_metadata"]:
        return ("no_signal",)
    if not row["has_ancestor"]:
        return ("objective_fact",)
    return ("safety_constraint", "objective_fact")


# Thresholds are those of ConstraintVetoMerge, fixed before this experiment.
VETO, CLEAR = 0.60, 0.30


def route_semantic(row: dict) -> str:
    """Route on a COMPUTED signal: does the ancestor contradict exactly one of
    the two branch values? That is the thing `safety_constraint` actually means,
    and unlike a predicate-name lexicon it is available at decision time."""
    if row["equal_metadata"]:
        return "no_signal"
    ca, cb = row.get("anc_contradicts_a"), row.get("anc_contradicts_b")
    if ca is None or cb is None:
        return "objective_fact"
    if (ca >= VETO and cb <= CLEAR) or (cb >= VETO and ca <= CLEAR):
        return "safety_constraint"
    return "objective_fact"


def candidates_semantic(row: dict) -> tuple[str, ...]:
    """Candidate set for the semantic router.

    The decisive quantity is continuous, so the set is a single channel when the
    two contradiction scores are clearly separated and both channels when they
    land in the undecided band.
    """
    if row["equal_metadata"]:
        return ("no_signal",)
    ca, cb = row.get("anc_contradicts_a"), row.get("anc_contradicts_b")
    if ca is None or cb is None:
        return ("objective_fact",)
    violated = (ca >= VETO and cb <= CLEAR) or (cb >= VETO and ca <= CLEAR)
    clearly_not = max(ca, cb) < CLEAR          # neither value contradicts anything
    if violated:
        return ("safety_constraint",)
    if clearly_not:
        return ("objective_fact",)
    return ("safety_constraint", "objective_fact")


def conservative_action(certs: dict, row: dict, cand_fn=None) -> tuple[str, tuple[str, ...]]:
    """Commit only when every candidate channel agrees on the action."""
    cand = (cand_fn or candidates)(row)
    actions = {certs[c].action for c in cand}
    if len(actions) == 1:
        act = actions.pop()
        return ({Action.FOLLOW: "follow", Action.INVERT: "invert",
                 Action.ABSTAIN: "abstain"}[act], cand)
    return "abstain", cand


def score(rows: list[dict], decide) -> dict:
    committed = wrong = 0
    per_channel: dict[str, list[int]] = defaultdict(lambda: [0, 0])
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
        else:
            if r["provenance_value"] is None:
                continue
            ok = not bool(r["provenance_correct"])
        committed += 1
        wrong += 0 if ok else 1
        per_channel[r["channel"]][0] += 1
        per_channel[r["channel"]][1] += 0 if ok else 1
    risks = {c: round(w / n, 4) for c, (n, w) in per_channel.items() if n}
    return {
        "coverage": round(committed / len(rows), 4) if rows else None,
        "n_committed": committed,
        "conditional_risk": round(wrong / committed, 4) if committed else None,
        "per_true_channel_risk": risks,
        "worst_true_channel_risk": max(risks.values()) if risks else None,
        "unsafe_true_channel_committed": any(v > LAMBDA for v in risks.values()),
        "expected_utility": round(((committed - wrong) - 4 * wrong) / len(rows), 4) if rows else None,
    }


def main() -> None:
    detector = NLIConflictDetector()
    rows = gov.build_rows(detector=detector)
    rng = random.Random(gov.SPLIT_SEED)
    rng.shuffle(rows)
    cut = len(rows) // 2
    calib, deploy = rows[:cut], rows[cut:]
    print(f"{len(rows)} conflicts; calibration {len(calib)}, deployment {len(deploy)}")

    true_of = lambda r: r["channel"]            # noqa: E731
    certs_oracle = gov.certify_on(calib, true_of)
    certs_router = gov.certify_on(calib, gov.route)
    certs_semantic = gov.certify_on(calib, route_semantic)

    # The assumption the conservative policy actually needs, measured.
    diag = {}
    for tag, route_fn, cand_fn in (("lexicon", gov.route, candidates),
                                   ("semantic", route_semantic, candidates_semantic)):
        sizes = defaultdict(int)
        for r in rows:
            sizes[len(cand_fn(r))] += 1
        diag[tag] = {
            "top1_accuracy": round(sum(1 for r in rows if route_fn(r) == r["channel"]) / len(rows), 4),
            "candidate_set_coverage": round(
                sum(1 for r in rows if r["channel"] in cand_fn(r)) / len(rows), 4),
            "candidate_set_sizes": dict(sizes),
            "mean_candidate_set_size": round(
                sum(len(cand_fn(r)) for r in rows) / len(rows), 4),
        }
        print(f"\n{tag} router: top-1 {diag[tag]['top1_accuracy']:.4f}  "
              f"candidate-set coverage {diag[tag]['candidate_set_coverage']:.4f}  "
              f"mean set size {diag[tag]['mean_candidate_set_size']:.2f}")

    _ACT = {Action.FOLLOW: "follow", Action.INVERT: "invert", Action.ABSTAIN: "abstain"}
    policies = {
        "oracle channels, certified only":
            lambda r: _ACT[certs_oracle[true_of(r)].action],
        "oracle channels + LLM fallback":
            lambda r: ("llm" if certs_oracle[true_of(r)].action is Action.ABSTAIN
                       else _ACT[certs_oracle[true_of(r)].action]),
        "predicted channel, certified only":
            lambda r: _ACT[certs_router[gov.route(r)].action],
        "predicted channel + LLM fallback":
            lambda r: ("llm" if certs_router[gov.route(r)].action is Action.ABSTAIN
                       else _ACT[certs_router[gov.route(r)].action]),
        "conservative (candidate set), certified only":
            lambda r: conservative_action(certs_router, r)[0],
        "conservative + LLM fallback":
            lambda r: ("llm" if conservative_action(certs_router, r)[0] == "abstain"
                       else conservative_action(certs_router, r)[0]),
        "conservative over ORACLE certificates":
            lambda r: conservative_action(certs_oracle, r)[0],
        "semantic router, certified only":
            lambda r: _ACT[certs_semantic[route_semantic(r)].action],
        "semantic router + LLM fallback":
            lambda r: ("llm" if certs_semantic[route_semantic(r)].action is Action.ABSTAIN
                       else _ACT[certs_semantic[route_semantic(r)].action]),
        "semantic conservative (candidate set), certified only":
            lambda r: conservative_action(certs_semantic, r, candidates_semantic)[0],
        "semantic conservative + LLM fallback":
            lambda r: ("llm" if conservative_action(
                certs_semantic, r, candidates_semantic)[0] == "abstain"
                else conservative_action(certs_semantic, r, candidates_semantic)[0]),
        # The deployment-realistic family. Theorem 3 already assumes the
        # calibration conflicts are independently ADJUDICATED, so their channels
        # are known; what a deployment lacks is the channel of a NEW conflict.
        # These policies therefore certify on clean channels and route only at
        # deployment time.
        "clean certs, lexicon routing at deployment":
            lambda r: _ACT[certs_oracle[gov.route(r)].action],
        "clean certs, semantic routing at deployment":
            lambda r: _ACT[certs_oracle[route_semantic(r)].action],
        "clean certs, conservative lexicon candidate sets":
            lambda r: conservative_action(certs_oracle, r, candidates)[0],
        "clean certs, conservative semantic candidate sets":
            lambda r: conservative_action(certs_oracle, r, candidates_semantic)[0],
        "clean certs, conservative semantic + LLM fallback":
            lambda r: ("llm" if conservative_action(
                certs_oracle, r, candidates_semantic)[0] == "abstain"
                else conservative_action(certs_oracle, r, candidates_semantic)[0]),
    }
    results = {name: score(deploy, fn) for name, fn in policies.items()}

    print(f"\n{'policy':46} {'cov':>6} {'risk':>7} {'worst':>7} {'util':>7}  unsafe")
    for name, r in results.items():
        cr = "  n/a " if r["conditional_risk"] is None else f"{r['conditional_risk']:.3f}"
        wc = "  n/a " if r["worst_true_channel_risk"] is None else f"{r['worst_true_channel_risk']:.3f}"
        print(f"{name:46} {r['coverage']:.3f} {cr:>7} {wc:>7} "
              f"{r['expected_utility']:7.3f}  {r['unsafe_true_channel_committed']}")

    # How pure must a routed channel be for an INVERT certificate to survive?
    # The certificate needs the exact upper bound to stay at or below lambda, so
    # any contamination by a channel where provenance is right lifts it. This is
    # the design requirement the sweep above expresses empirically.
    from branchmem.branchguard.contextual import clopper_pearson_interval
    import math
    purity = {}
    for n in (30, 60, 120, 240, 480):
        worst = None
        for k in range(0, n + 1):
            _, upper = clopper_pearson_interval(k, n, DELTA_PER_CHANNEL)
            if upper <= LAMBDA:
                worst = k
            else:
                break
        # k contaminating items each contribute a "provenance was right" outcome
        purity[f"n={n}"] = {
            "max_correct_outcomes_tolerated": worst,
            "max_contamination_rate": None if worst is None else round(worst / n, 4),
        }
    print("\nmaximum contamination of an inverted channel that still certifies "
          f"(lambda={LAMBDA}, per-channel alpha={DELTA_PER_CHANNEL:.4f}):")
    for k, v in purity.items():
        print(f"  {k:7} {v['max_correct_outcomes_tolerated']} of {k.split('=')[1]} "
              f"outcomes = {v['max_contamination_rate']}")

    payload = {
        "note": (
            "Post-hoc, exploratory (not in ANALYSIS_PLAN.md). Routing-aware certification. The "
            "conservative policy certifies over the SET of channels consistent with the two "
            "features a router reads exactly (metadata equality, ancestor presence), treating "
            "the predicate-lexicon match as the guess it is, and commits only where every "
            "candidate channel carries the same certified action. It therefore replaces the "
            "assumption 'the router assigns the true channel', false here at 7.9%, with the "
            "weaker 'the candidate set contains the true channel', whose rate is reported "
            "below. Zero new API."
        ),
        "lambda": LAMBDA, "split_seed": gov.SPLIT_SEED,
        "n_conflicts": len(rows), "n_calibration": len(calib), "n_deployment": len(deploy),
        "router_diagnostics": diag,
        "inverted_channel_purity_requirement": purity,
        "veto_thresholds": {"veto": VETO, "clear": CLEAR},
        "certificates_oracle": {c: {"n": x.n, "action": x.action.value,
                                    "lower": round(x.lower, 4), "upper": round(x.upper, 4)}
                                for c, x in certs_oracle.items()},
        "certificates_router": {c: {"n": x.n, "action": x.action.value,
                                    "lower": round(x.lower, 4), "upper": round(x.upper, 4)}
                                for c, x in certs_router.items()},
        "certificates_semantic_router": {c: {"n": x.n, "action": x.action.value,
                                             "lower": round(x.lower, 4), "upper": round(x.upper, 4)}
                                         for c, x in certs_semantic.items()},
        "deployment_results": results,
    }
    (OUT / "routing_aware_certification.json").write_text(json.dumps(payload, indent=2))
    print("\nWrote", OUT / "routing_aware_certification.json")


if __name__ == "__main__":
    main()
