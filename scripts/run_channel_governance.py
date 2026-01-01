#!/usr/bin/env python3
"""How a deployment would define, assign and maintain channels (post-hoc).

Contextual BranchGuard certifies a pre-declared family of channels. Three
operational questions follow, and this script answers each with a measurement
rather than a promise:

A. ASSIGNMENT. What procedure assigns a conflict to a channel before the merge
   decision is made? We implement a label-free router over features visible at
   merge time -- whether the two branches carry equal provenance metadata,
   whether the key has an ancestor value, and whether the predicate name
   matches a declared safety lexicon -- and measure its agreement with the
   hand-declared channels, then certify and deploy under the ROUTER's
   assignment instead of the construction's.

B. MISASSIGNMENT. What does noisy assignment cost? We corrupt the channel of
   each conflict independently with probability eps, re-certify on the
   calibration half, and score on the deployment half against the TRUE
   channels. The certificate still holds for the channels as defined; what
   degrades is the protection the operator actually wanted, and the sweep says
   how fast.

C. COST. What does calibration and audit cost? We compute the minimum audited
   sample per channel needed for a certificate at lambda with Bonferroni over
   K channels, and the steady-state audit volume implied by the drift
   monitor's detection delay at a given audit fraction.

Zero new API: all merge outcomes come from the content-hash cache.

Usage: OPENAI_API_KEY=cache-only-sentinel PYTHONPATH=. python scripts/run_channel_governance.py
"""
from __future__ import annotations

import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from branchmem.benchmark.misleading_provenance_generator import (
    generate_misleading_provenance_scenarios,
)
from branchmem.benchmark.semantic_resolvable_generator import (
    generate_semantic_resolvable_scenarios,
)
from branchmem.branchguard.contextual import (
    Action,
    certify_channels,
    clopper_pearson_interval,
)
from branchmem.llm.base import build_backend
from branchmem.merge.three_way_llm import ThreeWayLLMMerge

import run_empirical_channel_branchguard as ecb

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "final"
LAMBDA = 0.25
DELTA = 0.05
SPLIT_SEED = 90210
NOISE_SEED = 4711
NOISE_DRAWS = 200

# Pre-declared lexicon for the safety_constraint channel. It names predicate
# families a deployment would nominate in advance, and it is fixed before the
# router is evaluated: no label from the data is used to fit it.
SAFETY_LEXICON = re.compile(
    r"allerg|restrict|_rule$|_limit$|limit$|access|requirement|clearance|"
    r"avoid|budget|cap$|_status$|scope|authorization|capacity|interval|"
    r"deadline|validity|tolerance|need$|ability|mode$|method$|schedule|"
    r"count$|practice|observance|covering|activity|exposure|contact|visits|"
    r"ownership|distance|support|comprehension|preference|reaction|response|"
    r"jurisdiction|admission|window$",
    re.I,
)


def route(row: dict) -> str:
    """Assign a channel from features available before the merge decision."""
    if row["equal_metadata"]:
        return "no_signal"
    if row["has_ancestor"] and SAFETY_LEXICON.search(row["predicate"]):
        return "safety_constraint"
    return "objective_fact"


def build_rows(detector=None) -> list[dict]:
    """The 240 observed conflicts. With a detector, each row also carries the
    ancestor-contradiction scores a semantic router needs."""
    backend = build_backend({"backend": "openai_compatible", "model": ecb.MODEL,
                             "cache_dir": "llm_cache", "temperature": 0.0,
                             "max_tokens": 8000})
    strat = ThreeWayLLMMerge(backend=backend)
    rows: list[dict] = []
    rows += ecb.conflict_rows(ecb.locked_scenarios(), "objective_fact", strat, detector)
    rows += ecb.conflict_rows(generate_semantic_resolvable_scenarios(60, seed=3001),
                              "no_signal", strat, detector)
    rows += ecb.conflict_rows(generate_misleading_provenance_scenarios(60, seed=5150),
                              "safety_constraint", strat, detector)
    return rows


def certify_on(calib: list[dict], channel_of) -> dict:
    counts: dict[str, tuple[int, int]] = {}
    for channel in ("objective_fact", "no_signal", "safety_constraint"):
        usable = [r for r in calib
                  if channel_of(r) == channel and r["provenance_value"] is not None]
        counts[channel] = (sum(1 for r in usable if r["provenance_correct"]), len(usable))
    return certify_channels(counts, abstention_loss=LAMBDA, delta=DELTA)


_ACT = {Action.FOLLOW: "follow", Action.INVERT: "invert", Action.ABSTAIN: "abstain"}


def composite(certs: dict, channel_of):
    """Certified action where there is one, LLM fallback where there is not."""
    def decide(row: dict) -> str:
        cert = certs[channel_of(row)]
        return "llm" if cert.action is Action.ABSTAIN else _ACT[cert.action]
    return decide


def score_against_true(rows: list[dict], decide) -> dict:
    """Score a policy, attributing every outcome to the conflict's TRUE channel."""
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
    risks = {c: (w / n if n else None) for c, (n, w) in per_channel.items()}
    live = [v for v in risks.values() if v is not None]
    return {
        "coverage": round(committed / len(rows), 4) if rows else 0.0,
        "n_committed": committed,
        "conditional_risk": round(wrong / committed, 4) if committed else None,
        "per_true_channel_risk": {c: (None if v is None else round(v, 4))
                                  for c, v in risks.items()},
        "worst_true_channel_risk": round(max(live), 4) if live else None,
        "unsafe_true_channel_committed": any(v > LAMBDA for v in live),
        "expected_utility": round(((committed - wrong) - 4 * wrong) / len(rows), 4) if rows else 0.0,
    }


# --------------------------------------------------------------------- part C
def min_audited_n(rho: float, k_channels: int, lam: float = LAMBDA,
                  delta: float = DELTA, n_max: int = 4000) -> int | None:
    """Smallest n whose EXPECTED Clopper-Pearson lower bound reaches 1-lam.

    The audited count is taken as floor(rho*n), the modal outcome, so this is
    the sample a deployment should budget for a channel of true reliability
    rho, not a worst case.
    """
    alpha = delta / k_channels
    for n in range(5, n_max + 1):
        correct = math.floor(rho * n)
        lower, _ = clopper_pearson_interval(correct, n, alpha)
        if lower >= 1.0 - lam:
            return n
    return None


def audit_schedule(detection_delay_conflicts: int, excess_risk: float,
                   fractions=(1.0, 0.5, 0.25, 0.1, 0.05)) -> list[dict]:
    """Audit volume and exposure implied by an audit fraction.

    The e-process needs a roughly fixed number of AUDITED outcomes to revoke.
    Auditing a fraction f of committed conflicts multiplies the delay measured
    in committed conflicts by 1/f, and the extra wrong commitments incurred
    before revocation are that delay times the excess error rate.
    """
    out = []
    for f in fractions:
        conflicts_to_revoke = detection_delay_conflicts / f
        out.append({
            "audit_fraction": f,
            "audited_items_to_revocation": detection_delay_conflicts,
            "committed_conflicts_before_revocation": round(conflicts_to_revoke),
            "excess_wrong_commits_before_revocation": round(conflicts_to_revoke * excess_risk, 1),
        })
    return out


def main() -> None:
    rows = build_rows()
    rng = random.Random(SPLIT_SEED)
    rng.shuffle(rows)
    cut = len(rows) // 2
    calib, deploy = rows[:cut], rows[cut:]
    print(f"{len(rows)} conflicts; calibration {len(calib)}, deployment {len(deploy)}")

    # ---------------------------------------------------------------- part A
    true_of = lambda r: r["channel"]  # noqa: E731
    confusion = Counter((r["channel"], route(r)) for r in rows)
    agree = sum(v for (t, p), v in confusion.items() if t == p) / len(rows)
    print(f"\nA. router agreement with declared channels: {agree:.3f}")
    for (t, p), v in sorted(confusion.items()):
        if t != p:
            print(f"   {t} -> routed {p}: {v}")

    certs_true = certify_on(calib, true_of)
    certs_router = certify_on(calib, route)
    part_a = {
        "router_agreement": round(agree, 4),
        "confusion": {f"{t}->{p}": v for (t, p), v in sorted(confusion.items())},
        "certificates_declared": {c: {"n": x.n, "rho": x.empirical_reliability,
                                      "lower": x.lower, "upper": x.upper,
                                      "action": x.action.value}
                                  for c, x in certs_true.items()},
        "certificates_routed": {c: {"n": x.n, "rho": x.empirical_reliability,
                                    "lower": x.lower, "upper": x.upper,
                                    "action": x.action.value}
                                for c, x in certs_router.items()},
        "deployment_declared": score_against_true(deploy, composite(certs_true, true_of)),
        "deployment_routed": score_against_true(deploy, composite(certs_router, route)),
    }
    print("   certified under declared channels:",
          {c: x.action.value for c, x in certs_true.items()})
    print("   certified under routed channels:  ",
          {c: x.action.value for c, x in certs_router.items()})
    print(f"   deployment risk, declared: {part_a['deployment_declared']['conditional_risk']}, "
          f"routed: {part_a['deployment_routed']['conditional_risk']}")

    # ---------------------------------------------------------------- part B
    channels = ["objective_fact", "no_signal", "safety_constraint"]
    sweep = []
    for eps in (0.0, 0.05, 0.10, 0.20, 0.30, 0.50):
        risks, worst, unsafe, covs, utils, wrong_follow = [], [], 0, [], [], 0
        uncertified_safety = 0
        for draw in range(NOISE_DRAWS if eps else 1):
            nrng = random.Random(f"{NOISE_SEED}:{eps}:{draw}")
            noisy = {}
            for r in rows:
                c = r["channel"]
                if nrng.random() < eps:
                    c = nrng.choice([x for x in channels if x != c])
                noisy[id(r)] = c
            noisy_of = lambda r, m=noisy: m[id(r)]  # noqa: E731
            certs = certify_on(calib, noisy_of)
            s = score_against_true(deploy, composite(certs, noisy_of))
            if s["conditional_risk"] is not None:
                risks.append(s["conditional_risk"])
            if s["worst_true_channel_risk"] is not None:
                worst.append(s["worst_true_channel_risk"])
            unsafe += int(s["unsafe_true_channel_committed"])
            covs.append(s["coverage"])
            utils.append(s["expected_utility"])
            wrong_follow += int(certs["safety_constraint"].action is Action.FOLLOW)
            uncertified_safety += int(certs["safety_constraint"].action is Action.ABSTAIN)
        n_draws = NOISE_DRAWS if eps else 1
        sweep.append({
            "epsilon": eps,
            "draws": n_draws,
            "mean_coverage": round(sum(covs) / len(covs), 4),
            "mean_conditional_risk": round(sum(risks) / len(risks), 4) if risks else None,
            "mean_worst_true_channel_risk": round(sum(worst) / len(worst), 4) if worst else None,
            "p_unsafe_true_channel_committed": round(unsafe / n_draws, 4),
            "p_safety_channel_certified_follow": round(wrong_follow / n_draws, 4),
            "p_safety_channel_uncertified_llm_fallback": round(uncertified_safety / n_draws, 4),
            "mean_expected_utility": round(sum(utils) / len(utils), 4),
        })
    print("\nB. channel-misassignment sweep (deployment, scored on TRUE channels)")
    print(f"{'eps':>6} {'cov':>7} {'risk':>7} {'worst':>7} {'P(unsafe)':>10} {'util':>7}")
    for s in sweep:
        print(f"{s['epsilon']:6.2f} {s['mean_coverage']:7.3f} "
              f"{s['mean_conditional_risk'] if s['mean_conditional_risk'] is not None else float('nan'):7.3f} "
              f"{s['mean_worst_true_channel_risk'] if s['mean_worst_true_channel_risk'] is not None else float('nan'):7.3f} "
              f"{s['p_unsafe_true_channel_committed']:10.3f} {s['mean_expected_utility']:7.3f}")

    # ---------------------------------------------------------------- part C
    calibration_cost = {
        f"K={k}": {f"rho={rho}": min_audited_n(rho, k)
                   for rho in (1.0, 0.98, 0.95, 0.90, 0.85)}
        for k in (3, 4, 8)
    }
    print("\nC. audited items needed per channel to certify at lambda=0.25, delta=0.05")
    print(f"{'channels':>10} " + " ".join(f"{'rho='+str(r):>9}" for r in (1.0, 0.98, 0.95, 0.90, 0.85)))
    for k, row in calibration_cost.items():
        print(f"{k:>10} " + " ".join(
            f"{(v if v is not None else float('inf')):>9}" for v in row.values()))

    # Detection delay and post-drift error rate, both read from the drift
    # experiment's own outputs rather than assumed.
    import csv
    drift_dir = ROOT / "results" / "contextual_branchguard"
    with open(drift_dir / "drift_detection_summary.csv") as f:
        per_channel_delay = {r["channel"]: float(r["median_delay"]) for r in csv.DictReader(f)}
    slowest_channel = max(per_channel_delay, key=per_channel_delay.get)
    median_delay = int(per_channel_delay[slowest_channel])
    with open(drift_dir / "drift_recovery_window.csv") as f:
        window = {r["window"]: float(r["wrong_commits_per_opportunity"])
                  for r in csv.DictReader(f)}
    post_drift_risk = window["first_50_after_drift"]
    excess_risk = round(max(0.0, post_drift_risk - LAMBDA), 4)
    schedule = audit_schedule(median_delay, excess_risk=excess_risk)
    print(f"\n   audit cadence: slowest channel '{slowest_channel}', median detection delay "
          f"{median_delay} audited items; post-drift wrong-commit rate {post_drift_risk:.3f}, "
          f"excess over lambda {excess_risk:.3f}")
    for s in schedule:
        print(f"   f={s['audit_fraction']:<5} -> {s['committed_conflicts_before_revocation']:>7} "
              f"conflicts committed before revocation, "
              f"{s['excess_wrong_commits_before_revocation']:>7} of them wrong")

    payload = {
        "note": (
            "Post-hoc, exploratory (not in ANALYSIS_PLAN.md). Operational governance of the "
            "channel family: (A) a label-free router that assigns channels from features visible "
            "before the merge decision, certified and deployed in place of the construction's own "
            "channel labels; (B) a sweep over independent channel-misassignment probability eps, "
            "re-certifying each draw and scoring the deployment half against the TRUE channels, so "
            "the reported degradation is the protection an operator loses rather than a violation "
            "of the theorem, which continues to hold for the channels as defined; (C) calibration "
            "sample complexity and audit cadence. Zero new API."
        ),
        "lambda": LAMBDA, "delta": DELTA, "split_seed": SPLIT_SEED,
        "n_conflicts": len(rows), "n_calibration": len(calib), "n_deployment": len(deploy),
        "noise_seed": NOISE_SEED, "noise_draws": NOISE_DRAWS,
        "A_assignment": part_a,
        "B_misassignment_sweep": sweep,
        "C_calibration_cost_audited_items": calibration_cost,
        "C_median_detection_delay_audited_items": median_delay,
        "C_slowest_channel": slowest_channel,
        "C_per_channel_median_delay": per_channel_delay,
        "C_post_drift_wrong_commit_rate": post_drift_risk,
        "C_audit_schedule_excess_risk": excess_risk,
        "C_audit_schedule": schedule,
    }
    (OUT / "channel_governance.json").write_text(json.dumps(payload, indent=2))
    print("\nWrote", OUT / "channel_governance.json")


if __name__ == "__main__":
    main()
