"""End-to-end evaluation of pooled, contextual, and monitored BranchGuard.

The experiment uses heterogeneous provenance channels, a held-out deployment
history, calibration-size sweeps, abrupt source drift, and a direct comparison
with BayesianSourceMerge.  It produces all CSVs and the main figure used in the
revised manuscript.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from branchmem.branchguard.contextual import (
    Action,
    AnytimeRiskMonitor,
    BlockwiseRiskMonitor,
    bayesian_source_action,
    certify_channel,
    certify_channels,
)

ROOT = Path(__file__).resolve().parents[1]
# Certified conditional-risk level. The utility scale used in the reporting
# tables (+1 correct, -4 wrong, 0 abstain) implies a utility-optimal commit
# threshold of 0.2; set BRANCHGUARD_LAMBDA=0.20 to run that sensitivity.
LAMBDA = float(os.environ.get("BRANCHGUARD_LAMBDA", "0.25"))
_SUFFIX = "" if abs(LAMBDA - 0.25) < 1e-12 else f"_lambda{LAMBDA:g}"

OUT = ROOT / "results" / f"contextual_branchguard{_SUFFIX}"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 20260728
RNG = np.random.default_rng(SEED)
DELTA = 0.05
CHANNELS = ("preference", "contact", "medical", "inference")
WEIGHTS = np.array([0.50, 0.30, 0.05, 0.15])
RHO_CAL = np.array([0.90, 0.95, 0.20, 0.10])
# Abrupt deployment shift: two previously informative channels become unsafe.
RHO_DRIFT = np.array([0.30, 0.40, 0.20, 0.10])
CAL_SIZES = (50, 100, 200, 400, 800, 1600)
REPS = 2500
TEST_SIZE = 5000


@dataclass(frozen=True)
class Metrics:
    coverage: float
    wrong_commit_rate: float
    conditional_risk: float
    expected_utility: float
    worst_committed_channel_risk: float
    medical_risk: float


def action_error(action: Action, rho: float) -> float:
    if action == Action.FOLLOW:
        return 1.0 - rho
    if action == Action.INVERT:
        return rho
    return 0.0


def metrics_for_history(
    actions: list[Action],
    counts: np.ndarray,
    provenance_correct: np.ndarray,
    rhos: np.ndarray,
) -> Metrics:
    """Evaluate actions on one paired held-out deployment history.

    ``provenance_correct[c]`` is the number of test conflicts in channel ``c``
    for which the provenance-favoured branch is the adjudicated target.  Every
    policy sees the same history, so policy contrasts are paired.  The reported
    worst-channel quantity uses the channel's true deployment risk because the
    certificate controls expected channel risk, not a noisy test estimate.
    """
    committed = np.array([a != Action.ABSTAIN for a in actions], dtype=bool)
    errors = np.zeros(len(CHANNELS), dtype=int)
    for i, action in enumerate(actions):
        if action == Action.FOLLOW:
            errors[i] = int(counts[i] - provenance_correct[i])
        elif action == Action.INVERT:
            errors[i] = int(provenance_correct[i])
    commits = int(counts[committed].sum())
    wrong = int(errors.sum())
    total = int(counts.sum())
    coverage = commits / total
    wrong_rate = wrong / total
    conditional = wrong / commits if commits else math.nan
    correct_commits = commits - wrong
    utility = (correct_commits - 4.0 * wrong) / total
    true_risks = np.array([action_error(a, float(r)) for a, r in zip(actions, rhos)])
    worst = float(np.max(true_risks[committed])) if np.any(committed) else 0.0
    medical = float(true_risks[2]) if committed[2] else 0.0
    return Metrics(coverage, wrong_rate, conditional, utility, worst, medical)


@lru_cache(maxsize=None)
def cp_action(correct: int, n: int, alpha: float = DELTA / len(CHANNELS), drift_budget: float = 0.0) -> Action:
    if n == 0:
        return Action.ABSTAIN
    return certify_channel(
        "cached", correct, n,
        abstention_loss=LAMBDA,
        alpha=alpha,
        drift_budget=drift_budget,
    ).action


def contextual_actions(correct: np.ndarray, counts: np.ndarray, drift_budget: float = 0.0) -> list[Action]:
    return [
        cp_action(int(k), int(n), DELTA / len(CHANNELS), drift_budget)
        for k, n in zip(correct, counts)
    ]


def pooled_action(correct: np.ndarray, counts: np.ndarray, drift_budget: float = 0.0) -> Action:
    return cp_action(int(correct.sum()), int(counts.sum()), DELTA, drift_budget)


def bayesian_actions(correct: np.ndarray, counts: np.ndarray) -> list[Action]:
    return [bayesian_source_action(int(k), int(n), abstention_loss=LAMBDA) for k, n in zip(correct, counts)]


def pooled_bayes_action(correct: np.ndarray, counts: np.ndarray) -> Action:
    return bayesian_source_action(int(correct.sum()), int(counts.sum()), abstention_loss=LAMBDA)


def sample_calibration(m: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    counts = rng.multinomial(m, WEIGHTS)
    correct = rng.binomial(counts, RHO_CAL)
    return correct, counts


def run_size_sweep() -> pd.DataFrame:
    records: list[dict] = []
    oracle_actions = [Action.FOLLOW, Action.FOLLOW, Action.INVERT, Action.INVERT]
    always_follow = [Action.FOLLOW] * len(CHANNELS)
    always_abstain = [Action.ABSTAIN] * len(CHANNELS)

    for m in CAL_SIZES:
        for rep in range(REPS):
            correct, counts = sample_calibration(m, RNG)
            policies = {
                "Contextual BranchGuard": contextual_actions(correct, counts),
                "Pooled BranchGuard": [pooled_action(correct, counts)] * len(CHANNELS),
                "Contextual BayesianSource": bayesian_actions(correct, counts),
                "Pooled BayesianSource": [pooled_bayes_action(correct, counts)] * len(CHANNELS),
                "Always follow provenance": always_follow,
                "Oracle contextual": oracle_actions,
                "Always abstain": always_abstain,
            }
            for regime, rhos in (("stationary", RHO_CAL), ("abrupt_drift", RHO_DRIFT)):
                test_counts = RNG.multinomial(TEST_SIZE, WEIGHTS)
                test_correct = RNG.binomial(test_counts, rhos)
                for policy, actions in policies.items():
                    met = metrics_for_history(actions, test_counts, test_correct, rhos)
                    records.append({
                        "calibration_size": m,
                        "rep": rep,
                        "regime": regime,
                        "policy": policy,
                        **met.__dict__,
                        "risk_violation": float(
                            met.conditional_risk > LAMBDA
                            if not math.isnan(met.conditional_risk) else False
                        ),
                        "subgroup_violation": float(met.worst_committed_channel_risk > LAMBDA),
                    })
    df = pd.DataFrame.from_records(records)
    df.to_csv(OUT / "size_sweep_raw.csv", index=False)
    summary = (
        df.groupby(["calibration_size", "regime", "policy"], as_index=False)
        .agg(
            coverage=("coverage", "mean"),
            wrong_commit_rate=("wrong_commit_rate", "mean"),
            conditional_risk=("conditional_risk", "mean"),
            expected_utility=("expected_utility", "mean"),
            worst_channel_risk=("worst_committed_channel_risk", "mean"),
            medical_risk=("medical_risk", "mean"),
            risk_violation_probability=("risk_violation", "mean"),
            subgroup_violation_probability=("subgroup_violation", "mean"),
        )
    )
    summary.to_csv(OUT / "size_sweep_summary.csv", index=False)
    return summary


def run_drift_monitor(reps: int = 500, horizon: int = 3000, drift_at: int = 1000, audit_rate: float = 0.20) -> tuple[pd.DataFrame, pd.DataFrame]:
    binsize = 50
    bin_records: list[dict] = []
    delay_records: list[dict] = []
    n_bins = horizon // binsize

    for rep in range(reps):
        correct, counts = sample_calibration(800, RNG)
        initial = contextual_actions(correct, counts)
        static_actions = list(initial)
        monitored_actions = list(initial)
        monitors = {
            i: BlockwiseRiskMonitor(
                target_risk=LAMBDA,
                alt_risk=0.55,
                total_alpha=DELTA / len(CHANNELS),
                min_block_size=20,
            )
            for i, a in enumerate(initial) if a != Action.ABSTAIN
        }
        revoke_time: dict[int, int] = {}
        stats = {
            "Static contextual": np.zeros((n_bins, 3), dtype=float),
            "Monitored contextual": np.zeros((n_bins, 3), dtype=float),
        }

        for t in range(horizon):
            c = int(RNG.choice(len(CHANNELS), p=WEIGHTS))
            rho = RHO_CAL[c] if t < drift_at else RHO_DRIFT[c]
            provenance_correct = bool(RNG.random() < rho)
            b = t // binsize

            for name, actions in (("Static contextual", static_actions), ("Monitored contextual", monitored_actions)):
                action = actions[c]
                if action == Action.ABSTAIN:
                    continue
                error = (not provenance_correct) if action == Action.FOLLOW else provenance_correct
                stats[name][b, 0] += 1.0  # commits
                stats[name][b, 1] += float(error)
                stats[name][b, 2] += 1.0  # opportunities seen

            action = monitored_actions[c]
            if action != Action.ABSTAIN and RNG.random() < audit_rate:
                error = (not provenance_correct) if action == Action.FOLLOW else provenance_correct
                if monitors[c].update(error):
                    monitored_actions[c] = Action.ABSTAIN
                    revoke_time.setdefault(c, t)

        for name, arr in stats.items():
            for b in range(n_bins):
                commits, errors, _ = arr[b]
                bin_records.append({
                    "rep": rep,
                    "policy": name,
                    "time": b * binsize + binsize / 2,
                    "coverage": commits / binsize,
                    "wrong_commit_rate": errors / binsize,
                    "conditional_risk": errors / commits if commits > 0 else math.nan,
                })
        for c in range(len(CHANNELS)):
            delay_records.append({
                "rep": rep,
                "channel": CHANNELS[c],
                "initial_action": initial[c].value,
                "revoke_time": revoke_time.get(c, math.nan),
                "delay_after_drift": (
                    revoke_time[c] - drift_at if c in revoke_time and revoke_time[c] >= drift_at else math.nan
                ),
            })

    bins = pd.DataFrame(bin_records)
    delays = pd.DataFrame(delay_records)
    bins.to_csv(OUT / "drift_monitor_bins.csv", index=False)
    delays.to_csv(OUT / "drift_monitor_delays.csv", index=False)
    return bins, delays


def make_figure(summary: pd.DataFrame, bins: pd.DataFrame) -> None:
    policies = [
        "Contextual BranchGuard",
        "Pooled BranchGuard",
        "Contextual BayesianSource",
        "Always follow provenance",
        "Oracle contextual",
    ]
    stationary = summary[summary.regime == "stationary"]

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.25))
    for policy in policies:
        part = stationary[stationary.policy == policy].sort_values("calibration_size")
        axes[0].plot(part.calibration_size, part.coverage, marker="o", label=policy)
        axes[1].plot(part.calibration_size, part.subgroup_violation_probability, marker="o", label=policy)
    axes[0].set_xscale("log", base=2)
    axes[0].set_ylim(-0.03, 1.03)
    axes[0].set_xlabel("Adjudicated calibration conflicts")
    axes[0].set_ylabel("Commit coverage")
    axes[0].set_title("(a) Heterogeneous stationary channels")

    axes[1].set_xscale("log", base=2)
    axes[1].set_ylim(-0.03, 1.03)
    axes[1].set_xlabel("Adjudicated calibration conflicts")
    axes[1].set_ylabel("P(any committed channel risk > 0.25)")
    axes[1].set_title("(b) Pooled safety can mask a subgroup")

    drift_mean = bins.groupby(["policy", "time"], as_index=False).agg(
        wrong_commit_rate=("wrong_commit_rate", "mean"),
        coverage=("coverage", "mean"),
    )
    for policy in ("Static contextual", "Monitored contextual"):
        part = drift_mean[drift_mean.policy == policy]
        axes[2].plot(part.time, part.wrong_commit_rate, label=f"{policy}: wrong commits")
    axes[2].axvline(1000, linestyle="--", linewidth=1)
    axes[2].axhline(LAMBDA, linestyle=":", linewidth=1)
    axes[2].set_xlabel("Deployment conflict index")
    axes[2].set_ylabel("Wrong commits per opportunity")
    axes[2].set_title("(c) Abrupt drift and anytime revocation")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.10), frameon=False)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.savefig(ROOT / f"contextual_branchguard{_SUFFIX}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_key_results(summary: pd.DataFrame, delays: pd.DataFrame) -> None:
    key = summary[(summary.calibration_size == 800)].copy()
    key.to_csv(OUT / "key_results_m800.csv", index=False)
    shifted = delays[(delays.channel.isin(["preference", "contact"])) & delays.delay_after_drift.notna()]
    delay_summary = shifted.groupby("channel", as_index=False).agg(
        median_delay=("delay_after_drift", "median"),
        p90_delay=("delay_after_drift", lambda x: float(np.quantile(x, 0.9))),
        detected_fraction=("delay_after_drift", "count"),
    )
    denom = delays[delays.channel.isin(["preference", "contact"])].groupby("channel").size()
    delay_summary["detected_fraction"] = delay_summary.apply(
        lambda r: r.detected_fraction / denom.loc[r.channel], axis=1
    )
    delay_summary.to_csv(OUT / "drift_detection_summary.csv", index=False)

    # Recovery window: the monitored policy's behaviour immediately after the
    # shift at conflict 1000 and over the final 500 conflicts of the horizon.
    mon = bins[bins.policy == "Monitored contextual"]
    windows = {
        "first_50_after_drift": mon[(mon.time > 1000) & (mon.time < 1050)],
        "final_500": mon[mon.time >= 2500],
    }
    rows = [
        {
            "window": name,
            "wrong_commits_per_opportunity": float(w.wrong_commit_rate.mean()),
            "coverage": float(w.coverage.mean()),
        }
        for name, w in windows.items()
    ]
    pd.DataFrame(rows).to_csv(OUT / "drift_recovery_window.csv", index=False)


if __name__ == "__main__":
    summary = run_size_sweep()
    bins, delays = run_drift_monitor()
    make_figure(summary, bins)
    write_key_results(summary, delays)
    print(summary[(summary.calibration_size == 800) & (summary.regime == "stationary")][
        ["policy", "coverage", "conditional_risk", "subgroup_violation_probability", "expected_utility"]
    ].to_string(index=False))
    print("\nDrift detection:\n", pd.read_csv(OUT / "drift_detection_summary.csv").to_string(index=False))
    print("\nDrift recovery window:\n", pd.read_csv(OUT / "drift_recovery_window.csv").to_string(index=False))
