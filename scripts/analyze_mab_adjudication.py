"""Analyze completed independent MAB adjudication and held-out routing.

The script intentionally refuses to run on fewer than three annotators per
conflict.  It expects annotation responses mapped back to original A/B labels.
"""
from __future__ import annotations

import argparse
from collections import Counter
from itertools import combinations
from pathlib import Path
import json
import numpy as np
import pandas as pd

from branchmem.branchguard.contextual import Action, bayesian_source_action, certify_channels

LABELS = ("A", "B", "MERGE", "UNRESOLVED")
ITEM_REQUIRED = {"conflict_id", "source_row", "channel", "derived_label", "provenance_choice"}
ANN_REQUIRED = {"conflict_id", "annotator_id", "resolution", "confidence_1_5", "evidence_span"}


def fleiss_kappa(groups: list[list[str]]) -> float:
    n = len(groups)
    m = len(groups[0])
    counts = np.array([[g.count(l) for l in LABELS] for g in groups], dtype=float)
    p_bar = np.mean((np.sum(counts**2, axis=1) - m) / (m * (m - 1)))
    p = counts.sum(axis=0) / (n * m)
    p_e = float(np.sum(p**2))
    return float((p_bar - p_e) / (1 - p_e)) if p_e < 1 else 1.0


def majority(labels: list[str]) -> str:
    c = Counter(labels)
    label, count = c.most_common(1)[0]
    return label if count >= 2 else "NO_MAJORITY"


def policy_metrics(test: pd.DataFrame, actions: dict[str, Action]) -> dict[str, float]:
    commits = wrong = 0
    channel_risks: dict[str, list[int]] = {}
    for r in test.itertuples(index=False):
        action = actions.get(str(r.channel), Action.ABSTAIN)
        if action == Action.ABSTAIN or r.majority_label not in {"A", "B"}:
            continue
        commits += 1
        chosen = r.provenance_choice if action == Action.FOLLOW else ("B" if r.provenance_choice == "A" else "A")
        err = int(chosen != r.majority_label)
        wrong += err
        channel_risks.setdefault(str(r.channel), []).append(err)
    eligible = int(test.majority_label.isin(["A", "B"]).sum())
    return {
        "eligible": eligible,
        "commits": commits,
        "coverage": commits / eligible if eligible else float("nan"),
        "wrong": wrong,
        "conditional_risk": wrong / commits if commits else float("nan"),
        "worst_empirical_channel_risk": max((np.mean(v) for v in channel_risks.values()), default=float("nan")),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True)
    ap.add_argument("--annotations", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--delta", type=float, default=0.05)
    ap.add_argument("--abstention-loss", type=float, default=0.25)
    args = ap.parse_args()
    items = pd.read_csv(args.items, keep_default_na=False)
    anns = pd.read_csv(args.annotations, keep_default_na=False)
    for name, df, req in (("items", items, ITEM_REQUIRED), ("annotations", anns, ANN_REQUIRED)):
        miss = req - set(df.columns)
        if miss:
            raise SystemExit(f"{name} missing columns: {sorted(miss)}")
    anns.resolution = anns.resolution.str.upper()
    bad = sorted(set(anns.resolution) - set(LABELS))
    if bad:
        raise SystemExit(f"Unknown resolutions: {bad}")
    counts = anns.groupby("conflict_id").annotator_id.nunique()
    if (counts < 3).any():
        raise SystemExit("Every conflict needs at least three independent annotators")
    groups = anns.groupby("conflict_id").resolution.apply(list)
    if groups.map(len).nunique() != 1:
        raise SystemExit("Use the same number of annotators for every conflict")
    adjud = groups.map(majority).rename("majority_label").reset_index()
    merged = items.merge(adjud, on="conflict_id", validate="one_to_one")
    summary = {
        "n_items": int(len(merged)),
        "n_annotators": int(groups.map(len).iloc[0]),
        "fleiss_kappa": fleiss_kappa(groups.tolist()),
        "unanimity": float(np.mean([len(set(g)) == 1 for g in groups])),
        "majority_resolvable_rate": float(merged.majority_label.isin(["A", "B"]).mean()),
        "derived_label_agreement": float((merged.majority_label == merged.derived_label).mean()),
        "no_majority": int((merged.majority_label == "NO_MAJORITY").sum()),
    }
    rows = sorted(merged.source_row.astype(str).unique())
    fold_records = []
    for heldout in rows:
        cal = merged[merged.source_row.astype(str) != heldout]
        test = merged[merged.source_row.astype(str) == heldout]
        cal = cal[cal.majority_label.isin(["A", "B"]) & cal.provenance_choice.isin(["A", "B"])]
        channel_counts = {}
        for channel, part in cal.groupby("channel"):
            correct = int((part.majority_label == part.provenance_choice).sum())
            channel_counts[str(channel)] = (correct, int(len(part)))
        if not channel_counts:
            continue
        certs = certify_channels(channel_counts, abstention_loss=args.abstention_loss, delta=args.delta)
        bg_actions = {c: cert.action for c, cert in certs.items()}
        bayes_actions = {c: bayesian_source_action(k, n, abstention_loss=args.abstention_loss) for c, (k, n) in channel_counts.items()}
        for policy, actions in (("Contextual BranchGuard", bg_actions), ("Contextual BayesianSource", bayes_actions)):
            fold_records.append({"heldout_source_row": heldout, "policy": policy, **policy_metrics(test, actions)})
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "adjudication_summary.json").write_text(json.dumps(summary, indent=2))
    merged.to_csv(out / "majority_labels.csv", index=False)
    pd.DataFrame(fold_records).to_csv(out / "source_row_heldout_policy_metrics.csv", index=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
