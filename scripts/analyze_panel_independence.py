#!/usr/bin/env python3
"""Adjudication summary with the vendor-correlation question answered directly.

The earlier four-member panel had three OpenAI models in it, so its agreement
overstated independence. This script reports the same quantities separately for
three tiers:

  openai        ann1-ann3  gpt-5.4 / -mini / -nano
  independent   ann4-ann6  DeBERTa-v3 NLI, Llama 3.1 8B (local), RoBERTa NLI
  all           ann1-ann6

The `independent` tier shares no vendor, no weights and no training data with
the `openai` tier, and is the size the released human protocol asks for. Where
the two tiers disagree, the disagreement is reported rather than pooled away.

A tier reaches a VERDICT on an item when strictly more than half its members
choose the same branch; abstentions never form a verdict. Items without a
verdict carry to the next pass, which shows the adjudicator more evidence.

Usage:
  PYTHONPATH=. python scripts/analyze_panel_independence.py \
      --pass 6:results/final/mab_adjudication_panel6.csv \
      --pass 30:results/final/mab_adjudication_panel6_wide.csv \
      --items annotation/mab_item_master.csv \
      --out results/final/mab_adjudication_summary.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from branchmem.branchguard.contextual import clopper_pearson_interval

LABELS = ("branch_a", "branch_b", "ambiguous")
TIERS = {
    "openai": ["ann1", "ann2", "ann3"],
    "independent": ["ann4", "ann5", "ann6"],
    "all": ["ann1", "ann2", "ann3", "ann4", "ann5", "ann6"],
}


def fleiss_kappa(groups: list[list[str]]) -> float:
    if not groups:
        return float("nan")
    m = len(groups[0])
    if m < 2:
        return float("nan")
    counts = np.array([[g.count(l) for l in LABELS] for g in groups], dtype=float)
    p_bar = float(np.mean((np.sum(counts ** 2, axis=1) - m) / (m * (m - 1))))
    p = counts.sum(axis=0) / (len(groups) * m)
    p_e = float(np.sum(p ** 2))
    return float((p_bar - p_e) / (1 - p_e)) if p_e < 1 else 1.0


def pairwise_agreement(groups: list[list[str]]) -> float:
    """Mean agreement over all rater pairs, including abstentions as a label.

    Reported next to Fleiss' kappa because kappa is deflated when panel members
    have very different abstention propensities, which is exactly the case for
    the independent tier.
    """
    if not groups:
        return float("nan")
    hits = total = 0
    for g in groups:
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                total += 1
                hits += int(g[i] == g[j])
    return hits / total if total else float("nan")


def verdict(choices: list[str]) -> str | None:
    """Strict majority on a non-abstention choice, else no verdict."""
    counts = Counter(c for c in choices if c != "ambiguous")
    if not counts:
        return None
    top, n = counts.most_common(1)[0]
    return top if n * 2 > len(choices) else None


def tier_pass(df: pd.DataFrame, members: list[str]) -> dict[str, dict]:
    """Per-item verdict and raw choices for one tier within one pass."""
    sub = df[df.annotator_id.isin(members)]
    out: dict[str, dict] = {}
    for conflict_id, part in sub.groupby("conflict_id"):
        if part.annotator_id.nunique() != len(members):
            continue                       # incomplete: not adjudicated by this tier
        choices = [part[part.annotator_id == m].branch_choice.iloc[0] for m in members]
        out[str(conflict_id)] = {"choices": choices, "verdict": verdict(choices)}
    return out


def agreement(verdicts: dict[str, str], derived: dict[str, str]) -> dict:
    pairs = [(v, derived[k]) for k, v in verdicts.items() if k in derived]
    agree = sum(1 for v, d in pairs if v == d)
    lo, hi = clopper_pearson_interval(agree, len(pairs), 0.05) if pairs else (float("nan"),) * 2
    return {
        "n_verdict": len(pairs),
        "n_agree_with_derived": agree,
        "agreement": round(agree / len(pairs), 4) if pairs else None,
        "agreement_ci95": [round(lo, 4), round(hi, 4)] if pairs else None,
        "n_disagreements": len(pairs) - agree,
        "disagreeing_ids": [k for k, v in verdicts.items()
                            if k in derived and v != derived[k]][:20],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pass", dest="passes", action="append", required=True,
                    help="EVIDENCELINES:path/to/responses.csv, repeatable, in order")
    ap.add_argument("--items", default="annotation/mab_item_master.csv")
    ap.add_argument("--out", default="results/final/mab_adjudication_summary.json")
    args = ap.parse_args()

    items = pd.read_csv(args.items, keep_default_na=False)
    derived = dict(zip(items.conflict_id.astype(str), items.derived_label))
    n_items = len(items)

    passes = []
    for spec in args.passes:
        lines, _, path = spec.partition(":")
        df = pd.read_csv(path, keep_default_na=False)
        passes.append({"evidence_lines": int(lines), "path": path, "df": df})

    report: dict = {
        "note": (
            "Blinded independent adjudication of all MAB-derived conflicts following "
            "annotation/external_validation_protocol.md, run with a six-member model panel. "
            "Three OpenAI models form the 'openai' tier; a local DeBERTa-v3 NLI model, a local "
            "open-weight Llama 3.1 8B and a local RoBERTa NLI model form the 'independent' tier, "
            "which shares no vendor, weights or training data with it. A tier reaches a verdict "
            "when strictly more than half its members pick the same branch. Items without a "
            "verdict carry to the next pass with a wider evidence window. This remains a "
            "SUBSTITUTE for the released three-human protocol, not a discharge of it."
        ),
        "n_items": n_items,
        "panel": {t: m for t, m in TIERS.items()},
        "adjudicators": {},
        "passes": [],
        "combined": {},
    }

    first = passes[0]["df"]
    for ann, part in first.groupby("annotator_id"):
        committed = part[part.branch_choice != "ambiguous"]
        solo_agree = sum(1 for r in committed.itertuples(index=False)
                         if derived.get(str(r.conflict_id)) == r.branch_choice)
        report["adjudicators"][str(ann)] = {
            "adjudicator": str(part.adjudicator.iloc[0]),
            "vendor": str(part.vendor.iloc[0]) if "vendor" in part else "unknown",
            "n": int(len(part)),
            "abstain_rate": round(float((part.branch_choice == "ambiguous").mean()), 4),
            # How often this adjudicator alone matches the derived label, among
            # the items it chose to answer. Not a verdict, just a per-member
            # quality signal.
            "n_committed": int(len(committed)),
            "solo_agreement_with_derived": round(solo_agree / len(committed), 4)
            if len(committed) else None,
        }

    carried: dict[str, dict[str, str]] = {t: {} for t in TIERS}   # tier -> id -> verdict
    for p in passes:
        entry = {"evidence_lines": p["evidence_lines"], "path": p["path"], "tiers": {}}
        for tier, members in TIERS.items():
            res = tier_pass(p["df"], members)
            pending = {k: v for k, v in res.items() if k not in carried[tier]}
            groups = [v["choices"] for v in res.values()]
            new_verdicts = {k: v["verdict"] for k, v in pending.items() if v["verdict"]}
            carried[tier].update(new_verdicts)
            entry["tiers"][tier] = {
                "n_adjudicated": len(res),
                "n_newly_resolved": len(new_verdicts),
                "fleiss_kappa": round(fleiss_kappa(groups), 4),
                "unanimity": round(float(np.mean([len(set(g)) == 1 for g in groups])), 4)
                if groups else None,
                "pairwise_agreement": round(pairwise_agreement(groups), 4),
                **agreement(new_verdicts, derived),
            }
        report["passes"].append(entry)

    for tier in TIERS:
        report["combined"][tier] = {
            "verdict_rate": round(len(carried[tier]) / n_items, 4),
            "n_no_verdict": n_items - len(carried[tier]),
            **agreement(carried[tier], derived),
        }

    # Cross-vendor pair checks: for every (openai, independent) pair, how often
    # do they pick the same branch among items BOTH chose to answer? This is the
    # tier comparison without a majority rule, so a member that never abstains
    # cannot carry a tier on its own.
    report["cross_vendor_pairs"] = {}
    for pdata in passes:
        df = pdata["df"]
        choice = {(str(r.annotator_id), str(r.conflict_id)): r.branch_choice
                  for r in df.itertuples(index=False)}
        ids = sorted({str(r.conflict_id) for r in df.itertuples(index=False)})
        pair_rows = {}
        for a in TIERS["openai"]:
            for b in TIERS["independent"]:
                both_commit = [i for i in ids
                               if choice.get((a, i), "ambiguous") != "ambiguous"
                               and choice.get((b, i), "ambiguous") != "ambiguous"]
                agree = sum(1 for i in both_commit if choice[(a, i)] == choice[(b, i)])
                with_derived = [i for i in both_commit
                                if i in derived and choice[(a, i)] == choice[(b, i)]
                                and choice[(a, i)] == derived[i]]
                pair_rows[f"{a}+{b}"] = {
                    "n_both_commit": len(both_commit),
                    "pair_agreement": round(agree / len(both_commit), 4) if both_commit else None,
                    "n_pair_agrees_and_matches_derived": len(with_derived),
                    "agreement_with_derived_when_pair_agrees": (
                        round(len(with_derived) / agree, 4) if agree else None),
                }
        report["cross_vendor_pairs"][f"evidence_lines_{pdata['evidence_lines']}"] = pair_rows

    # Who drives the independent tier's disagreements? For every item where that
    # tier's majority differs from the derived label, record each member's own
    # answer, so a tier statistic is never mistaken for a member statistic.
    disputed = [k for k, v in carried["independent"].items()
                if k in derived and v != derived[k]]
    member_on_disputed = {}
    for pdata in passes:
        df = pdata["df"]
        choice = {(str(r.annotator_id), str(r.conflict_id)): r.branch_choice
                  for r in df.itertuples(index=False)}
        per_member = {}
        for ann in TIERS["all"]:
            answered = [i for i in disputed
                        if choice.get((ann, i), "ambiguous") != "ambiguous"]
            per_member[ann] = {
                "n_answered": len(answered),
                "n_matches_derived": sum(1 for i in answered if choice[(ann, i)] == derived[i]),
            }
        member_on_disputed[f"evidence_lines_{pdata['evidence_lines']}"] = per_member
    report["independent_tier_disputed_items"] = {
        "n_disputed": len(disputed), "ids": disputed, "by_member": member_on_disputed,
    }

    both = set(carried["openai"]) & set(carried["independent"])
    cross_agree = sum(1 for k in both if carried["openai"][k] == carried["independent"][k])
    lo, hi = clopper_pearson_interval(cross_agree, len(both), 0.05) if both else (0.0, 1.0)
    report["cross_tier"] = {
        "n_both_reach_verdict": len(both),
        "n_tiers_agree": cross_agree,
        "tier_agreement": round(cross_agree / len(both), 4) if both else None,
        "tier_agreement_ci95": [round(lo, 4), round(hi, 4)],
        "n_only_openai": len(set(carried["openai"]) - set(carried["independent"])),
        "n_only_independent": len(set(carried["independent"]) - set(carried["openai"])),
    }

    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ("combined", "cross_tier")}, indent=2))
    print("\ncross-vendor pairs (items both answered):")
    for win, rows in report["cross_vendor_pairs"].items():
      print(f" {win}")
      for name, v in rows.items():
        print(f"   {name:20} n={v['n_both_commit']:4d} agree={v['pair_agreement']} "
              f"matches derived when they agree={v['agreement_with_derived_when_pair_agrees']}")
    dd = report["independent_tier_disputed_items"]
    print(f"\nindependent-tier majority disputes {dd['n_disputed']} items; by member:")
    for win, pm in dd["by_member"].items():
        print(f" {win}: " + "  ".join(
            f"{a}={v['n_matches_derived']}/{v['n_answered']}" for a, v in pm.items()))
    print("\nper-adjudicator abstention:")
    for ann, a in sorted(report["adjudicators"].items()):
        solo = a["solo_agreement_with_derived"]
        print(f"  {ann} {a['adjudicator']:34} {a['vendor']:15} "
              f"abstain={a['abstain_rate']:.3f} "
              f"solo_agree={'n/a' if solo is None else f'{solo:.3f}'} (n={a['n_committed']})")
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()
