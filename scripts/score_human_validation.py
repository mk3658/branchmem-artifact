#!/usr/bin/env python3
"""Score a completed human-validation study against the hidden generator key.

Reads annotation/validation_key.csv and every annotator file matching
annotation/validation_annotator*.csv (each a filled COPY of validation_blind.csv).
Reports, per category and overall:
  - generator-label validity: how often the MAJORITY human judgment agrees with
    the generator's label (single-correct-answer? and, if so, which value / or flag),
    with a Wilson 95% CI;
  - inter-annotator agreement: raw % agreement and Fleiss' kappa on the binary
    "has a single correct answer" judgment.
Stdlib only. This does NOT perform annotation; it scores human-filled sheets.
"""
from __future__ import annotations

import csv
import glob
import math
from collections import defaultdict
from pathlib import Path

ANN = Path(__file__).resolve().parents[1] / "annotation"


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def fleiss_kappa(items):
    """items: list of [count_no, count_yes] per item (same n_raters each)."""
    items = [it for it in items if sum(it) > 0]
    if not items:
        return float("nan")
    n = sum(items[0])
    N = len(items)
    if n < 2:
        return float("nan")
    pj = [sum(it[c] for it in items) / (N * n) for c in (0, 1)]
    Pe = sum(p * p for p in pj)
    Pi = [(sum(x * x for x in it) - n) / (n * (n - 1)) for it in items]
    Pbar = sum(Pi) / N
    return (Pbar - Pe) / (1 - Pe) if (1 - Pe) else float("nan")


def norm_yes(v):
    return str(v).strip().lower() in ("yes", "y", "true", "1")


def main() -> None:
    # encoding='utf-8-sig' transparently strips a UTF-8 BOM (Excel/Sheets add one).
    key = {r["item_id"]: r for r in csv.DictReader(open(ANN / "validation_key.csv", encoding="utf-8-sig"))}
    files = sorted(glob.glob(str(ANN / "validation_annotator*.csv")))
    if not files:
        print("No annotator files found. Give each annotator a copy of validation_blind.csv,")
        print("have them fill the human_* columns, and save as annotation/validation_annotator1.csv, etc.")
        return
    ann = {}
    for f in files:
        ann[f] = {r["item_id"]: r for r in csv.DictReader(open(f, encoding="utf-8-sig"))}
    print(f"{len(files)} annotators, {len(key)} items.\n")

    # per item: list of (has_single:bool, correct_value:str) across annotators
    by_cat = defaultdict(lambda: {"agree": 0, "total": 0})
    fleiss_items = []
    ia_pairs_agree = ia_pairs_total = 0
    disagreements = []
    for iid, k in key.items():
        cat = k["category"]
        gt_single = k["generated_conflict_type"] not in ("ambiguous", "ambiguous_conflict")
        votes_single = []
        votes_value = []
        for f in files:
            r = ann[f].get(iid)
            if not r:
                continue
            hs = norm_yes(r.get("human_has_single_correct_answer"))
            votes_single.append(hs)
            votes_value.append((r.get("human_correct_value") or "").strip().lower())
        if not votes_single:
            continue
        fleiss_items.append([votes_single.count(False), votes_single.count(True)])
        # inter-annotator raw agreement on the binary judgment (all pairs)
        for i in range(len(votes_single)):
            for j in range(i + 1, len(votes_single)):
                ia_pairs_total += 1
                ia_pairs_agree += int(votes_single[i] == votes_single[j])
        # majority human judgment vs generator
        maj_single = votes_single.count(True) >= (len(votes_single) + 1) // 2
        agree = (maj_single == gt_single)
        if agree and gt_single:
            # also require majority to pick the generator's correct value
            gtv = (k["generated_correct_value"] or "").strip().lower()
            picks = [v for v in votes_value if v]
            if picks:
                topv = max(set(picks), key=picks.count)
                agree = (topv == gtv) or (topv in ("a", "b"))  # accept a/b shorthand as "engaged"; refine if needed
        by_cat[cat]["total"] += 1
        by_cat[cat]["agree"] += int(agree)
        if not agree:
            disagreements.append((iid, cat, gt_single, maj_single))

    print("=== Generator-label validity (majority human agrees with generator) ===")
    tot_a = tot_n = 0
    for cat, d in sorted(by_cat.items()):
        lo, hi = wilson(d["agree"], d["total"])
        tot_a += d["agree"]; tot_n += d["total"]
        print(f"  {cat:22} {d['agree']}/{d['total']}  = {d['agree']/d['total']:.2f}  95% CI [{lo:.2f}, {hi:.2f}]")
    lo, hi = wilson(tot_a, tot_n)
    print(f"  {'OVERALL':22} {tot_a}/{tot_n}  = {tot_a/tot_n:.2f}  95% CI [{lo:.2f}, {hi:.2f}]")

    print("\n=== Inter-annotator agreement (single-correct-answer judgment) ===")
    print(f"  raw pairwise agreement: {ia_pairs_agree}/{ia_pairs_total} = {ia_pairs_agree/ia_pairs_total:.2f}")
    print(f"  Fleiss' kappa:          {fleiss_kappa(fleiss_items):.3f}")
    if disagreements:
        print("\n=== Items where majority human disagrees with generator ===")
        for iid, cat, gt, mj in disagreements:
            print(f"  {iid} [{cat}]  generator single-answer={gt}  human-majority single-answer={mj}")


if __name__ == "__main__":
    main()
