#!/usr/bin/env python3
"""Build a BLIND human-validation package for benchmark ground truth
(for independent validation of construction correctness).

Unlike annotation/audit_sample.csv, this HIDES the generator's label from the
annotator: annotators see only what a merger sees (ancestor value/constraint,
each branch's value/source/confidence) and independently record their own
judgment. Their judgments are later scored against the hidden key by
scripts/score_human_validation.py.

Stratified sample across the categories worth checking:
  - locked `resolvable`      (source/confidence signal decides)   : 12
  - locked `ambiguous`       (correct behavior is to flag)        : 8
  - `semantic_resolvable`    (ancestor constraint decides)        : 10
Total 30 items -> annotation/validation_blind.csv (give a COPY to each of
>=3 annotators) + annotation/validation_key.csv (hidden generator labels).
"""
from __future__ import annotations

import csv
import random
from pathlib import Path

from branchmem.benchmark.scenario_generator import ScenarioConfig, ScenarioGenerator
from branchmem.benchmark.semantic_resolvable_generator import generate_semantic_resolvable_scenarios

SEED = 4242  # sampling seed for the validation set (disjoint from run seeds)
ANN_DIR = Path(__file__).resolve().parents[1] / "annotation"
BLIND = ANN_DIR / "validation_blind.csv"
KEY = ANN_DIR / "validation_key.csv"

BLIND_FIELDS = [
    "item_id", "ancestor_value", "branch_a_value", "branch_a_source", "branch_a_confidence",
    "branch_b_value", "branch_b_source", "branch_b_confidence",
    # annotator fills these, blind to the generator:
    "human_has_single_correct_answer",  # yes/no  (no => should flag as unresolvable)
    "human_correct_value",              # a / b / (free text) / blank if flag
    "human_confidence_1to5", "human_notes",
]
KEY_FIELDS = ["item_id", "category", "generated_conflict_type", "generated_correct_value", "template"]


def _pull(sc, kind, want, out_rows, key_rows, rng):
    fl = {f.fact_id: f for f in sc.branch_a.facts + sc.branch_b.facts}
    anc = {f.key(): f.value for f in sc.ancestor.facts}
    picked = 0
    for pair in sc.conflict_pairs:
        if picked >= want:
            break
        ctype = pair.conflict_type or "orthogonal"
        if kind == "resolvable" and ctype != "resolvable_conflict" and ctype != "resolvable":
            continue
        if kind == "ambiguous" and ctype not in ("ambiguous_conflict", "ambiguous"):
            continue
        a, b = fl[pair.fact_a_id], fl[pair.fact_b_id]
        out_rows.append({
            "item_id": None, "ancestor_value": anc.get(a.key(), "(none)"),
            "branch_a_value": a.value, "branch_a_source": a.source, "branch_a_confidence": a.confidence,
            "branch_b_value": b.value, "branch_b_source": b.source, "branch_b_confidence": b.confidence,
            "human_has_single_correct_answer": "", "human_correct_value": "",
            "human_confidence_1to5": "", "human_notes": "",
            "_gt": pair.ground_truth_value or "", "_ctype": ctype, "_template": sc.metadata.get("template_predicate", ""),
        })
        picked += 1
    return picked


def main() -> None:
    rng = random.Random(SEED)
    gen = ScenarioGenerator()
    locked = gen.generate(60, ScenarioConfig(divergence_span=10.0), seed=2026)
    sem = generate_semantic_resolvable_scenarios(150, seed=3001)

    rows: list[dict] = []
    need = {"resolvable": 12, "ambiguous": 8}
    for sc in locked:
        for kind in list(need):
            if need[kind] > 0:
                need[kind] -= _pull(sc, kind, need[kind], rows, None, rng)
    # semantic_resolvable: 10 distinct templates (label them explicitly, not by
    # the underlying pair.conflict_type which is a generic "resolvable_conflict").
    seen_tpl, sem_taken = set(), 0
    for sc in sem:
        tpl = sc.metadata.get("template_predicate")
        if tpl in seen_tpl or sem_taken >= 10:
            continue
        fl = {f.fact_id: f for f in sc.branch_a.facts + sc.branch_b.facts}
        anc = {f.key(): f.value for f in sc.ancestor.facts}
        p = sc.conflict_pairs[0]
        a, b = fl[p.fact_a_id], fl[p.fact_b_id]
        rows.append({
            "item_id": None, "ancestor_value": anc.get(a.key(), "(none)"),
            "branch_a_value": a.value, "branch_a_source": a.source, "branch_a_confidence": a.confidence,
            "branch_b_value": b.value, "branch_b_source": b.source, "branch_b_confidence": b.confidence,
            "human_has_single_correct_answer": "", "human_correct_value": "",
            "human_confidence_1to5": "", "human_notes": "",
            "_gt": p.ground_truth_value or "", "_ctype": "semantic_resolvable", "_template": tpl,
        })
        seen_tpl.add(tpl)
        sem_taken += 1

    rng.shuffle(rows)
    for i, r in enumerate(rows, 1):
        r["item_id"] = f"v{i:03d}"

    ANN_DIR.mkdir(parents=True, exist_ok=True)
    with open(BLIND, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=BLIND_FIELDS, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    with open(KEY, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=KEY_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({"item_id": r["item_id"], "category": r["_ctype"],
                        "generated_conflict_type": r["_ctype"],
                        "generated_correct_value": r["_gt"], "template": r["_template"]})
    print(f"Wrote {len(rows)} items -> {BLIND} (annotator-facing, blind) and {KEY} (hidden key)")


if __name__ == "__main__":
    main()
