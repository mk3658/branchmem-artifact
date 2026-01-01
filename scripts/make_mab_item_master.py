#!/usr/bin/env python3
"""Materialise the 270 MAB-derived branch conflicts as an adjudication master sheet.

`annotation/mab_item_master_template.csv` ships with headers only. This script
fills it from the same construction the reconciled MAB run used (seed 2026,
max_conflict_keys=50), so the adjudication population is exactly the released
270 conflicts and not a fresh draw.

For every conflict it records the two branch values, the common-ancestor value
where one exists, and the evidence context an adjudicator needs: the numbered
MemoryAgentBench lines that mention the entity or either value. It does NOT
record source/confidence metadata, which the protocol keeps hidden from
adjudicators.

Needs network access once to fetch the public MAB parquet.

Usage: PYTHONPATH=. python scripts/make_mab_item_master.py
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import pandas as pd

from branchmem.benchmark.downstream_tasks import generate_downstream_questions
from branchmem.benchmark.mab_extension import (
    build_branch_scenario_from_mab_row,
    load_mab_conflict_resolution,
)
from branchmem.merge.base import current_facts_by_key, find_collisions

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_LINES = int(os.environ.get("MAB_EVIDENCE_LINES", "6"))
_SUFFIX = "" if EVIDENCE_LINES == 6 else f"_ev{EVIDENCE_LINES}"
OUT = ROOT / "annotation" / f"mab_item_master{_SUFFIX}.csv"
SEED = 2026
MAX_CONFLICT_KEYS = 50


def evidence_for(context: str, entity: str, values: list[str]) -> str:
    """The numbered context lines bearing on this conflict, in original order."""
    lines = [ln.strip() for ln in context.split("\n") if ln.strip()]
    needles = [entity.lower()] + [v.lower()[:40] for v in values if v]
    hits = [ln for ln in lines if any(nd and nd in ln.lower() for nd in needles)]
    if not hits:
        hits = lines[:EVIDENCE_LINES]
    # Keep the last lines: MAB's ordering is its freshness ground truth, and the
    # final restatement of a key is the one its label is derived from.
    return " || ".join(hits[-EVIDENCE_LINES:])


def predicate_family(predicate: str) -> str:
    """Coarse family used for stratification; derived from the relation name."""
    p = predicate.lower()
    for family, pattern in (
        ("location", r"countr|citizen|capital|locat|headquart|continent"),
        ("affiliation", r"employ|member|owned|develop|produc|publish"),
        ("creative_work", r"author|director|composer|screenwrit|creator"),
        ("language", r"languag"),
        ("person_role", r"spouse|father|mother|child|position|occupation"),
    ):
        if re.search(pattern, p):
            return family
    return "other"


def main() -> None:
    df = load_mab_conflict_resolution()
    rows: list[dict] = []
    for i, mab_row in df.iterrows():
        scenario = build_branch_scenario_from_mab_row(
            mab_row, scenario_id=f"mab_{i}", seed=SEED, max_conflict_keys=MAX_CONFLICT_KEYS
        )
        if scenario is None:
            continue
        questions = {(q.entity, q.predicate): q for q in generate_downstream_questions(scenario)}
        a_facts = current_facts_by_key(scenario.branch_a)
        b_facts = current_facts_by_key(scenario.branch_b)
        ancestor = current_facts_by_key(scenario.ancestor)
        collisions = find_collisions(scenario.branch_a, scenario.branch_b)
        context = str(mab_row["context"])

        for key in sorted(collisions):
            fa, fb = collisions[key]
            q = questions.get(key)
            if q is None:
                continue
            entity, predicate = key
            anc = ancestor.get(key)
            derived = q.expected_answer
            if derived == fa.value:
                derived_label = "branch_a"
            elif derived == fb.value:
                derived_label = "branch_b"
            else:
                derived_label = "other"
            provenance_choice = (
                "tie"
                if fa.confidence == fb.confidence
                else ("branch_a" if fa.confidence > fb.confidence else "branch_b")
            )
            rows.append(
                {
                    "conflict_id": f"mab_{i}_{entity}_{predicate}".replace(" ", "_"),
                    "source_row": int(i),
                    "channel": q.category,
                    "predicate_family": predicate_family(predicate),
                    "entity": entity,
                    "predicate": predicate,
                    "ancestor_value": "" if anc is None else anc.value,
                    "branch_a_value": fa.value,
                    "branch_b_value": fb.value,
                    "evidence_context": evidence_for(context, entity, [fa.value, fb.value]),
                    "derived_label": derived_label,
                    "provenance_choice": provenance_choice,
                }
            )

    items = pd.DataFrame(rows)
    items = items.drop_duplicates(subset="conflict_id")
    items.to_csv(OUT, index=False)
    print(f"wrote {len(items)} conflicts to {OUT}")
    print("\nby source row:\n", items.source_row.value_counts().sort_index().to_string())
    print("\nby predicate family:\n", items.predicate_family.value_counts().to_string())
    print("\nderived label:\n", items.derived_label.value_counts().to_string())
    print("\nprovenance choice:\n", items.provenance_choice.value_counts().to_string())


if __name__ == "__main__":
    main()
