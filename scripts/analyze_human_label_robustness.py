#!/usr/bin/env python3
"""Do the conclusions survive swapping construction targets for human labels?

The benchmark scores against the label its generator scripted, which raises the
obvious question: how much of the paper depends on that, and what changes if
the three-annotator blind study's majority judgment is used instead --- above
all on defeasible preferences, where a construction target is a convention
rather than a fact.

This script answers it on the only items where both labels exist: the $30$
blind-study items (12 locked resolvable, 8 locked ambiguous, 10
semantic-resolvable). Each item is matched back to the scenario it came from by
its ancestor and branch values, every policy's decision is replayed from cache,
and each decision is scored twice: once against the generator's target and once
against the human majority.

Human majority construction. An annotator who answered "no single correct
answer" is treated as abstaining, which is the correct behaviour on the
ambiguous items; otherwise their stated value is the label. The majority needs
at least two of three annotators agreeing, and an item with no majority is
reported rather than dropped.

Zero new API: every LLM decision is served from the content-hash cache.

Usage: OPENAI_API_KEY=cache-only-sentinel PYTHONPATH=. python scripts/analyze_human_label_robustness.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from branchmem.benchmark.scenario_generator import ScenarioConfig, ScenarioGenerator
from branchmem.benchmark.semantic_resolvable_generator import (
    generate_semantic_resolvable_scenarios,
)
from branchmem.conflict.nli_detector import NLIConflictDetector
from branchmem.llm.base import build_backend
from branchmem.memory.schemas import Resolution
from branchmem.merge.base import current_facts_by_key, find_collisions
from branchmem.merge.confidence_rule import ConfidenceRuleMerge
from branchmem.merge.constraint_veto import ConstraintVetoMerge
from branchmem.merge.last_writer_wins import LastWriterWins
from branchmem.merge.three_way_llm import ThreeWayLLMMerge

from run_constraint_veto_hybrid import NLIGatedMerge

ROOT = Path(__file__).resolve().parents[1]
ANN = ROOT / "annotation"
OUT = ROOT / "results" / "final"
ANNOTATORS = ("validation_annotator1.csv", "validation_annotator2.csv",
              "validation_annotator3.csv")


def _norm(v) -> str:
    return str(v).strip().lower()


def scenario_index() -> dict[tuple[str, str, str], tuple[object, tuple[str, str]]]:
    """Map (ancestor, branch_a, branch_b) values to the scenario and key.

    The blind sheets were drawn from these two populations with these seeds, so
    matching on values recovers the provenance of every item without having to
    replay the sampler's RNG.
    """
    gen = ScenarioGenerator()
    pops = list(gen.generate(60, ScenarioConfig(divergence_span=10.0), seed=2026))
    pops += list(generate_semantic_resolvable_scenarios(150, seed=3001))
    index: dict[tuple[str, str, str], tuple[object, tuple[str, str]]] = {}
    for sc in pops:
        anc = {k: f.value for k, f in current_facts_by_key(sc.ancestor).items()}
        for key, (fa, fb) in find_collisions(sc.branch_a, sc.branch_b).items():
            sig = (_norm(anc.get(key, "(none)")), _norm(fa.value), _norm(fb.value))
            index.setdefault(sig, (sc, key))
    return index


def human_labels() -> dict[str, dict]:
    """Per-item majority judgment over the three annotators."""
    frames = [pd.read_csv(ANN / f, keep_default_na=False) for f in ANNOTATORS]
    out: dict[str, dict] = {}
    for item_id in frames[0].item_id:
        votes = []
        for df in frames:
            row = df[df.item_id == item_id].iloc[0]
            if _norm(row.human_has_single_correct_answer) in ("no", "n", "false"):
                votes.append(None)                      # abstain
            else:
                votes.append(_norm(row.human_correct_value) or None)
        counts = Counter(v if v is not None else "\x00abstain" for v in votes)
        top, n = counts.most_common(1)[0]
        out[str(item_id)] = {
            "votes": ["(abstain)" if v is None else v for v in votes],
            "majority": None if top == "\x00abstain" else top,
            "has_majority": n >= 2,
            "abstain_majority": top == "\x00abstain" and n >= 2,
        }
    return out


def decision(strategy, scenario, key) -> tuple[str | None, bool]:
    result = strategy.merge(scenario.ancestor, scenario.branch_a, scenario.branch_b)
    for rf in result.resulting_facts:
        # Skip a DROPPED loser emitted next to the KEPT winner for the same key
        # (LastWriterWins does this); it is not the strategy's decision.
        if rf.fact.key() == key and rf.resolution != Resolution.DROPPED:
            if rf.resolution == Resolution.FLAGGED_UNRESOLVED:
                return None, True
            return _norm(rf.fact.value), False
    return None, True


def score(rows: list[dict], label_field: str) -> dict:
    live = [r for r in rows if r[label_field + "_known"]]
    committed = [r for r in live if not r["abstained"]]
    expect_abstain = [r for r in live if r[label_field] is None]
    correct = 0
    for r in live:
        if r[label_field] is None:
            correct += int(r["abstained"])
        else:
            correct += int(not r["abstained"] and r["committed"] == r[label_field])
    wrong = sum(1 for r in committed
                if r[label_field] is None or r["committed"] != r[label_field])
    return {
        "n_scored": len(live),
        "accuracy": round(correct / len(live), 4) if live else None,
        "coverage": round(len(committed) / len(live), 4) if live else None,
        "conditional_risk": round(wrong / len(committed), 4) if committed else None,
        "n_items_whose_correct_action_is_abstain": len(expect_abstain),
    }


def main() -> None:
    index = scenario_index()
    humans = human_labels()
    key_df = pd.read_csv(ANN / "validation_key.csv", keep_default_na=False)
    sheet = pd.read_csv(ANN / ANNOTATORS[0], keep_default_na=False)
    gen_label = dict(zip(key_df.item_id.astype(str), key_df.generated_correct_value))
    category = dict(zip(key_df.item_id.astype(str), key_df.category))

    matched: list[dict] = []
    unmatched: list[str] = []
    for r in sheet.itertuples(index=False):
        sig = (_norm(r.ancestor_value), _norm(r.branch_a_value), _norm(r.branch_b_value))
        if sig not in index:
            unmatched.append(str(r.item_id))
            continue
        sc, key = index[sig]
        matched.append({"item_id": str(r.item_id), "scenario": sc, "key": key})
    print(f"matched {len(matched)}/{len(sheet)} blind items back to their scenarios")
    if unmatched:
        print("  unmatched:", unmatched)

    detector = NLIConflictDetector()
    policies: dict[str, object] = {
        "Recency (LWW)": LastWriterWins(),
        "ConfidenceRule": ConfidenceRuleMerge(),
        "NliGated": NLIGatedMerge(detector=detector),
        "ConstraintVeto": ConstraintVetoMerge(detector=detector),
    }
    for model in ("gpt-5.4-nano", "gpt-5.4"):
        backend = build_backend({"backend": "openai_compatible", "model": model,
                                 "cache_dir": "llm_cache", "temperature": 0.0,
                                 "max_tokens": 8000})
        policies[f"ThreeWayLLM ({model})"] = ThreeWayLLMMerge(backend=backend)

    results: dict[str, dict] = {}
    per_item: list[dict] = []
    for name, strat in policies.items():
        rows = []
        for m in matched:
            committed, abstained = decision(strat, m["scenario"], m["key"])
            item = m["item_id"]
            h = humans[item]
            g = _norm(gen_label[item]) or None
            rows.append({
                "item_id": item, "category": category[item],
                "committed": committed, "abstained": abstained,
                "generator": g, "generator_known": True,
                "human": h["majority"], "human_known": h["has_majority"],
            })
            per_item.append({"policy": name, **{k: v for k, v in rows[-1].items()}})
        cats = sorted({r["category"] for r in rows})
        results[name] = {
            "under_generator_target": score(rows, "generator"),
            "under_human_majority": score(rows, "human"),
            "by_category_generator": {c: score([r for r in rows if r["category"] == c],
                                               "generator") for c in cats},
            "by_category_human": {c: score([r for r in rows if r["category"] == c],
                                           "human") for c in cats},
        }

    disagree = [i for i, h in humans.items()
                if h["has_majority"] and h["majority"] != (_norm(gen_label[i]) or None)]
    no_majority = [i for i, h in humans.items() if not h["has_majority"]]

    print(f"\nlabel sets differ on {len(disagree)}/{len(humans)} items: {disagree}")
    print(f"no human majority on {len(no_majority)} items: {no_majority}")
    print(f"\n{'policy':26} {'acc(gen)':>9} {'acc(human)':>11} "
          f"{'risk(gen)':>10} {'risk(human)':>12}")
    for name, r in results.items():
        g, h = r["under_generator_target"], r["under_human_majority"]
        def f(x):
            return "   n/a" if x is None else f"{x:6.3f}"
        print(f"{name:26} {f(g['accuracy']):>9} {f(h['accuracy']):>11} "
              f"{f(g['conditional_risk']):>10} {f(h['conditional_risk']):>12}")

    payload = {
        "note": (
            "Post-hoc, exploratory (not in ANALYSIS_PLAN.md). Every policy's cached decision on "
            "the 30 blind human-validation items, scored twice: against the generator's "
            "construction target and against the three-annotator majority. An annotator who "
            "answered 'no single correct answer' is treated as abstaining, so on the ambiguous "
            "items the correct action is to abstain under both label sets. Items are matched back "
            "to their scenarios by ancestor and branch values, not by replaying the sampler. "
            "Zero new API."
        ),
        "n_items": int(len(sheet)),
        "n_matched": len(matched),
        "unmatched_items": unmatched,
        "items_where_label_sets_differ": disagree,
        "items_without_human_majority": no_majority,
        "human_votes": {i: h["votes"] for i, h in humans.items()},
        "results": results,
        "per_item": per_item,
    }
    (OUT / "human_label_robustness.json").write_text(json.dumps(payload, indent=2))
    print("\nWrote", OUT / "human_label_robustness.json")


if __name__ == "__main__":
    main()
