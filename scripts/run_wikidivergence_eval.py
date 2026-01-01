#!/usr/bin/env python3
"""Evaluate merge policies on WikiDivergence: conflicts we did not construct.

`scripts/build_wikidivergence.py` collects three-way merges out of English
Wikipedia revision history: an infobox field's value before two edits, the
value one editor wrote, the value a different editor wrote next, and the value
the article holds today. Nothing in that chain was authored by us -- not the
divergence, not the provenance metadata, not the label -- which is what makes
it the paper's one external-validity check.

What this script measures:

1. Policy comparison. Recency (the serial order Wikipedia actually has),
   provenance (editor experience), ancestor semantics (local NLI), the hybrid,
   and LLM three-way merging, each with coverage and conditional risk.
2. The paper's central diagnostic, re-run on real items. Provenance is swapped
   between the two branches with the content held fixed, and we report the
   label-free flip rate: the share of decisions that move when only the
   metadata moves.
3. Contextual BranchGuard with reliabilities ESTIMATED from these real
   conflicts, certified on one half and deployed on the other, with channels
   read off the infobox field name.

Results are reported for the whole collection and separately for the
`substitution` subset, where the second editor wrote a third value rather than
restoring the ancestor's. That split matters: restorations are dominated by
vandalism reversion, which an ancestor-reading policy gets for free.

Usage: OPENAI_API_KEY=... PYTHONPATH=. python scripts/run_wikidivergence_eval.py
"""
from __future__ import annotations

import json
import os
import random
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from branchmem.benchmark.downstream_tasks import generate_downstream_questions
from branchmem.benchmark.scenario_generator import Scenario
from branchmem.branchguard.contextual import Action, certify_channel, certify_channels
from branchmem.llm.base import build_backend
from branchmem.memory.branch_sim import BranchSimulator, UpdateSpec
from branchmem.memory.schemas import Resolution
from branchmem.merge.base import find_collisions
from branchmem.merge.confidence_rule import ConfidenceRuleMerge
from branchmem.merge.constraint_veto import ConstraintVetoMerge
from branchmem.merge.last_writer_wins import LastWriterWins
from branchmem.merge.three_way_llm import ThreeWayLLMMerge
from branchmem.conflict.nli_detector import NLIConflictDetector

from run_constraint_veto_hybrid import NLIGatedMerge

ROOT = Path(__file__).resolve().parents[1]
ITEMS = Path(os.environ.get("WIKIDIV_ITEMS", ROOT / "data" / "wikidivergence_items.csv"))
OUT = ROOT / "results" / "final"
MODELS = tuple(os.environ.get("WIKIDIV_MODELS", "gpt-5.4,gpt-5.4-nano").split(","))
LAMBDA = 0.25
DELTA = 0.05
SPLIT_SEED = 90210
_EPOCH = datetime(2001, 1, 1)


def _t(stamp: str) -> float:
    return (datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ") - _EPOCH).total_seconds() / 86400.0


def scenarios_from(items: pd.DataFrame, invert: bool = False) -> list[Scenario]:
    """One scenario per real conflict. `invert` swaps the two provenance blocks
    between branches, leaving every value and timestamp untouched."""
    sim = BranchSimulator()
    out: list[Scenario] = []
    for r in items.itertuples(index=False):
        ancestor = sim.create_common_ancestor(
            facts=[(r.page, r.field, r.ancestor_value)], timestamp=0.0
        )
        truth = r.branch_a_value if r.label == "branch_a" else r.branch_b_value
        src_a, conf_a = (r.b_source, r.b_confidence) if invert else (r.a_source, r.a_confidence)
        src_b, conf_b = (r.a_source, r.a_confidence) if invert else (r.b_source, r.b_confidence)
        ta, tb = _t(r.a_timestamp), _t(r.b_timestamp)
        mk = lambda v, t, s, c: [UpdateSpec(  # noqa: E731
            entity=r.page, predicate=r.field, value=v, timestamp=t, source=s, confidence=c,
            divergence_type="resolvable_conflict", ground_truth_value=truth)]
        branch_a, branch_b, pairs = sim.diverge(
            ancestor,
            mk(r.branch_a_value, max(ta, 0.001), src_a, conf_a),
            mk(r.branch_b_value, max(tb, 0.002), src_b, conf_b),
            fork_point_timestamp=0.0,
        )
        out.append(Scenario(
            scenario_id=r.conflict_id, ancestor=ancestor, branch_a=branch_a,
            branch_b=branch_b, conflict_pairs=pairs, source="wikipedia",
            metadata={"category": "wikidivergence", "channel": r.channel,
                      "is_revert": bool(r.is_revert_to_ancestor), "page": r.page,
                      "field": r.field, "label": r.label},
        ))
    return out


def decide_all(strategy, scens: list[Scenario]) -> list[dict]:
    rows = []
    for s in scens:
        (key, (fa, fb)), = find_collisions(s.branch_a, s.branch_b).items()
        result = strategy.merge(s.ancestor, s.branch_a, s.branch_b)
        committed, abstained = None, True
        for rf in result.resulting_facts:
            # A strategy may emit a DROPPED loser alongside the KEPT winner for
            # the same key (LastWriterWins does); the dropped one is not the
            # decision.
            if rf.fact.key() == key and rf.resolution != Resolution.DROPPED:
                abstained = rf.resolution == Resolution.FLAGGED_UNRESOLVED
                committed = None if abstained else rf.fact.value
        truth = s.conflict_pairs[0].ground_truth_value
        favoured = (fa.value if fa.confidence > fb.confidence
                    else (fb.value if fb.confidence > fa.confidence else None))
        rows.append({
            "conflict_id": s.scenario_id, "channel": s.metadata["channel"],
            "is_revert": s.metadata["is_revert"], "committed": committed,
            "abstained": abstained, "correct": (None if abstained else committed == truth),
            "provenance_favoured": favoured,
            "provenance_correct": (None if favoured is None else favoured == truth),
            "follows_provenance": committed is not None and committed == favoured,
        })
    return rows


def summarise(rows: list[dict]) -> dict:
    n = len(rows)
    committed = [r for r in rows if not r["abstained"]]
    wrong = sum(1 for r in committed if not r["correct"])
    return {
        "n": n,
        "coverage": round(len(committed) / n, 4) if n else None,
        "accuracy": round(sum(1 for r in committed if r["correct"]) / n, 4) if n else None,
        "conditional_risk": round(wrong / len(committed), 4) if committed else None,
        "n_committed": len(committed), "n_wrong": wrong,
    }


def flip_rate(base: list[dict], inv: list[dict]) -> dict:
    pairs = [(b, i) for b, i in zip(base, inv) if not b["abstained"] and not i["abstained"]]
    flips = sum(1 for b, i in pairs if b["committed"] != i["committed"])
    return {"n_commit_both": len(pairs),
            "flip_rate": round(flips / len(pairs), 4) if pairs else None,
            "inverted": summarise(inv)}


def certify_and_deploy(rows: list[dict]) -> dict:
    """Contextual BranchGuard on reliabilities estimated from these conflicts."""
    usable = [r for r in rows if r["provenance_favoured"] is not None]
    rng = random.Random(SPLIT_SEED)
    rng.shuffle(usable)
    cut = len(usable) // 2
    calib, deploy = usable[:cut], usable[cut:]
    counts: dict[str, tuple[int, int]] = {}
    for ch in sorted({r["channel"] for r in usable}):
        part = [r for r in calib if r["channel"] == ch]
        counts[ch] = (sum(1 for r in part if r["provenance_correct"]), len(part))
    certs = certify_channels(counts, abstention_loss=LAMBDA, delta=DELTA)
    pooled = certify_channel("pooled", sum(1 for r in calib if r["provenance_correct"]),
                             len(calib), abstention_loss=LAMBDA, alpha=DELTA)

    def run(action_of) -> dict:
        committed = wrong = 0
        per_ch: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        for r in deploy:
            act = action_of(r)
            if act is Action.ABSTAIN:
                continue
            ok = r["provenance_correct"] if act is Action.FOLLOW else not r["provenance_correct"]
            committed += 1
            wrong += 0 if ok else 1
            per_ch[r["channel"]][0] += 1
            per_ch[r["channel"]][1] += 0 if ok else 1
        risks = {c: round(w / n, 4) for c, (n, w) in per_ch.items() if n}
        return {"coverage": round(committed / len(deploy), 4) if deploy else None,
                "n_committed": committed,
                "conditional_risk": round(wrong / committed, 4) if committed else None,
                "per_channel_risk": risks,
                "worst_channel_risk": max(risks.values()) if risks else None}

    return {
        "n_usable": len(usable), "n_calibration": len(calib), "n_deployment": len(deploy),
        "channel_certificates": {
            c: {"n": x.n, "correct": x.correct,
                "rho": None if x.n == 0 else round(x.empirical_reliability, 4),
                "lower": round(x.lower, 4), "upper": round(x.upper, 4),
                "action": x.action.value}
            for c, x in certs.items()},
        "pooled_certificate": {"n": pooled.n, "correct": pooled.correct,
                               "rho": round(pooled.empirical_reliability, 4),
                               "lower": round(pooled.lower, 4), "upper": round(pooled.upper, 4),
                               "action": pooled.action.value},
        "contextual_deployment": run(lambda r: certs[r["channel"]].action),
        "pooled_deployment": run(lambda r: pooled.action),
        "always_follow_deployment": run(lambda r: Action.FOLLOW),
    }


def main() -> None:
    items = pd.read_csv(ITEMS, keep_default_na=False)
    print(f"{len(items)} real conflicts over {items.page.nunique()} articles")
    base_scens = scenarios_from(items)
    inv_scens = scenarios_from(items, invert=True)

    detector = NLIConflictDetector()
    policies: dict[str, object] = {
        "Recency (last writer wins)": LastWriterWins(),
        "Provenance (confidence rule)": ConfidenceRuleMerge(),
        "NliGated (semantics only)": NLIGatedMerge(detector=detector),
        "ConstraintVeto (semantics then provenance)": ConstraintVetoMerge(detector=detector),
    }
    for model in MODELS:
        backend = build_backend({"backend": "openai_compatible", "model": model,
                                 "cache_dir": "llm_cache", "temperature": 0.0,
                                 "max_tokens": 8000})
        policies[f"ThreeWayLLM ({model})"] = ThreeWayLLMMerge(backend=backend)

    results: dict[str, dict] = {}
    for name, strat in policies.items():
        print(f"running {name} ...")
        base = decide_all(strat, base_scens)
        print(f"running {name} [provenance inverted] ...")
        inv = decide_all(strat, inv_scens)
        subs = [r for r in base if not r["is_revert"]]
        revs = [r for r in base if r["is_revert"]]
        results[name] = {
            "all": summarise(base),
            "substitution_only": summarise(subs),
            "revert_only": summarise(revs),
            "provenance_inversion": flip_rate(base, inv),
            "by_channel": {c: summarise([r for r in base if r["channel"] == c])
                           for c in sorted({r["channel"] for r in base})},
        }
        if name == "Provenance (confidence rule)":
            results[name]["certification"] = certify_and_deploy(base)

    print(f"\n{'policy':44} {'cov':>6} {'acc':>6} {'risk':>7} {'flip':>6}")
    for name, r in results.items():
        a = r["all"]
        fl = r["provenance_inversion"]["flip_rate"]
        print(f"{name:44} {a['coverage']:6.3f} {a['accuracy']:6.3f} "
              f"{(a['conditional_risk'] if a['conditional_risk'] is not None else float('nan')):7.3f} "
              f"{(fl if fl is not None else float('nan')):6.3f}")

    cert = results["Provenance (confidence rule)"]["certification"]
    print("\nestimated channel reliability (calibration half):")
    for c, x in cert["channel_certificates"].items():
        print(f"  {c:16} n={x['n']:4d} rho={x['rho']}  "
              f"CI=[{x['lower']},{x['upper']}] -> {x['action']}")
    print(f"  {'POOLED':16} n={cert['pooled_certificate']['n']:4d} "
          f"rho={cert['pooled_certificate']['rho']} -> {cert['pooled_certificate']['action']}")
    for label in ("contextual_deployment", "pooled_deployment", "always_follow_deployment"):
        d = cert[label]
        print(f"  {label:28} cov={d['coverage']} risk={d['conditional_risk']} "
              f"worst={d['worst_channel_risk']}")

    payload = {
        "note": (
            "Post-hoc, exploratory (not in ANALYSIS_PLAN.md). WikiDivergence: three-way merges "
            "extracted from English Wikipedia infobox revision history, where the divergence, the "
            "provenance metadata and the label are all external to this paper. The label is the "
            "value the article's current revision holds, kept only when it equals one of the two "
            "branch values. Provenance is a pre-declared mapping of real editor metadata: source "
            "from account type and user group, confidence monotone in lifetime edit count. "
            "Channels are read off the infobox field name. 'substitution_only' excludes conflicts "
            "whose second edit restored the ancestor value."
        ),
        "items_file": str(ITEMS.relative_to(ROOT)) if ITEMS.is_relative_to(ROOT) else str(ITEMS),
        "n_conflicts": int(len(items)),
        "n_articles": int(items.page.nunique()),
        "n_substitution": int((~items.is_revert_to_ancestor.astype(bool)).sum()),
        "n_revert": int(items.is_revert_to_ancestor.astype(bool).sum()),
        "label_distribution": items.label.value_counts().to_dict(),
        "channel_distribution": items.channel.value_counts().to_dict(),
        "median_intervening_revisions": float(items.n_intervening_revisions.median()),
        "median_divergence_days": float(items.divergence_days.median()),
        "lambda": LAMBDA, "delta": DELTA, "split_seed": SPLIT_SEED,
        "results": results,
    }
    out_name = os.environ.get("WIKIDIV_OUT", "wikidivergence.json")
    (OUT / out_name).write_text(json.dumps(payload, indent=2))
    print("\nWrote", OUT / out_name)


if __name__ == "__main__":
    main()
