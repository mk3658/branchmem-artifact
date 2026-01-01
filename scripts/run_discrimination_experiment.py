#!/usr/bin/env python3
"""Can a merger tell a constraint violation from a legitimate correction?

Prospective experiment P1 of ANALYSIS_PLAN_ADDENDUM.md. The protocol, the
hypotheses and the item bank were all fixed before this script was first run;
`data/discrimination_bank.sha256` pins the bank.

The paper's existing diagnostic says a merger follows provenance over
conflicting meaning. That is only a defect in half of the cases where it
happens. When a user corrects a stale value, the newer high-confidence update
SHOULD win; when an update would violate a standing constraint, it should not.
The metadata is the same in both. So the question this script asks is not
whether a merger is biased towards provenance but whether it can discriminate.

Four conflict types, one bank, with provenance held constant across types:

  immutable_constraint     correct action: reject the update
  legitimate_override      correct action: accept the update
  independent_observation  correct action: accept whichever the CONTENT marks current
  ambiguous                correct action: abstain

Primary measure, per the preregistration:

  D = accuracy(legitimate_override) - accuracy(immutable_constraint)

under the `update_favoured` provenance arm, where the update always carries
source=user/confidence=.95 and the other branch the weak values. A policy that
reads only the field commits to the update in both types and so scores
(0, 1), D = +1, by construction. A policy that reads the ancestor's semantics
should be high in both, D near 0. Both low with D near 0 is a third,
distinguishable failure and is reported as such.

Secondary arms: `neutral` (metadata identical) and `inverted` (the weak/strong
assignment swapped). Which physical branch carries the update is randomised per
item from a fixed seed so no positional shortcut exists.

Usage:
  OPENAI_API_KEY=... PYTHONPATH=. python scripts/run_discrimination_experiment.py \
      --models gpt-5.4,gpt-5.4-nano --out discrimination_openai.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

from branchmem.benchmark.scenario_generator import Scenario
from branchmem.llm.base import build_backend
from branchmem.memory.branch_sim import BranchSimulator, UpdateSpec
from branchmem.memory.schemas import Resolution
from branchmem.merge.base import find_collisions
from branchmem.merge.confidence_rule import ConfidenceRuleMerge
from branchmem.merge.three_way_llm import ThreeWayLLMMerge

ROOT = Path(__file__).resolve().parents[1]
BANK = ROOT / "data" / "discrimination_bank.json"
OUT_DIR = ROOT / "results" / "final"
SEED = 7717
ENTITY = "user"

LOCAL_BASE_URL = "http://localhost:11434/v1"
OPEN_WEIGHT = {"llama3.1:8b", "qwen2.5:7b", "mistral:7b",
               # reasoning/thinking open-weight models, added in revision
               "deepseek-r1:8b", "qwen3:8b"}

# source/confidence given to the update side and the other side, per arm.
ARMS = {
    "update_favoured": (("user", 0.95), ("inference", 0.55)),
    "neutral": (("observation", 0.70), ("observation", 0.70)),
    "inverted": (("inference", 0.55), ("user", 0.95)),
}
# correct resolution per type: "update", "keep", or None for abstain
CORRECT = {
    "immutable_constraint": "keep",
    "legitimate_override": "update",
    "independent_observation": "update",
    "ambiguous": None,
}


def load_bank() -> list[dict]:
    payload = json.loads(BANK.read_text())
    digest = hashlib.sha256(BANK.read_bytes()).hexdigest()
    pinned = (BANK.parent / "discrimination_bank.sha256").read_text().split()[0]
    if digest != pinned:
        raise SystemExit(
            f"bank hash mismatch: the frozen bank has changed.\n  on disk {digest}\n  pinned  {pinned}")
    return [it for fam in payload["families"] for it in fam["items"]]


def build_scenario(item: dict, arm: str) -> Scenario:
    (src_u, conf_u), (src_k, conf_k) = ARMS[arm]
    rng = random.Random(f"{SEED}:{item['predicate']}")
    update_on_a = rng.random() < 0.5
    sim = BranchSimulator()
    ancestor = sim.create_common_ancestor(
        facts=[(ENTITY, item["predicate"], item["ancestor_value"])], timestamp=0.0)
    truth_key = CORRECT[item["type"]]
    truth = None if truth_key is None else item[f"{truth_key}_value"]

    def spec(value: str, src: str, conf: float, ts: float) -> list[UpdateSpec]:
        return [UpdateSpec(
            entity=ENTITY, predicate=item["predicate"], value=value, timestamp=ts,
            source=src, confidence=conf,
            divergence_type="resolvable_conflict" if truth else "ambiguous_conflict",
            ground_truth_value=truth)]

    # Timestamps are drawn independently of correctness, as everywhere else in
    # the benchmark, so recency carries no signal.
    ts_a = 1.0 + rng.uniform(0.1, 1.0) * 10.0
    ts_b = 1.0 + rng.uniform(0.1, 1.0) * 10.0
    upd = spec(item["update_value"], src_u, conf_u, ts_a if update_on_a else ts_b)
    kep = spec(item["keep_value"], src_k, conf_k, ts_b if update_on_a else ts_a)
    branch_a, branch_b, pairs = sim.diverge(
        ancestor, upd if update_on_a else kep, kep if update_on_a else upd,
        fork_point_timestamp=1.0)
    return Scenario(
        scenario_id=f"{item['family']}_{item['type']}_{item['predicate']}",
        ancestor=ancestor, branch_a=branch_a, branch_b=branch_b, conflict_pairs=pairs,
        source="synthetic",
        metadata={"category": item["type"], "family": item["family"],
                  "domain": item["domain"], "arm": arm,
                  "update_value": item["update_value"], "keep_value": item["keep_value"],
                  "truth": truth},
    )


def decide(strategy, scenario: Scenario) -> dict:
    (key, (fa, fb)), = find_collisions(scenario.branch_a, scenario.branch_b).items()
    result = strategy.merge(scenario.ancestor, scenario.branch_a, scenario.branch_b)
    committed, abstained = None, True
    for rf in result.resulting_facts:
        if rf.fact.key() == key and rf.resolution != Resolution.DROPPED:
            abstained = rf.resolution == Resolution.FLAGGED_UNRESOLVED
            committed = None if abstained else rf.fact.value
    md = scenario.metadata
    truth = md["truth"]
    return {
        "scenario": scenario.scenario_id, "type": md["category"], "family": md["family"],
        "arm": md["arm"], "abstained": abstained,
        "took_update": (not abstained) and committed == md["update_value"],
        "took_keep": (not abstained) and committed == md["keep_value"],
        # Abstaining is correct exactly when there is no determinate answer.
        "correct": abstained if truth is None else ((not abstained) and committed == truth),
        "wrong_commit": (not abstained) and (truth is None or committed != truth),
    }


def cluster_ci(rows: list[dict], field: str, n_boot: int = 2000) -> list[float] | None:
    """95% interval, resampling TEMPLATE FAMILIES rather than items, because
    items authored together are not independent."""
    by_fam: dict[str, list[int]] = defaultdict(list)
    for r in rows:
        by_fam[r["family"]].append(int(bool(r[field])))
    fams = list(by_fam)
    if len(fams) < 3:
        return None
    rng = random.Random(SEED)
    means = []
    for _ in range(n_boot):
        pick = [by_fam[rng.choice(fams)] for _ in fams]
        flat = [v for g in pick for v in g]
        means.append(sum(flat) / len(flat))
    means.sort()
    return [round(means[int(.025 * n_boot)], 4), round(means[int(.975 * n_boot)] , 4)]


def summarise(rows: list[dict]) -> dict:
    n = len(rows)
    if n == 0:
        return {}
    committed = [r for r in rows if not r["abstained"]]
    return {
        "n": n,
        "accuracy": round(sum(r["correct"] for r in rows) / n, 4),
        "accuracy_ci95": cluster_ci(rows, "correct"),
        "coverage": round(len(committed) / n, 4),
        "wrong_commit_rate": round(sum(r["wrong_commit"] for r in rows) / n, 4),
        "took_update_rate": round(sum(r["took_update"] for r in rows) / n, 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True, help="comma-separated model ids")
    ap.add_argument("--out", required=True)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--types", default="", help="comma-separated conflict types (default: all)")
    args = ap.parse_args()

    items = load_bank()
    if args.types:
        keep = set(args.types.split(","))
        unknown = keep - set(CORRECT)
        if unknown:
            raise SystemExit(f"unknown types: {sorted(unknown)}")
        items = [it for it in items if it["type"] in keep]
    if args.limit:
        items = items[: args.limit]
    arms = args.arms.split(",")
    print(f"{len(items)} items, arms {arms}")

    results: dict[str, dict] = {}
    raw: list[dict] = []

    # A deterministic metadata-only policy is the reference every model is read
    # against: by construction it scores (0, 1) on the critical pair.
    policies: dict[str, object] = {"ConfidenceRule": ConfidenceRuleMerge()}
    for model in args.models.split(","):
        cfg = {"backend": "openai_compatible", "model": model, "cache_dir": "llm_cache",
               "temperature": 0.0, "max_tokens": 8000}
        if model in OPEN_WEIGHT:
            cfg["base_url"] = LOCAL_BASE_URL
        policies[f"ThreeWayLLM ({model})"] = ThreeWayLLMMerge(backend=build_backend(cfg))

    for name, strat in policies.items():
        results[name] = {}
        for arm in arms:
            rows = []
            for item in items:
                r = decide(strat, build_scenario(item, arm))
                r["policy"] = name
                rows.append(r)
                raw.append(r)
            present = [t for t in CORRECT if any(r["type"] == t for r in rows)]
            by_type = {t: summarise([r for r in rows if r["type"] == t]) for t in present}
            crit = [r for r in rows if r["type"] in
                    ("immutable_constraint", "legitimate_override")]
            acc_c = by_type.get("immutable_constraint", {}).get("accuracy")
            acc_o = by_type.get("legitimate_override", {}).get("accuracy")
            have_crit = acc_c is not None and acc_o is not None
            results[name][arm] = {
                "overall": summarise(rows),
                "by_type": by_type,
                "discrimination_gap": round(acc_o - acc_c, 4) if have_crit else None,
                "balanced_accuracy_critical": round((acc_o + acc_c) / 2, 4) if have_crit else None,
                "critical_pair_n": len(crit),
            }
            if have_crit:
                print(f"{name:28} {arm:16} constraint={acc_c:.3f} override={acc_o:.3f} "
                      f"D={acc_o - acc_c:+.3f} bal={(acc_o+acc_c)/2:.3f}")
            else:
                print(f"{name:28} {arm:16} acc={results[name][arm]['overall']['accuracy']:.3f}")

    payload = {
        "note": (
            "PROSPECTIVE experiment P1 (ANALYSIS_PLAN_ADDENDUM.md), run after the bank was "
            "frozen and hashed and the hypotheses stated. Four conflict types on one bank with "
            "provenance held constant across types; the primary measure is the discrimination "
            "gap D between legitimate_override and immutable_constraint accuracy under the "
            "update_favoured arm, where a metadata-only policy scores (0,1) by construction. "
            "Intervals resample template families, not items."
        ),
        "bank": str(BANK.relative_to(ROOT)),
        "bank_sha256": hashlib.sha256(BANK.read_bytes()).hexdigest(),
        "seed": SEED, "n_items": len(items), "arms": arms,
        "correct_action_by_type": CORRECT,
        "provenance_arms": {k: {"update_side": v[0], "other_side": v[1]} for k, v in ARMS.items()},
        "results": results,
        "raw": raw,
    }
    (OUT_DIR / args.out).write_text(json.dumps(payload, indent=2))
    print("\nWrote", OUT_DIR / args.out)


if __name__ == "__main__":
    main()
