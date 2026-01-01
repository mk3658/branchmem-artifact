#!/usr/bin/env python3
"""Does the specification fix survive when the lexical cue is removed?

Post-hoc, exploratory; added in revision. On the discrimination bank a prompt
that asks the model to classify the stored entry and says what follows (P4)
lifts `gpt-5.4` from .500 to .938 balanced accuracy. But a modal marker
separates that bank's two critical types perfectly, so P4's number there is an
upper bound: the model may be keying on the ancestor's register rather than
reasoning about standing.

This script re-runs the same prompts end-to-end on
`data/hard_standing_bank.json`, where all four obligation types carry an
obligation-bearing ancestor and only one makes rejecting the update correct.
Metrics match `run_standing_detection.py` so the merge result and the detection
result can be read side by side:

  violations_caught  share of standing_indefinite items correctly kept
  freezing_rate      share of the 48 update-correct items wrongly kept
  balanced           mean of caught and (1 - frozen)

The provenance arm is the primary one from the discrimination experiment: the
update side always carries source=user, confidence=.95, so the metadata is
uninformative about which type an item is and a field-reading policy accepts
every update.

Usage:
  OPENAI_API_KEY=... PYTHONPATH=.:scripts python scripts/run_hard_bank_merge.py \
      --models gpt-5.4 --out hard_bank_merge.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from branchmem.benchmark.scenario_generator import Scenario
from branchmem.llm.base import build_backend
from branchmem.memory.branch_sim import BranchSimulator, UpdateSpec
from branchmem.memory.schemas import Resolution
from branchmem.merge.base import find_collisions
from branchmem.merge.confidence_rule import ConfidenceRuleMerge
from branchmem.merge.three_way_llm import ThreeWayLLMMerge

from run_specification_ablation import build_prompts, _COT_SCHEMA_NOTE  # type: ignore
from run_standing_detection import LOCAL_BASE_URL, OPEN_WEIGHT, score  # type: ignore

ROOT = Path(__file__).resolve().parents[1]
BANK = ROOT / "data" / "hard_standing_bank.json"
OUT_DIR = ROOT / "results" / "final"
SEED = 4242
ENTITY = "user"
STRONG = ("user", 0.95)
WEAK = ("inference", 0.55)


def load_bank() -> list[dict]:
    digest = hashlib.sha256(BANK.read_bytes()).hexdigest()
    pinned = (BANK.parent / "hard_standing_bank.sha256").read_text().split()[0]
    if digest != pinned:
        raise SystemExit(f"hard bank hash mismatch\n  disk {digest}\n  pin  {pinned}")
    payload = json.loads(BANK.read_text())
    return [it for f in payload["families"] for it in f["items"]]


def build_scenario(item: dict) -> Scenario:
    rng = random.Random(f"{SEED}:{item['predicate']}")
    update_on_a = rng.random() < 0.5
    sim = BranchSimulator()
    ancestor = sim.create_common_ancestor(
        facts=[(ENTITY, item["predicate"], item["ancestor_value"])], timestamp=0.0)
    truth = item["update_value"] if item["correct_action"] == "update" else item["keep_value"]

    def spec(value, src, conf, ts):
        return [UpdateSpec(entity=ENTITY, predicate=item["predicate"], value=value,
                           timestamp=ts, source=src, confidence=conf,
                           divergence_type="resolvable_conflict",
                           ground_truth_value=truth)]

    ts_a = 1.0 + rng.uniform(0.1, 1.0) * 10.0
    ts_b = 1.0 + rng.uniform(0.1, 1.0) * 10.0
    upd = spec(item["update_value"], *STRONG, ts_a if update_on_a else ts_b)
    kep = spec(item["keep_value"], *WEAK, ts_b if update_on_a else ts_a)
    a, b, pairs = sim.diverge(ancestor, upd if update_on_a else kep,
                              kep if update_on_a else upd, fork_point_timestamp=1.0)
    return Scenario(
        scenario_id=f"{item['family']}_{item['type']}_{item['predicate']}",
        ancestor=ancestor, branch_a=a, branch_b=b, conflict_pairs=pairs,
        source="synthetic",
        metadata={"category": item["type"], "family": item["family"],
                  "update_value": item["update_value"], "keep_value": item["keep_value"],
                  "truth": truth, "correct_action": item["correct_action"]},
    )


def decide_keep(strategy, scenario: Scenario) -> int | None:
    """1 if the policy rejected the update, 0 if it accepted, None if it abstained."""
    (key, _), = find_collisions(scenario.branch_a, scenario.branch_b).items()
    result = strategy.merge(scenario.ancestor, scenario.branch_a, scenario.branch_b)
    md = scenario.metadata
    for rf in result.resulting_facts:
        if rf.fact.key() == key and rf.resolution != Resolution.DROPPED:
            if rf.resolution == Resolution.FLAGGED_UNRESOLVED:
                return None
            return int(rf.fact.value == md["keep_value"])
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="gpt-5.4")
    ap.add_argument("--prompts", default="P1_shipped,P4_unmarked_rule")
    ap.add_argument("--out", default="hard_bank_merge.json")
    args = ap.parse_args()

    items = load_bank()
    prompts = {k: v for k, v in build_prompts().items() if k in args.prompts.split(",")}
    print(f"{len(items)} items, prompts {list(prompts)}")
    results: dict[str, dict] = {}

    # the metadata-only reference: by construction it accepts every update, so
    # it is correct on the 48 update items and wrong on all 12 keep items.
    rows = [decide_keep(ConfidenceRuleMerge(), build_scenario(it)) for it in items]
    n_abst = sum(r is None for r in rows)
    e = score(items, [0 if r is None else r for r in rows])
    e["n_abstained"] = n_abst
    results["ConfidenceRule"] = e
    print(f"{'ConfidenceRule':34} caught={e['violations_caught']:.3f} "
          f"frozen={e['freezing_rate']:.3f} bal={e['balanced']:.3f}")

    for model in args.models.split(","):
        cfg = {"backend": "openai_compatible", "model": model, "cache_dir": "llm_cache",
               "temperature": 0.0, "max_tokens": 8000}
        if model in OPEN_WEIGHT:
            cfg["base_url"] = LOCAL_BASE_URL
        for pid, system in prompts.items():
            strat = ThreeWayLLMMerge(
                backend=build_backend(cfg), system_prompt=system,
                prompt_suffix=_COT_SCHEMA_NOTE if pid == "P6_elicited_cot" else None)
            rows = [decide_keep(strat, build_scenario(it)) for it in items]
            n_abst = sum(r is None for r in rows)
            e = score(items, [0 if r is None else r for r in rows])
            e.update(n_abstained=n_abst, n_llm_calls=strat.n_calls,
                     n_parse_failures=strat.n_parse_failures)
            results[f"{model}:{pid}"] = e
            print(f"{model+':'+pid:34} caught={e['violations_caught']:.3f} "
                  f"frozen={e['freezing_rate']:.3f} bal={e['balanced']:.3f} "
                  f"ci={e['balanced_ci95_family_clustered']} abst={n_abst} "
                  f"parsefail={strat.n_parse_failures}")

    payload = {
        "note": (
            "POST-HOC, EXPLORATORY (added in revision; in neither preregistration). The "
            "specification prompts re-run end-to-end on BranchMem-Hard, where an "
            "obligation-bearing ancestor no longer indicates whether the update should "
            "be rejected. Tests whether P4's .938 on the discrimination bank survives "
            "the removal of that bank's lexical cue. Primary provenance arm: the update "
            "side always carries source=user/confidence=.95, so a field-reading policy "
            "accepts every update and scores 12 keeps wrong by construction. An "
            "abstention is counted as accepting the update, which is the conservative "
            "reading for the keep items and the permissive one for the update items; "
            "n_abstained is reported so either accounting can be recovered."
        ),
        "bank": str(BANK.relative_to(ROOT)),
        "bank_sha256": hashlib.sha256(BANK.read_bytes()).hexdigest(),
        "seed": SEED,
        "n_items": len(items),
        "provenance_arm": {"update_side": STRONG, "other_side": WEAK},
        "results": results,
    }
    (OUT_DIR / args.out).write_text(json.dumps(payload, indent=2))
    print("\nWrote", OUT_DIR / args.out)


if __name__ == "__main__":
    main()
