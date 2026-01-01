#!/usr/bin/env python3
"""Are the two fixes complementary or redundant?

The paper offers two remedies, measured on the same bank with the same metric
but never composed:

  the standing check   shown the stored entry WITH the proposed update, one
                       general question per key, reaching .948 balanced
                       (`run_standing_detection.py`, scope ancestor_update)
  the specification    the merge prompt that asks the model to classify the
  prompt (P4)          entry and says what follows, .958 end-to-end here
                       (`run_hard_bank_merge.py`)

A deployment would have to pick one or chain them, so this composes them per
item and reports whether chaining beats either part. Two compositions are
scored, because they trade the two error directions against each other:

  EITHER   reject the update if the gate rejects it or the merge keeps the
           stored value. Maximises violations caught; freezing is the union of
           the two policies' freezes.
  BOTH     reject only when gate and merge agree. Minimises freezing; a
           violation slips through unless both catch it.

It also reports how often the two agree, which is the thing a composition
argument actually turns on: two policies that fail on the same items compose
to nothing, and the per-item table says which case this is.

No model is called that is not already cached: both stages re-run the prompts
the two scripts already ran, so the whole thing is cache-served and
`OPENAI_API_KEY=cache-only-sentinel` is enough. Post-hoc and exploratory, like
everything it composes.

Usage:
  OPENAI_API_KEY=cache-only-sentinel PYTHONPATH=.:scripts \
      python scripts/run_composed_pipeline.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from branchmem.llm.base import build_backend
from branchmem.merge.three_way_llm import ThreeWayLLMMerge

from run_specification_ablation import build_prompts  # type: ignore
from run_standing_detection import load_bank, predict_llm, score  # type: ignore
from run_hard_bank_merge import build_scenario, decide_keep  # type: ignore

ROOT = Path(__file__).resolve().parents[1]
BANK = ROOT / "data" / "hard_standing_bank.json"
OUT = ROOT / "results" / "final" / "composed_pipeline.json"
GATE_SCOPE = "ancestor_update"
MERGE_PROMPT = "P4_unmarked_rule"


def _merge_decisions(model: str, items: list[dict]) -> tuple[list[int], int]:
    """Per-item keep decisions under the specification prompt.

    An abstention is scored as accepting the update, as in
    `run_hard_bank_merge.py`, so the two scripts' numbers stay comparable.
    """
    cfg = {"backend": "openai_compatible", "model": model, "cache_dir": "llm_cache",
           "temperature": 0.0, "max_tokens": 8000}
    strat = ThreeWayLLMMerge(backend=build_backend(cfg),
                             system_prompt=build_prompts()[MERGE_PROMPT])
    raw = [decide_keep(strat, build_scenario(it)) for it in items]
    return [0 if r is None else r for r in raw], sum(r is None for r in raw)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-5.4")
    args = ap.parse_args()

    items = load_bank()
    truth = [int(it["correct_action"] == "keep") for it in items]

    gate, gate_unparsed = predict_llm(items, GATE_SCOPE, args.model)
    merge, merge_abstained = _merge_decisions(args.model, items)

    either = [max(g, m) for g, m in zip(gate, merge)]
    both = [min(g, m) for g, m in zip(gate, merge)]

    arms = {
        "gate_only": gate,
        "merge_only": merge,
        "composed_either_rejects": either,
        "composed_both_must_reject": both,
    }
    scored = {name: score(items, pred) for name, pred in arms.items()}

    agree = sum(g == m for g, m in zip(gate, merge))
    # Where each stage is wrong, and whether those sets overlap: this is what
    # decides whether composing helps or only moves the error around.
    gate_wrong = {i for i, (g, t) in enumerate(zip(gate, truth)) if g != t}
    merge_wrong = {i for i, (m, t) in enumerate(zip(merge, truth)) if m != t}
    rescued = sorted(gate_wrong - merge_wrong)
    introduced = sorted(merge_wrong - gate_wrong)
    shared = sorted(gate_wrong & merge_wrong)

    payload = {
        "note": (
            "POST-HOC, EXPLORATORY (in neither preregistration). The standing check and "
            "the specification prompt composed per item on BranchMem-Hard, to answer "
            "whether the paper's two remedies are complementary or redundant. Both "
            "stages are cache-served; no new model call is made. An abstention in the "
            "merge stage counts as accepting the update, as in hard_bank_merge.json."),
        "bank": str(BANK.relative_to(ROOT)),
        "bank_sha256": hashlib.sha256(BANK.read_bytes()).hexdigest(),
        "model": args.model,
        "gate": {"script": "run_standing_detection.py", "scope": GATE_SCOPE,
                 "n_unparsed_replies": gate_unparsed},
        "merge": {"script": "run_hard_bank_merge.py", "prompt": MERGE_PROMPT,
                  "n_abstained": merge_abstained},
        "n_items": len(items),
        "arms": scored,
        "stage_agreement": {
            "items_both_stages_agree": agree,
            "n_items": len(items),
            "rate": round(agree / len(items), 4),
            "gate_wrong_merge_right": rescued,
            "merge_wrong_gate_right": introduced,
            "both_wrong": shared,
        },
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")

    print(f"{len(items)} items, model {args.model}\n")
    print(f"{'arm':32} {'caught':>7} {'frozen':>7} {'bal':>7} {'95% CI':>18}")
    for name, e in scored.items():
        print(f"{name:32} {e['violations_caught']:7.3f} {e['freezing_rate']:7.3f} "
              f"{e['balanced']:7.3f} {str(e['balanced_ci95_family_clustered']):>18}")
    print(f"\nstages agree on {agree}/{len(items)} items "
          f"({agree/len(items):.3f})")
    print(f"  gate wrong where merge is right: {len(rescued)}")
    print(f"  merge wrong where gate is right: {len(introduced)}")
    print(f"  both wrong on the same item:     {len(shared)}")
    print("\nwrote", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
