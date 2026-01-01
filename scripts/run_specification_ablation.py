#!/usr/bin/env python3
"""Is the discrimination failure a property of the model or of the merge prompt?

Post-hoc and exploratory: written after the preregistered runs, and in
neither ANALYSIS_PLAN.md nor its addendum. Registered predictions are not
claimed for anything here.

The preregistered discrimination experiment runs the SHIPPED merge system
prompt, which names source and confidence as the signals to use and never asks
whether the stored entry is a standing rule. One objection to the diagnosis is
that this under-specifies the task rather than showing that mergers cannot
discriminate, and that the earlier prompt ablation does not settle it because

  (a) it ran on the older 60-item misleading-provenance bank, not on the frozen
      320-item discrimination bank, and
  (b) its only constraint-aware condition (P3) ASSERTS constraint primacy and
      enumerates the categories that count, which pre-solves the
      classification the experiment is about.

The missing control is a prompt that raises the question in general terms
without supplying the answer. This script runs six matched system prompts over
the frozen discrimination bank. Every condition shares the template text, the
output schema, the no-timestamp rule and the abstention instruction
byte-for-byte; only the inserted clause differs:

  P0 unprimed         mentions no metadata, no trust ordering and no standing
                      rules at all: the merger is told only that two branches
                      disagree, that timestamps are unusable, and how to flag
                      an unresolvable conflict. Unlike P2-P6 this replaces
                      three spans of the shipped prompt, not one, because the
                      shipped prompt names source and confidence twice more
                      (in the input description and in the abstention
                      instruction). It is the only condition that is not a
                      one-clause diff, and it is the control for "the model was
                      primed to read the fields".
  P1 shipped          the locked run's prompt: states the trust ordering.
  P2 neutral          same four signals, no ordering stated.
  P3 marked           asserts constraint primacy AND enumerates qualifying
                      categories (the earlier ablation's condition).
  P4 unmarked+rule    asks the model to FIRST decide whether the ancestor is a
                      standing commitment or an ordinary value, and gives the
                      decision rule for each case -- but names no categories
                      and never says which items are which. This is the
                      control that objection calls for.
  P5 unmarked, norule only raises the question ("consider what kind of entry the
                      ancestor is"), with no decision rule at all. Isolates
                      whether merely prompting the question suffices.
  P6 elicited CoT     the shipped prompt plus a required reasoning field, so the
                      model must analyse each conflict in writing before
                      resolving it. This is ELICITED chain-of-thought, not a
                      reasoning-mode model; we have no API access to one.

Scenario construction, arms, seeds and scoring are imported from
`run_discrimination_experiment.py`, so the items are identical to the
preregistered run and P1 is served from its cache.

The primary measure is the preregistered one: per-type accuracy on
`immutable_constraint` and `legitimate_override` under the `update_favoured`
arm, the gap D between them, and their balanced accuracy. Intervals resample
template families.

Usage:
  OPENAI_API_KEY=... PYTHONPATH=. python scripts/run_specification_ablation.py \
      --models gpt-5.4 --types immutable_constraint,legitimate_override \
      --out specification_ablation_gpt54.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

from branchmem.llm.base import build_backend
from branchmem.merge.three_way_llm import _SYSTEM_PROMPT, ThreeWayLLMMerge

from run_discrimination_experiment import (  # type: ignore
    ARMS,
    BANK,
    CORRECT,
    LOCAL_BASE_URL,
    OPEN_WEIGHT,
    SEED,
    build_scenario,
    decide,
    load_bank,
)

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "final"
CRIT = ("immutable_constraint", "legitimate_override")

# ---------------------------------------------------------------------------
# The six system prompts. P1 is the shipped prompt verbatim; every other
# condition is built from it by replacing exactly one clause, so the diff
# between any two conditions is the priority/specification clause and nothing
# else. The clause being replaced is the shipped trust-ordering sentence.
# ---------------------------------------------------------------------------

_SHIPPED_CLAUSE = (
    "Resolve each conflict "
    "using whatever signal actually distinguishes the two updates — e.g. a "
    "user-sourced, higher-confidence update is ordinarily more trustworthy than an "
    "inferred, lower-confidence one; the ancestor value can indicate which branch's "
    "update is more likely a genuine correction vs. a stale duplicate."
)
assert _SHIPPED_CLAUSE in _SYSTEM_PROMPT, "shipped prompt changed; update this ablation"

# ---------------------------------------------------------------------------
# P0: the unprimed control. The shipped prompt mentions the metadata three
# times -- when it describes the input, in the trust-ordering clause, and in
# the abstention instruction -- so all three spans are replaced. Nothing in
# the result names source, confidence, a trust ordering, or a standing rule.
# The branch payloads still CARRY source and confidence (they are the data,
# and removing them would change the task rather than the prompt); P0 tests
# only whether being told about them is what produces the field-reading.
# ---------------------------------------------------------------------------

_SHIPPED_INPUT_DESC = (
    "For each conflicting (entity, predicate) key, you are given the common "
    "ancestor's value (if any) and each branch's current value, source, and "
    "confidence. 'source' is how that update was recorded: \"user\" means the "
    "user explicitly stated or corrected it; \"inference\" means the agent "
    "inferred or guessed it; \"observation\" means the agent observed it "
    "indirectly."
)
assert _SHIPPED_INPUT_DESC in _SYSTEM_PROMPT, "input description changed; update P0"

_P0_INPUT_DESC = (
    "For each conflicting (entity, predicate) key, you are given the common "
    "ancestor's value (if any) and each branch's current record."
)

_SHIPPED_ABSTAIN = (
    "If source, confidence, and ancestor context give no real basis to prefer "
    "one branch over the other (e.g. both are equally plausible, "
    "equally-sourced preference changes)"
)
assert _SHIPPED_ABSTAIN in _SYSTEM_PROMPT, "abstention clause changed; update P0"

_P0_ABSTAIN = (
    "If nothing you are given is a real basis to prefer one branch over the "
    "other (e.g. both are equally plausible)"
)

_P0_CLAUSE = "Resolve each conflict."


def _build_p0() -> str:
    s = _SYSTEM_PROMPT.replace(_SHIPPED_INPUT_DESC, _P0_INPUT_DESC)
    s = s.replace(_SHIPPED_CLAUSE, _P0_CLAUSE)
    s = s.replace(_SHIPPED_ABSTAIN, _P0_ABSTAIN)
    for banned in ("source", "confidence", "trust", "standing", "rule"):
        assert banned not in s.lower().replace("unresolvable", ""), (
            f"P0 still mentions {banned!r}: {s}")
    return s


_P2_NEUTRAL = (
    "Resolve each conflict using whatever signal actually distinguishes the two "
    "updates. The signals available to you are each branch's value, its source, "
    "its confidence, and the ancestor value. We state no ordering among them; "
    "decide which one is doing real work for the conflict in front of you."
)

_P3_MARKED = (
    "Resolve each conflict using whatever signal actually distinguishes the two "
    "updates, subject to one priority. If the ancestor value records a persistent "
    "constraint about the entity (for example a medical restriction, an allergy, a "
    "permanent physical limitation, or another standing requirement) and one "
    "branch's value would violate that constraint, keep the value that is "
    "compatible with the constraint, EVEN IF the violating branch is user-sourced "
    "or carries higher confidence. A persistent constraint of this kind is not "
    "superseded by a more confident or more authoritatively sourced update; it is "
    "superseded only by an update that explicitly retracts the constraint itself. "
    "Source and confidence decide the conflict only when no such ancestor "
    "constraint is at stake."
)

# P4: raises the classification question and states what follows from each
# answer, but supplies no categories, no examples, and no indication of which
# items are which. The model must do the classification itself.
_P4_UNMARKED_RULE = (
    "Resolve each conflict in two steps. First, decide for yourself what kind of "
    "entry the ancestor value is: a standing commitment that remains in force "
    "until it is explicitly withdrawn, or an ordinary value that a later, better "
    "update may legitimately replace. Then decide the conflict accordingly. If it "
    "is a standing commitment and one branch's value is incompatible with it, keep "
    "the compatible value even if the incompatible branch is user-sourced or "
    "carries higher confidence. If it is an ordinary value, let source and "
    "confidence decide, preferring a user-sourced, higher-confidence update over "
    "an inferred, lower-confidence one."
)

# P5: the weakest possible nudge. Raises the question and stops.
_P5_UNMARKED_NORULE = (
    "Resolve each conflict using whatever signal actually distinguishes the two "
    "updates. Before you decide, consider what kind of entry the ancestor value "
    "is and what that implies for the conflict in front of you."
)

_COT_SUFFIX = (
    " For each key, before you give a resolution, write out your analysis of that "
    "conflict step by step: what the ancestor value commits the entity to, what "
    "each branch would make true, whether the two are compatible, and only then "
    "which to keep. Put that analysis in the 'reasoning' field."
)

_COT_SCHEMA_NOTE = (
    "\n\nFor EVERY key, include an additional field \"reasoning\" containing your "
    "step-by-step analysis of that conflict, written BEFORE you commit to the "
    "resolution value. The \"reasoning\" field is required."
)


def build_prompts() -> dict[str, str]:
    return {
        "P0_unprimed": _build_p0(),
        "P1_shipped": _SYSTEM_PROMPT,
        "P2_neutral": _SYSTEM_PROMPT.replace(_SHIPPED_CLAUSE, _P2_NEUTRAL),
        "P3_marked": _SYSTEM_PROMPT.replace(_SHIPPED_CLAUSE, _P3_MARKED),
        "P4_unmarked_rule": _SYSTEM_PROMPT.replace(_SHIPPED_CLAUSE, _P4_UNMARKED_RULE),
        "P5_unmarked_norule": _SYSTEM_PROMPT.replace(_SHIPPED_CLAUSE, _P5_UNMARKED_NORULE),
        "P6_elicited_cot": _SYSTEM_PROMPT + _COT_SUFFIX,
    }


def gap_ci(rows: list[dict], n_boot: int = 2000) -> tuple[float | None, list[float] | None]:
    """D and its interval, resampling families so the two types move together.

    Identical to `analyze_discrimination.gap_ci`; duplicated here so this
    script stands alone when only a subset of types was run.
    """
    by_fam: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["type"] in CRIT:
            by_fam[r["family"]][r["type"]].append(int(bool(r["correct"])))
    fams = [f for f, d in by_fam.items() if all(t in d for t in CRIT)]
    if len(fams) < 3:
        return None, None

    def d_of(sample: list[str]) -> float:
        c = [v for f in sample for v in by_fam[f]["immutable_constraint"]]
        o = [v for f in sample for v in by_fam[f]["legitimate_override"]]
        return sum(o) / len(o) - sum(c) / len(c)

    point = d_of(fams)
    rng = random.Random(SEED)
    boots = sorted(d_of([rng.choice(fams) for _ in fams]) for _ in range(n_boot))
    return round(point, 4), [round(boots[int(.025 * n_boot)], 4),
                             round(boots[int(.975 * n_boot)], 4)]


def balanced_ci(rows: list[dict], n_boot: int = 2000) -> list[float] | None:
    """Family-clustered interval on the balanced accuracy of the critical pair."""
    by_fam: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["type"] in CRIT:
            by_fam[r["family"]][r["type"]].append(int(bool(r["correct"])))
    fams = [f for f, d in by_fam.items() if all(t in d for t in CRIT)]
    if len(fams) < 3:
        return None

    def bal_of(sample: list[str]) -> float:
        c = [v for f in sample for v in by_fam[f]["immutable_constraint"]]
        o = [v for f in sample for v in by_fam[f]["legitimate_override"]]
        return (sum(c) / len(c) + sum(o) / len(o)) / 2

    rng = random.Random(SEED)
    boots = sorted(bal_of([rng.choice(fams) for _ in fams]) for _ in range(n_boot))
    return [round(boots[int(.025 * n_boot)], 4), round(boots[int(.975 * n_boot)], 4)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True, help="comma-separated model ids")
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompts", default="", help="comma-separated prompt ids (default: all)")
    ap.add_argument("--arms", default="update_favoured")
    ap.add_argument("--types", default="", help="comma-separated types (default: all four)")
    ap.add_argument("--max-tokens", type=int, default=8000,
                    help="output budget. Local reasoning models served by Ollama default "
                         "to a 4096-token context, so a budget above that makes them emit "
                         "scratchpad until truncation and never reach the JSON; pass "
                         "~3000 for those.")
    args = ap.parse_args()

    items = load_bank()
    if args.types:
        keep = set(args.types.split(","))
        unknown = keep - set(CORRECT)
        if unknown:
            raise SystemExit(f"unknown types: {sorted(unknown)}")
        items = [it for it in items if it["type"] in keep]

    prompts = build_prompts()
    if args.prompts:
        want = args.prompts.split(",")
        unknown = [p for p in want if p not in prompts]
        if unknown:
            raise SystemExit(f"unknown prompts: {unknown}; have {list(prompts)}")
        prompts = {k: prompts[k] for k in want}

    arms = args.arms.split(",")
    print(f"{len(items)} items, {len(prompts)} prompts, arms {arms}")

    results: dict[str, dict] = {}
    raw: list[dict] = []

    for model in args.models.split(","):
        cfg = {"backend": "openai_compatible", "model": model, "cache_dir": "llm_cache",
               "temperature": 0.0, "max_tokens": args.max_tokens}
        if model in OPEN_WEIGHT:
            cfg["base_url"] = LOCAL_BASE_URL
        results[model] = {}
        for pid, system in prompts.items():
            # The CoT condition also needs the output schema to carry a
            # reasoning field; every other condition leaves the user template
            # byte-identical to the preregistered run.
            strat = ThreeWayLLMMerge(
                backend=build_backend(cfg),
                system_prompt=system,
                prompt_suffix=_COT_SCHEMA_NOTE if pid == "P6_elicited_cot" else None,
            )
            results[model][pid] = {}
            for arm in arms:
                rows = []
                for item in items:
                    r = decide(strat, build_scenario(item, arm))
                    r.update(policy=f"{model}:{pid}", model=model, prompt_id=pid)
                    rows.append(r)
                    raw.append(r)
                present = sorted({r["type"] for r in rows})
                by_type = {
                    t: {
                        "n": len([r for r in rows if r["type"] == t]),
                        "accuracy": round(
                            sum(r["correct"] for r in rows if r["type"] == t)
                            / len([r for r in rows if r["type"] == t]), 4),
                    }
                    for t in present
                }
                d, dci = gap_ci(rows)
                ac = by_type.get("immutable_constraint", {}).get("accuracy")
                ao = by_type.get("legitimate_override", {}).get("accuracy")
                have = ac is not None and ao is not None
                results[model][pid][arm] = {
                    "n": len(rows),
                    "by_type": by_type,
                    "discrimination_gap": d,
                    "discrimination_gap_ci95_family_clustered": dci,
                    "balanced_accuracy_critical": round((ac + ao) / 2, 4) if have else None,
                    "balanced_accuracy_ci95_family_clustered": balanced_ci(rows) if have else None,
                    "coverage": round(sum(not r["abstained"] for r in rows) / len(rows), 4),
                    "wrong_commit_rate": round(sum(r["wrong_commit"] for r in rows) / len(rows), 4),
                    "took_update_rate": round(sum(r["took_update"] for r in rows) / len(rows), 4),
                    # Unparseable replies are NOT abstentions. Reasoning models
                    # emit scratchpads that can swallow the JSON, so this is
                    # reported rather than silently scored as a decline.
                    "n_llm_calls": strat.n_calls,
                    "n_parse_failures": strat.n_parse_failures,
                    "parse_failure_rate": (round(strat.n_parse_failures / strat.n_calls, 4)
                                           if strat.n_calls else None),
                }
                e = results[model][pid][arm]
                if have:
                    print(f"{model:16} {pid:20} {arm:16} constr={ac:.3f} overr={ao:.3f} "
                          f"D={d:+.3f} bal={e['balanced_accuracy_critical']:.3f} "
                          f"cov={e['coverage']:.3f} "
                          f"parsefail={e['parse_failure_rate']}")
                else:
                    print(f"{model:16} {pid:20} {arm:16} cov={e['coverage']:.3f}")

    payload = {
        "note": (
            "POST-HOC, EXPLORATORY (in neither ANALYSIS_PLAN.md nor its "
            "addendum). Seven matched merge system prompts over the frozen 320-item "
            "discrimination bank. P1-P6 differ from the shipped prompt in exactly one "
            "clause, the one that specifies how to weigh the signals. P4 and P5 are the "
            "controls for the under-specification objection: they raise the "
            "standing-rule question "
            "WITHOUT naming qualifying categories or indicating which items are which, "
            "where P3 asserts constraint primacy and enumerates categories. P6 is "
            "ELICITED chain-of-thought, not a reasoning-mode model, to which we have no "
            "API access. P0 is the one condition that is NOT a one-clause diff: the "
            "shipped prompt names the metadata in three places (the input description, "
            "the trust-ordering clause, the abstention instruction) and P0 replaces all "
            "three, so it mentions no metadata, no trust ordering and no standing rule "
            "at all. The branch payloads still carry source and confidence, since those "
            "are the data rather than the prompt; P0 tests only whether being TOLD "
            "about them is what produces the field-reading. Note P0's lower coverage: "
            "an unprimed merger abstains more, so read took_update_rate and "
            "wrong_commit_rate alongside the accuracies. Scenario "
            "construction, arms and scoring are imported from "
            "run_discrimination_experiment.py so the items match the preregistered run "
            "exactly and P1 is cache-served. Intervals resample template families."
        ),
        "bank": str(BANK.relative_to(ROOT)),
        "bank_sha256": hashlib.sha256(BANK.read_bytes()).hexdigest(),
        "seed": SEED,
        "n_items": len(items),
        "max_tokens": args.max_tokens,
        "types_run": sorted({it["type"] for it in items}),
        "arms": arms,
        "prompt_ids": list(prompts),
        "prompt_texts": prompts,
        "provenance_arms": {k: {"update_side": v[0], "other_side": v[1]} for k, v in ARMS.items()},
        "results": results,
        "raw": raw,
    }
    (OUT_DIR / args.out).write_text(json.dumps(payload, indent=2))
    print("\nWrote", OUT_DIR / args.out)


if __name__ == "__main__":
    main()
