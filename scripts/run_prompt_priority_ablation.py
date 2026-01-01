#!/usr/bin/env python3
"""Prompt-priority ablation (post-hoc, exploratory; NOT in ANALYSIS_PLAN.md).

The misleading-provenance result shows the frontier merger committing to the
constraint-violating value on 55/60 items. One objection is that the merge
system prompt itself says a user-sourced, higher-confidence update is
"ordinarily more trustworthy" than an inferred, lower-confidence one. On that
reading the experiment shows only that the model obeys an explicit instruction,
not that LLM mergers inherently privilege provenance over meaning.

This script separates the two by running THREE matched system prompts over the
SAME scenarios, changing only the clause that orders the available signals:

  P1 provenance-preferring  the shipped prompt (states the trust ordering)
  P2 neutral                lists the same signals and states no ordering
  P3 constraint-first       explicitly ranks a persistent ancestor constraint
                            above source and confidence

Everything else (template, output schema, the no-timestamps rule, the
abstention instruction) is byte-identical across conditions, and both regimes
use the same generator seeds, so the only moving part is the priority clause.

Both regimes are evaluated:
  misleading   provenance points AWAY from the ancestor constraint
  informative  same items, provenance flipped to point at the target

P3 is expected to help on `misleading`. The question the informative regime
answers is whether it does so by actually reading the constraint or by simply
inverting its trust in the metadata, which would show up as a drop there.

Overriding the system prompt changes the content-hash cache key, so P1 is
served from the locked run's cache and only P2/P3 issue new calls.

Usage: OPENAI_API_KEY=... PYTHONPATH=. python scripts/run_prompt_priority_ablation.py [n]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from branchmem.benchmark.downstream_tasks import generate_downstream_questions
from branchmem.benchmark.misleading_provenance_generator import (
    MisleadingProvenanceConfig,
    generate_misleading_provenance_scenarios,
)
from branchmem.eval.abstention_metrics import compute_abstention_metrics
from branchmem.evaluation.metrics import score_downstream
from branchmem.llm.base import build_backend
from branchmem.merge.three_way_llm import _SYSTEM_PROMPT, ThreeWayLLMMerge

OUT = Path(__file__).resolve().parents[1] / "results" / "final"
MODEL = "gpt-5.4"
SEED = 5150

# The matched informative control: same generator, same seed, same template
# text, with the provenance assignment flipped so the strong signal sits on the
# constraint-compatible branch instead of the contradicting one.
_INFORMATIVE_CONFIG = MisleadingProvenanceConfig(
    misleading_source="inference",
    misleading_confidence=0.55,
    truthful_source="user",
    truthful_confidence=0.95,
)

# The one clause that differs between conditions. P1's text is spliced out of
# the shipped prompt so the three strings differ in nothing else.
_P1_CLAUSE = (
    "Resolve each conflict using whatever signal actually distinguishes the two "
    "updates — e.g. a user-sourced, higher-confidence update is ordinarily more "
    "trustworthy than an inferred, lower-confidence one; the ancestor value can "
    "indicate which branch's update is more likely a genuine correction vs. a "
    "stale duplicate."
)
_P2_CLAUSE = (
    "Resolve each conflict using whatever signal actually distinguishes the two "
    "updates. The signals available to you are each branch's value, its source, "
    "its confidence, and the ancestor value. We state no ordering among them; "
    "decide which one is doing real work for the conflict in front of you."
)
_P3_CLAUSE = (
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

PROMPTS = {
    "P1_provenance_preferring": _SYSTEM_PROMPT,
    "P2_neutral": _SYSTEM_PROMPT.replace(_P1_CLAUSE, _P2_CLAUSE),
    "P3_constraint_first": _SYSTEM_PROMPT.replace(_P1_CLAUSE, _P3_CLAUSE),
}


def _check_prompts() -> None:
    """Fail loudly if the splice silently did nothing."""
    if _P1_CLAUSE not in _SYSTEM_PROMPT:
        raise SystemExit(
            "P1 clause not found verbatim in the shipped system prompt; the "
            "ablation would compare three identical prompts. Re-sync _P1_CLAUSE "
            "with branchmem/merge/three_way_llm.py before running."
        )
    if len({p for p in PROMPTS.values()}) != 3:
        raise SystemExit("the three ablation prompts are not distinct")


def evaluate(scenarios, strat) -> dict:
    rows = []
    for s in scenarios:
        qs = generate_downstream_questions(s)
        rows.extend(score_downstream(qs, strat.merge(s.ancestor, s.branch_a, s.branch_b)).detail)
    cats = tuple(sorted({d["category"] for d in rows}))
    m = compute_abstention_metrics(rows, categories_in_scope=cats)
    return {
        "accuracy": sum(d["correct"] for d in rows) / len(rows),
        "n": len(rows),
        "commit_rate": m.commit_rate,
        "conditional_on_commit_accuracy": m.conditional_on_commit_accuracy,
        "wrong_commit_rate": m.wrong_commit_rate,
        "n_committed": m.n_committed,
        "n_correct_committed": m.n_correct_committed,
        "n_wrong_committed": m.n_wrong_committed,
        "n_flagged": m.n_flagged,
    }


def main() -> None:
    _check_prompts()
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    backend = build_backend(
        {
            "backend": "openai_compatible",
            "model": MODEL,
            "cache_dir": "llm_cache",
            "temperature": 0.0,
            "max_tokens": 8000,
        }
    )

    regimes = {
        "misleading": generate_misleading_provenance_scenarios(n, seed=SEED),
        "informative": generate_misleading_provenance_scenarios(
            n, seed=SEED, config=_INFORMATIVE_CONFIG
        ),
    }

    results: dict[str, dict] = {}
    for regime, scenarios in regimes.items():
        results[regime] = {}
        for label, system in PROMPTS.items():
            strat = ThreeWayLLMMerge(backend=backend, system_prompt=system)
            r = evaluate(scenarios, strat)
            results[regime][label] = r
            print(
                f"{regime:12} {label:26} acc={r['accuracy']:.3f}  "
                f"commit={r['commit_rate']:.3f}  wrong_commit={r['wrong_commit_rate']:.3f}  "
                f"({r['n_committed']} commits, {r['n_wrong_committed']} wrong, "
                f"{r['n_flagged']} flagged)"
            )
        print()

    payload = {
        "note": (
            "Post-hoc, exploratory (not in ANALYSIS_PLAN.md). Prompt-priority ablation: three "
            "matched system prompts differing ONLY in the clause that orders the available "
            "signals, run over the same scenarios in both regimes. Separates obeying an "
            "explicit provenance instruction from an inherent preference for provenance over "
            "meaning. P1 is the shipped prompt and is served from cache."
        ),
        "model": MODEL,
        "n_scenarios": n,
        "seed": SEED,
        "prompt_clauses": {"P1": _P1_CLAUSE, "P2": _P2_CLAUSE, "P3": _P3_CLAUSE},
        "by_regime": results,
    }
    (OUT / "prompt_priority_ablation.json").write_text(json.dumps(payload, indent=2))
    print("Wrote", OUT / "prompt_priority_ablation.json")


if __name__ == "__main__":
    main()
