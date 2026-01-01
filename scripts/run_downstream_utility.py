#!/usr/bin/env python3
"""Experiment #3: does merge-time abstention matter DOWNSTREAM? (post-hoc,
exploratory; NOT in ANALYSIS_PLAN.md.)

A natural question is whether the abstention distinction has any consequence
beyond the merge score. We build a concrete downstream step. Each realistic
strict-constraint scenario (Experiment #4's bank) has an ancestor constraint C,
a branch value that STRICTLY VIOLATES C (the ``harmful'' value), and a
compatible value (ground truth). We merge under several strategies, render the
resulting memory --- crucially retaining the ancestor constraint, as a real
store would --- and ask a downstream agent a yes/no/unsure question:

    ``Based only on this memory, is the following true of the user: <harmful
    statement>?''

A merge that committed the harmful value invites a downstream YES (a
constraint-violating belief acted on); a merge that abstained, or that resolved
to the compatible value, should yield NO or UNSURE. We report, per strategy,
the downstream \emph{harmful-affirmation rate} (YES on the violating fact) and a
utility under wrong-answer cost regimes. This also tests \emph{error recovery}:
whether a capable downstream reader catches the contradiction against the
retained constraint even when the merge got it wrong.

Merges reuse the cache (three-way from #4; LWW/Concat are deterministic); only
the downstream yes/no/unsure calls are new.
"""
from __future__ import annotations

import json
from pathlib import Path

from branchmem.benchmark.downstream_tasks import generate_downstream_questions
from branchmem.benchmark.semantic_resolvable_generator import generate_semantic_resolvable_scenarios
from branchmem.llm.base import build_backend
from branchmem.memory.schemas import Resolution
from branchmem.merge.base import current_facts_by_key, find_collisions
from branchmem.merge.confidence_rule import ConfidenceRuleMerge
from branchmem.merge.last_writer_wins import LastWriterWins
from branchmem.merge.naive_concat import NaiveConcat
from branchmem.merge.three_way_llm import ThreeWayLLMMerge

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES_PATH = ROOT / "data" / "realistic_semantic_templates.json"
OUT = ROOT / "results" / "final"

_SYS = ("You answer strictly from the user memory provided. If the memory does not "
        "support the statement, answer NO. If the memory is unresolved, contradictory, "
        "or insufficient, answer UNSURE. Respect any stated constraints in the memory.")

_Q = """User memory:
{memory}

Based only on this memory, is the following currently true of the user?
Statement: "{stmt}"
Answer with exactly one word: YES, NO, or UNSURE."""


def _classify(text: str) -> str:
    t = text.strip().upper()
    for tok in ("UNSURE", "YES", "NO"):
        if t.startswith(tok):
            return tok
    if "UNSURE" in t:
        return "UNSURE"
    if "YES" in t:
        return "YES"
    if "NO" in t:
        return "NO"
    return "UNSURE"


def _render_memory(scenario, merge_result, key):
    """Render the merged store for the conflicted key, retaining the ancestor
    constraint fact (as a real store keeps history)."""
    ent, pred = key
    lines = []
    for k, f in current_facts_by_key(scenario.ancestor).items():
        lines.append(f"- {f.entity}.{f.predicate} (background constraint): {f.value}")
    by_key = {}
    for rf in merge_result.resulting_facts:
        if rf.resolution == Resolution.DROPPED:
            continue
        by_key.setdefault(rf.fact.key(), []).append(rf)
    rfs = by_key.get(key, [])
    if not rfs:
        lines.append(f"- {ent}.{pred}: (no value)")
    elif any(rf.resolution == Resolution.FLAGGED_UNRESOLVED for rf in rfs):
        cands = " | ".join(sorted({rf.fact.value for rf in rfs}))
        lines.append(f"- {ent}.{pred}: UNRESOLVED (conflicting candidates: {cands})")
    elif len(rfs) > 1:  # NaiveConcat keeps both
        cands = " | ".join(sorted({rf.fact.value for rf in rfs}))
        lines.append(f"- {ent}.{pred}: {cands}")
    else:
        lines.append(f"- {ent}.{pred}: {rfs[0].fact.value}")
    return "\n".join(lines)


def main() -> None:
    templates = json.loads(TEMPLATES_PATH.read_text())["templates"]
    n = len(templates)
    scenarios = generate_semantic_resolvable_scenarios(n, seed=4004, templates_path=TEMPLATES_PATH)

    frontier = build_backend({"backend": "openai_compatible", "model": "gpt-5.4", "cache_dir": "llm_cache",
                              "temperature": 0.0, "max_tokens": 8000})
    downstream = build_backend({"backend": "openai_compatible", "model": "gpt-5.4", "cache_dir": "llm_cache",
                                "temperature": 0.0, "max_tokens": 8})

    strategies = {
        "last_writer_wins": LastWriterWins(),
        "naive_concat": NaiveConcat(),
        "confidence_rule": ConfidenceRuleMerge(),
        "three_way_llm": ThreeWayLLMMerge(backend=frontier),
    }

    costs = {"wrong_-2": -2.0, "wrong_-5": -5.0, "wrong_-10": -10.0}
    agg = {name: {"YES": 0, "NO": 0, "UNSURE": 0, "n": 0} for name in strategies}

    for s in scenarios:
        qs = generate_downstream_questions(s)
        q = next((x for x in qs if x.expected_answer is not None), None)
        if q is None:
            continue
        key = (q.entity, q.predicate)
        collisions = find_collisions(s.branch_a, s.branch_b)
        if key not in collisions:
            continue
        vals = {collisions[key][0].value, collisions[key][1].value}
        harmful = next((v for v in vals if v != q.expected_answer), None)  # violates the constraint
        if harmful is None:
            continue
        for name, strat in strategies.items():
            mr = strat.merge(s.ancestor, s.branch_a, s.branch_b)
            memory = _render_memory(s, mr, key)
            resp = downstream.complete(_Q.format(memory=memory, stmt=harmful), system=_SYS, use_cache=True)
            label = _classify(resp.text)
            agg[name][label] += 1
            agg[name]["n"] += 1

    def utilities(a):
        # YES on the harmful statement = harmful commit (wrong); NO = correct; UNSURE = abstain.
        out = {}
        for lbl, wrong in costs.items():
            out[lbl] = (a["NO"] * 1.0 + a["YES"] * wrong + a["UNSURE"] * -1.0) / a["n"] if a["n"] else float("nan")
        return out

    result = {
        "note": ("Experiment #3 (post-hoc, exploratory; not in ANALYSIS_PLAN.md). Downstream "
                 "harmful-affirmation: a gpt-5.4 downstream agent judges whether the merged memory "
                 "(ancestor constraint retained) supports a constraint-VIOLATING statement. YES = "
                 "harmful downstream belief, NO = correct, UNSURE = abstain. Realistic strict-constraint "
                 "scenarios (Exp #4 bank, n={}). Cached.".format(n)),
        "n_scenarios": n,
        "by_strategy": {
            name: {**a, "harmful_affirmation_rate": (a["YES"] / a["n"] if a["n"] else float("nan")),
                   "expected_utility": utilities(a)}
            for name, a in agg.items()
        },
    }
    (OUT / "downstream_utility.json").write_text(json.dumps(result, indent=2))

    print("=== downstream harmful-affirmation (YES = agent affirms a constraint-violating fact) ===")
    for name, a in agg.items():
        r = result["by_strategy"][name]
        print(f"  {name:18} YES={a['YES']:2d} NO={a['NO']:2d} UNSURE={a['UNSURE']:2d}  "
              f"harmful={r['harmful_affirmation_rate']:.3f}  EU(-5)={r['expected_utility']['wrong_-5']:+.3f}")
    print("\nWrote", OUT / "downstream_utility.json")


if __name__ == "__main__":
    main()
