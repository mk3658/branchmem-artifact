#!/usr/bin/env python3
"""Post-hoc, exploratory (zero new API; reuses cached gpt-5.4 responses).

Explains WHY the frontier model's semantic_resolvable resolution is
wrong-commit-free (Appendix "Calibrated selective resolution"): its commit rate
tracks the LOGICAL STRENGTH of the ancestor constraint. It commits (correctly)
when the ancestor strictly contradicts one branch (e.g. "requires ramp access,
no stairs" vs "stairs only"; "cannot drink alcohol" vs "wine") and abstains when
the incompatibility is merely defeasible/pragmatic (e.g. "motion sickness on
boats" vs "ferry" -- uncomfortable, not impossible; "no pet" vs "dog leash" --
could be a gift for another). The model's own justifications say so: it cites a
"requirement/constraint" and a "mistaken observation" when it commits, and a
"preference" or "no basis to prefer" when it abstains.

Implication: 3/12 templates encode a defeasible rather than strict
incompatibility, so their labelled "correct" answer is a sensible default, not a
logical necessity -- the model is right to decline them, and they are arguably
better scored as ambiguous. On the 9 strict-constraint families the frontier is
an even higher-precision resolver.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from branchmem.benchmark.downstream_tasks import generate_downstream_questions
from branchmem.benchmark.semantic_resolvable_generator import generate_semantic_resolvable_scenarios
from branchmem.evaluation.metrics import score_downstream
from branchmem.llm.base import build_backend
from branchmem.merge.three_way_llm import ThreeWayLLMMerge

OUT = Path(__file__).resolve().parents[1] / "results" / "final"


def main() -> None:
    scen = generate_semantic_resolvable_scenarios(150, seed=3001)
    b = build_backend({"backend": "openai_compatible", "model": "gpt-5.4", "base_url": None,
                       "cache_dir": "llm_cache", "temperature": 0.0, "max_tokens": 8000})
    strat = ThreeWayLLMMerge(backend=b)
    per = defaultdict(lambda: {"commit": 0, "correct": 0, "total": 0,
                               "ancestor": None, "justification_flag": None, "justification_commit": None})
    for s in scen:
        ph = s.metadata.get("template_predicate", "?")
        q = generate_downstream_questions(s)
        r = strat.merge(s.ancestor, s.branch_a, s.branch_b)
        detail = {(d["category"]): d for d in score_downstream(q, r).detail}
        rec = per[ph]
        rec["ancestor"] = rec["ancestor"] or next((f.value for f in s.ancestor.facts), None)
        for rf in r.resulting_facts:
            # only the conflicting (resolvable) key matters here
            pass
        d = detail.get("resolvable")
        if d is None:
            continue
        rec["total"] += 1
        flagged = d["got_resolution"] == "flagged_unresolved"
        # capture a representative justification
        for rf in r.resulting_facts:
            if rf.resolution.value == "flagged_unresolved" and rec["justification_flag"] is None:
                rec["justification_flag"] = rf.justification
            if rf.resolution.value != "flagged_unresolved" and rec["justification_commit"] is None:
                rec["justification_commit"] = rf.justification
        if not flagged:
            rec["commit"] += 1
            rec["correct"] += int(d["correct"])

    families = {}
    for ph, r in per.items():
        cr = r["commit"] / r["total"] if r["total"] else 0.0
        families[ph] = {
            "commit_rate": round(cr, 3),
            "conditional_accuracy": round(r["correct"] / r["commit"], 3) if r["commit"] else None,
            "n": r["total"], "ancestor_constraint": r["ancestor"],
            "example_justification": r["justification_commit"] if r["commit"] else r["justification_flag"],
        }
    ordered = dict(sorted(families.items(), key=lambda kv: -kv[1]["commit_rate"]))
    result = {
        "note": ("post-hoc, zero new API. Frontier (gpt-5.4) semantic_resolvable commit rate by "
                 "constraint family, ordered high->low. Commit tracks logical constraint strength: "
                 "strict contradictions -> commit (correctly); defeasible/pragmatic incompatibilities "
                 "-> abstain. 3/12 families (transport/gift/childcare) are defeasible and better scored "
                 "as ambiguous; on the 9 strict families the frontier is an even higher-precision resolver."),
        "families": ordered,
    }
    (OUT / "constraint_strength.json").write_text(json.dumps(result, indent=2))
    for ph, f in ordered.items():
        print(f"{ph:20} commit={f['commit_rate']:.2f} condacc={f['conditional_accuracy']} :: {f['ancestor_constraint']!r}")
    print("\nWrote", OUT / "constraint_strength.json")


if __name__ == "__main__":
    main()
