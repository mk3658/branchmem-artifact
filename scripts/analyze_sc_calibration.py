#!/usr/bin/env python3
"""Self-consistency as CALIBRATION, not accuracy. Self-consistency (k-sample
majority) did not raise mean semantic_resolvable accuracy. But does inter-sample
AGREEMENT predict correctness? If the model is reliably right exactly when its k
samples agree, then "commit only when agreement >= tau, else abstain" turns the
LLM into a reliable-when-it-commits resolver -- a within-boundary capability the
deterministic rule (0.000 here) lacks.

Reuses cached samples (zero new API when the k samples are already cached).

Usage: OPENAI_API_KEY=... PROBE_MODEL=gpt-5.4 python scripts/analyze_sc_calibration.py [n] [k]
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter

from branchmem.benchmark.downstream_tasks import generate_downstream_questions
from branchmem.benchmark.semantic_resolvable_generator import generate_semantic_resolvable_scenarios
from branchmem.llm.base import build_backend
from branchmem.merge.base import current_facts_by_key, find_collisions
from branchmem.merge.three_way_llm import _PROMPT_TEMPLATE, _SYSTEM_PROMPT, _parse_response

MODEL = os.environ.get("PROBE_MODEL", "llama3.1:8b")
BASE_URL = os.environ.get("PROBE_BASE_URL") or None


def _collision_prompt(ancestor, collisions):
    af = current_facts_by_key(ancestor)
    def fj(f): return {"value": f.value, "source": f.source, "confidence": f.confidence}
    aj = json.dumps({f"{k[0]}|{k[1]}": af[k].value for k in collisions if k in af}, indent=2)
    baj = json.dumps({f"{k[0]}|{k[1]}": fj(v[0]) for k, v in collisions.items()}, indent=2)
    bbj = json.dumps({f"{k[0]}|{k[1]}": fj(v[1]) for k, v in collisions.items()}, indent=2)
    return _PROMPT_TEMPLATE.format(ancestor_json=aj, branch_a_json=baj, branch_b_json=bbj)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    scenarios = generate_semantic_resolvable_scenarios(150, seed=3001)[:n]
    backend = build_backend({"backend": "openai_compatible", "model": MODEL, "base_url": BASE_URL,
                             "cache_dir": "llm_cache", "temperature": 0.7, "max_tokens": 8000})
    # per resolvable key: (agreement_count_of_majority, committed_value_or_FLAG, expected)
    records = []
    for s in scenarios:
        qs = generate_downstream_questions(s)
        expected = {(q.entity, q.predicate): q.expected_answer for q in qs if q.category == "resolvable"}
        collisions = find_collisions(s.branch_a, s.branch_b)
        if not collisions:
            continue
        prompt = _collision_prompt(s.ancestor, collisions)
        votes = {f"{a}|{b}": [] for (a, b) in collisions}
        for i in range(k):
            sysp = _SYSTEM_PROMPT + f"\n\n[Independent reasoning pass {i+1} of {k}. Reason freshly from the facts.]"
            resp = backend.complete(prompt, system=sysp, use_cache=True)
            by_key = _parse_response(resp.text)
            for ks in votes:
                d = by_key.get(ks)
                votes[ks].append(str(d["value"]) if (d and d.get("resolution") != "flagged_unresolved" and d.get("value")) else "__FLAG__")
        for (a, b) in collisions:
            ks = f"{a}|{b}"
            exp = expected.get((a, b))
            if exp is None:
                continue
            top_val, top_cnt = Counter(votes[ks]).most_common(1)[0]
            records.append({"agreement": top_cnt, "value": top_val, "expected": exp,
                            "committed": top_val != "__FLAG__", "correct": top_val == exp})

    total = len(records)
    print(f"MODEL={MODEL}  n_scenarios={n}  k={k}  resolvable_keys={total}\n")
    # baseline: majority-vote accuracy (commit majority even if low agreement; flag scores 0)
    maj_correct = sum(1 for r in records if r["correct"])
    print(f"majority-vote accuracy (all keys):           {maj_correct/total:.3f}")
    # calibration: accuracy by exact agreement level among COMMITTED keys
    print("\nagreement -> committed keys / correct / cond.acc  (of committed at that agreement)")
    for lvl in range(1, k + 1):
        at = [r for r in records if r["committed"] and r["agreement"] == lvl]
        if at:
            acc = sum(r["correct"] for r in at) / len(at)
            print(f"  exactly {lvl}/{k} agree:  {len(at):2d} keys   {sum(r['correct'] for r in at):2d} correct   cond.acc={acc:.3f}")
    # policy sweep: commit only if majority value has >= tau votes, else abstain
    print("\nselective policy: commit iff a value gets >= tau votes (else abstain)")
    print("  tau   coverage(commit rate)   cond.acc(of commits)")
    for tau in range(1, k + 1):
        commits = [r for r in records if r["committed"] and r["agreement"] >= tau]
        cov = len(commits) / total
        cacc = (sum(r["correct"] for r in commits) / len(commits)) if commits else float("nan")
        print(f"  {tau}/{k}    {cov:.3f}                  {cacc:.3f}   ({len(commits)} commits)")
    print("\nRule baseline on this category: 0.000 accuracy, 0.000 commit rate (equal confidence -> always flags).")


if __name__ == "__main__":
    main()
