#!/usr/bin/env python3
"""PROBE (not yet a paper result): does test-time compute close the
semantic_resolvable gap? Self-consistency = sample the three-way merge k times
(temperature > 0, each sample cached under a distinct key via a pass-index
nonce) and majority-vote the per-key resolved value. General technique, not
benchmark-specific; also directly tests the paper's preregistered single-call
limitation. Run free on the local open-weight model on a subset first.

Usage:
    OPENAI_API_KEY=ollama python scripts/probe_selfconsistency.py [n_scenarios] [k]
"""
from __future__ import annotations

import sys
from collections import Counter

from branchmem.benchmark.downstream_tasks import generate_downstream_questions
from branchmem.benchmark.semantic_resolvable_generator import generate_semantic_resolvable_scenarios
from branchmem.evaluation.metrics import score_downstream
from branchmem.llm.base import build_backend
from branchmem.memory.schemas import MergeResult, Resolution, ResolvedFact
from branchmem.merge.base import current_facts_by_key, find_collisions
from branchmem.merge.three_way_llm import (
    ThreeWayLLMMerge, _PROMPT_TEMPLATE, _SYSTEM_PROMPT, _parse_response,
)
import json
import os

MODEL = os.environ.get("PROBE_MODEL", "llama3.1:8b")
BASE_URL = os.environ.get("PROBE_BASE_URL") or None  # None -> OpenAI default endpoint


def _collision_prompt(ancestor, branch_a, branch_b, collisions):
    af = current_facts_by_key(ancestor)
    def fj(f): return {"value": f.value, "source": f.source, "confidence": f.confidence}
    aj = json.dumps({f"{k[0]}|{k[1]}": af[k].value for k in collisions if k in af}, indent=2)
    baj = json.dumps({f"{k[0]}|{k[1]}": fj(v[0]) for k, v in collisions.items()}, indent=2)
    bbj = json.dumps({f"{k[0]}|{k[1]}": fj(v[1]) for k, v in collisions.items()}, indent=2)
    return _PROMPT_TEMPLATE.format(ancestor_json=aj, branch_a_json=baj, branch_b_json=bbj)


def self_consistent_merge(backend, ancestor, branch_a, branch_b, k):
    """k samples, majority-vote the committed value per key; tie or majority-flag -> flag."""
    collisions = find_collisions(branch_a, branch_b)
    resolved = []
    # non-conflicting keys resolved deterministically (reuse vanilla logic path)
    base = ThreeWayLLMMerge(backend=backend)
    if not collisions:
        return base.merge(ancestor, branch_a, branch_b)
    prompt = _collision_prompt(ancestor, branch_a, branch_b, collisions)
    # gather k samples
    votes = {f"{k0}|{k1}": [] for (k0, k1) in collisions}
    for i in range(k):
        sysp = _SYSTEM_PROMPT + f"\n\n[Independent reasoning pass {i+1} of {k}. Reason freshly from the facts.]"
        resp = backend.complete(prompt, system=sysp, use_cache=True)
        by_key = _parse_response(resp.text)
        for key_str in votes:
            d = by_key.get(key_str)
            if d and d.get("resolution") != "flagged_unresolved" and d.get("value"):
                votes[key_str].append(str(d["value"]))
            else:
                votes[key_str].append("__FLAG__")
    # build resolved facts by majority vote
    for key, (fa, fb) in collisions.items():
        ks = f"{key[0]}|{key[1]}"
        winner, _ = Counter(votes[ks]).most_common(1)[0]
        if winner == "__FLAG__":
            resolved.append(ResolvedFact(fact=fa, resolution=Resolution.FLAGGED_UNRESOLVED,
                                         source_branch_ids=[branch_a.branch_id, branch_b.branch_id],
                                         justification="majority flagged"))
        else:
            merged = fa.model_copy(update={"value": winner})
            resolved.append(ResolvedFact(fact=merged, resolution=Resolution.MERGED,
                                         source_branch_ids=[branch_a.branch_id, branch_b.branch_id],
                                         justification="self-consistency majority"))
    return MergeResult(strategy_name="self_consistency", resulting_facts=resolved)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    scenarios = generate_semantic_resolvable_scenarios(150, seed=3001)[:n]
    base_backend = build_backend({"backend": "openai_compatible", "model": MODEL, "base_url": BASE_URL,
                                  "cache_dir": "llm_cache", "temperature": 0.0, "max_tokens": 8000})
    sc_backend = build_backend({"backend": "openai_compatible", "model": MODEL, "base_url": BASE_URL,
                                "cache_dir": "llm_cache", "temperature": 0.7, "max_tokens": 8000})
    base_strat = ThreeWayLLMMerge(backend=base_backend)
    base_acc, sc_acc = [], []
    for idx, s in enumerate(scenarios):
        q = generate_downstream_questions(s)
        b = base_strat.merge(s.ancestor, s.branch_a, s.branch_b)
        base_acc.append(score_downstream(q, b).accuracy_for("resolvable"))
        sc = self_consistent_merge(sc_backend, s.ancestor, s.branch_a, s.branch_b, k)
        sc_acc.append(score_downstream(q, sc).accuracy_for("resolvable"))
        if (idx + 1) % 5 == 0:
            def m(v): v = [x for x in v if x == x]; return sum(v)/len(v) if v else float("nan")
            print(f"[{idx+1}/{n}] baseline={m(base_acc):.3f}  self-consistency(k={k})={m(sc_acc):.3f}", flush=True)
    def m(v): v = [x for x in v if x == x]; return sum(v)/len(v) if v else float("nan")
    print(f"\nFINAL n={n} k={k}: baseline three-way={m(base_acc):.3f}  self-consistency={m(sc_acc):.3f}")


if __name__ == "__main__":
    main()
