#!/usr/bin/env python3
"""Experiment #4: does frontier wrong-commit-free selective resolution survive
outside the 12 hand-written templates? (post-hoc, exploratory; NOT in
ANALYSIS_PLAN.md.)

The A9/A14 positive --- gpt-5.4 resolves ~47% of `semantic_resolvable`
conflicts with zero wrong commits --- is on 12 hand-authored strict-constraint
templates. A central doubt about the result is that the 1.000 conditional
accuracy is a template artifact. This experiment stress-tests surface realism: we have
gpt-5.4 AUTHOR a larger, domain-diverse bank of strict-constraint semantic
conflicts (same schema as data/semantic_resolvable_templates.json), then run
them through the SAME validated scenario construction
(generate_semantic_resolvable_scenarios) and the SAME frontier merge, and check
whether calibration holds.

Honest scope. The authoring model is also the solving model, so this tests
robustness to realistic surface variety, not to an adversarial or human-authored
distribution; and ground truth is the strict logical incompatibility the author
was told to encode (we structurally validate every item and print them all for
inspection). It is a realism check, not a real-world-data claim (real MAB
content carries no recoverable signal at all, Section~sec:mab-negative).

Two steps, both cached (re-runs cost nothing):
  1. author + validate a template bank -> data/realistic_semantic_templates.json
  2. build scenarios from it, run frontier ThreeWayLLMMerge, score the
     abstention decomposition; compare to the 12-template set.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from branchmem.benchmark.downstream_tasks import generate_downstream_questions
from branchmem.benchmark.semantic_resolvable_generator import generate_semantic_resolvable_scenarios
from branchmem.eval.abstention_metrics import compute_abstention_metrics
from branchmem.evaluation.metrics import score_downstream
from branchmem.llm.base import build_backend
from branchmem.merge.three_way_llm import ThreeWayLLMMerge

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES_PATH = ROOT / "data" / "realistic_semantic_templates.json"
OUT = ROOT / "results" / "final"
MODEL = "gpt-5.4"

DOMAINS = [
    "food allergies and dietary restrictions", "physical accessibility and mobility",
    "medical conditions and medication constraints", "religious or cultural observance",
    "scheduling and hard time constraints", "travel and visa/document constraints",
    "pet ownership and animal allergies", "budget and hard spending limits",
    "professional certification or licensing limits", "language and communication needs",
    "sensory sensitivities (light/sound/motion)", "childcare and family-composition constraints",
]

_GEN_SYSTEM = (
    "You author test items for a memory-reconciliation benchmark. Each item is a "
    "STRICT logical incompatibility: given a stated background constraint, exactly "
    "one of two candidate values is logically INCOMPATIBLE with it and the other is "
    "compatible. Not a mere preference or soft default -- a genuine contradiction a "
    "careful reader would catch. Vary phrasing and entities; make them read like "
    "real user memory notes, not templates."
)

_GEN_PROMPT = """Domain: {domain}

Produce {k} DISTINCT items as strict JSON: {{"items": [
  {{"predicate": "<snake_case key>",
    "ancestor_predicate": "<same snake_case key>",
    "ancestor_value": "<the background constraint, e.g. 'is severely lactose intolerant'>",
    "branch_a_value": "<a value that STRICTLY CONTRADICTS the constraint>",
    "branch_b_value": "<a value COMPATIBLE with the constraint>",
    "correct_branch": "b"}}
]}}
Rules: branch_a_value must be logically impossible given ancestor_value; branch_b_value
must be consistent with it; both must be plausible things a user might say; correct_branch
is always "b". No two items may share a predicate. Output JSON only."""


def _parse_items(text: str) -> list[dict]:
    s = text.strip()
    if s.startswith("```"):
        s = s.split("```", 2)[1].lstrip("json").strip() if "```" in s else s
    start, end = s.find("{"), s.rfind("}")
    obj = json.loads(s[start:end + 1])
    return obj.get("items", [])


_REQUIRED = {"predicate", "ancestor_predicate", "ancestor_value", "branch_a_value", "branch_b_value", "correct_branch"}


def _valid(it: dict) -> bool:
    if not _REQUIRED.issubset(it):
        return False
    if it["correct_branch"] != "b":
        return False
    if it["branch_a_value"].strip().lower() == it["branch_b_value"].strip().lower():
        return False
    return all(isinstance(it[k], str) and it[k].strip() for k in _REQUIRED)


def author_templates(backend, per_domain: int) -> list[dict]:
    items, seen = [], set()
    for domain in DOMAINS:
        resp = backend.complete(_GEN_PROMPT.format(domain=domain, k=per_domain),
                                system=_GEN_SYSTEM, use_cache=True)
        try:
            cand = _parse_items(resp.text)
        except Exception as exc:
            print(f"  [{domain}] parse failed: {exc}")
            continue
        for it in cand:
            if _valid(it) and it["predicate"] not in seen:
                seen.add(it["predicate"])
                items.append({k: it[k] for k in _REQUIRED})
    return items


def score_set(scenarios, backend):
    strat = ThreeWayLLMMerge(backend=backend)
    detail = []
    for s in scenarios:
        qs = generate_downstream_questions(s)
        res = strat.merge(s.ancestor, s.branch_a, s.branch_b, use_cache=True)
        detail.extend(score_downstream(qs, res).detail)
    cats = tuple(sorted({d["category"] for d in detail}))
    m = compute_abstention_metrics(detail, categories_in_scope=cats)
    return {
        "n": len(detail), "accuracy": sum(d["correct"] for d in detail) / len(detail),
        "commit_rate": m.commit_rate, "conditional_on_commit_accuracy": m.conditional_on_commit_accuracy,
        "wrong_commit_rate": m.wrong_commit_rate, "n_committed": m.n_committed,
        "n_wrong_committed": m.n_wrong_committed,
    }


def main() -> None:
    per_domain = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    backend = build_backend({"backend": "openai_compatible", "model": MODEL, "cache_dir": "llm_cache",
                             "temperature": 0.7, "max_tokens": 4000})

    if TEMPLATES_PATH.exists():
        templates = json.loads(TEMPLATES_PATH.read_text())["templates"]
        print(f"Reusing {len(templates)} cached templates at {TEMPLATES_PATH}")
    else:
        print(f"Authoring templates with {MODEL} ({per_domain}/domain x {len(DOMAINS)} domains)...")
        templates = author_templates(backend, per_domain)
        TEMPLATES_PATH.write_text(json.dumps(
            {"note": (f"LLM-authored ({MODEL}) realistic strict-constraint semantic_resolvable "
                      "templates for Experiment #4. Structurally validated; branch_a strictly "
                      "contradicts the ancestor constraint, branch_b is compatible, correct_branch='b'."),
             "templates": templates}, indent=2))
        print(f"Wrote {len(templates)} validated templates -> {TEMPLATES_PATH}")

    print("\n=== authored items (inspect for genuine strict contradictions) ===")
    for t in templates:
        print(f"  [{t['predicate']}] constraint: {t['ancestor_value']}")
        print(f"      A (should contradict): {t['branch_a_value']}")
        print(f"      B (should fit):        {t['branch_b_value']}")

    n = len(templates)
    realistic = generate_semantic_resolvable_scenarios(n, seed=4004, templates_path=TEMPLATES_PATH)
    synth = generate_semantic_resolvable_scenarios(n, seed=4004)  # 12-template set, matched n

    # Solve with the frontier AND a weak model. If the weak model still fails on
    # the SAME frontier-authored realistic items, the frontier's success is a
    # capability effect, not self-recognition of its own generations.
    solver_cfgs = [
        ("gpt-5.4", {"backend": "openai_compatible", "model": "gpt-5.4"}),
        ("gpt-5.4-nano", {"backend": "openai_compatible", "model": "gpt-5.4-nano"}),
    ]
    common = {"cache_dir": "llm_cache", "temperature": 0.0, "max_tokens": 8000}
    res_real, res_synth = {}, {}
    for label, cfg in solver_cfgs:
        solver = build_backend({**cfg, **common})
        res_real[label] = score_set(realistic, solver)
        res_synth[label] = score_set(synth, solver)

    payload = {
        "note": ("Experiment #4 (post-hoc, exploratory; not in ANALYSIS_PLAN.md). Selective "
                 "resolution on LLM-authored realistic strict-constraint semantic conflicts vs the 12 "
                 "hand templates, matched n. Templates authored by gpt-5.4; solved by both gpt-5.4 and "
                 "gpt-5.4-nano to separate capability from self-recognition. Cached."),
        "n_templates": n,
        "realistic_llm_authored": res_real,
        "synthetic_12_templates": res_synth,
    }
    (OUT / "realistic_semantic.json").write_text(json.dumps(payload, indent=2))
    print("\n=== RESULTS (ThreeWayLLMMerge) ===")
    for solver_label in [c[0] for c in solver_cfgs]:
        for setlabel, d in (("realistic", res_real), ("synthetic", res_synth)):
            r = d[solver_label]
            print(f"  {solver_label:14} {setlabel:10} acc={r['accuracy']:.3f}  commit={r['commit_rate']:.3f}  "
                  f"cond_acc={r['conditional_on_commit_accuracy']}  wrong_commit={r['wrong_commit_rate']:.3f}  "
                  f"({r['n_committed']} commits, {r['n_wrong_committed']} wrong)")
    print("\nWrote", OUT / "realistic_semantic.json")


if __name__ == "__main__":
    main()
