#!/usr/bin/env python3
"""Does routing by signal type fix the discrimination failure? (P1 hypothesis H4)

H4, stated before the experiment: a two-stage policy that first classifies the
stored entry as a standing constraint or an ordinary value, then applies the
corresponding rule, achieves higher balanced accuracy over the two critical
types than either the metadata rule or the one-shot LLM merger.

The policy is `ConstraintAwareRouterMerge`. Stage one is one classification
call on the ancestor value, which never sees the metadata or which value a
system would prefer. Stage two is deterministic: a local NLI contradiction
comparison when the entry is standing, the confidence rule when it is ordinary,
abstention when stage one cannot tell. Because stage two is API-free, pairing a
local classifier with it gives a pipeline that runs entirely without an API,
which we evaluate alongside the frontier classifier.

Stage-one accuracy is reported separately, since it is this policy's ceiling
and the bank records each item's true type.

Usage:
  OPENAI_API_KEY=... PYTHONPATH=. python scripts/run_constraint_aware_router.py \
      --classifiers gpt-5.4,llama3.1:8b
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from branchmem.conflict.nli_detector import NLIConflictDetector
from branchmem.llm.base import build_backend
from branchmem.merge.constraint_aware_router import ConstraintAwareRouterMerge

import run_discrimination_experiment as disc

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "final"

# The route each type SHOULD take, fixed by the bank's own definitions.
EXPECTED_ROUTE = {
    "immutable_constraint": "standing",
    "legitimate_override": "ordinary",
    "independent_observation": "ordinary",
    "ambiguous": "ordinary",
}
CRITICAL = ("immutable_constraint", "legitimate_override")


def _cluster_ci(rows: list[dict], fn, n_boot: int = 2000) -> list[float] | None:
    """95% interval for a statistic of the critical pair, resampling TEMPLATE
    FAMILIES rather than items -- the same clustering the preregistered
    analysis uses, so the two-stage rows are comparable to the merger rows."""
    import random
    from collections import defaultdict
    crit = ("immutable_constraint", "legitimate_override")
    by_fam: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["type"] in crit:
            by_fam[r["family"]][r["type"]].append(int(bool(r["correct"])))
    fams = [f for f, d in by_fam.items() if all(c in d for c in crit)]
    if len(fams) < 3:
        return None
    rng = random.Random(disc.SEED)
    boots = []
    for _ in range(n_boot):
        s = [rng.choice(fams) for _ in fams]
        c = [v for f in s for v in by_fam[f]["immutable_constraint"]]
        o = [v for f in s for v in by_fam[f]["legitimate_override"]]
        boots.append(fn(sum(c) / len(c), sum(o) / len(o)))
    boots.sort()
    return [round(boots[int(.025 * n_boot)], 4), round(boots[int(.975 * n_boot)], 4)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--classifiers", default="gpt-5.4")
    ap.add_argument("--arms", default="update_favoured")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="constraint_aware_router.json")
    args = ap.parse_args()

    items = disc.load_bank()
    if args.limit:
        items = items[: args.limit]
    detector = NLIConflictDetector()
    results: dict[str, dict] = {}

    for model in args.classifiers.split(","):
        cfg = {"backend": "openai_compatible", "model": model, "cache_dir": "llm_cache",
               "temperature": 0.0, "max_tokens": 600}
        if model in disc.OPEN_WEIGHT:
            cfg["base_url"] = disc.LOCAL_BASE_URL
        backend = build_backend(cfg)
        results[model] = {}

        for arm in args.arms.split(","):
            strat = ConstraintAwareRouterMerge(backend=backend, detector=detector)
            rows, stage1 = [], []
            for item in items:
                scen = disc.build_scenario(item, arm)
                r = disc.decide(strat, scen)
                route = strat.routes.get(("user", item["predicate"]))
                r["route"] = route
                r["route_correct"] = route == EXPECTED_ROUTE[item["type"]]
                rows.append(r)
                stage1.append(r)

            by_type = {t: disc.summarise([r for r in rows if r["type"] == t])
                       for t in disc.CORRECT}
            acc_c = by_type["immutable_constraint"]["accuracy"]
            acc_o = by_type["legitimate_override"]["accuracy"]
            route_acc = sum(r["route_correct"] for r in stage1) / len(stage1)
            by_type_route = {
                t: round(sum(r["route_correct"] for r in stage1 if r["type"] == t)
                         / max(1, sum(1 for r in stage1 if r["type"] == t)), 4)
                for t in disc.CORRECT}
            route_dist: dict[str, int] = defaultdict(int)
            for r in stage1:
                route_dist[str(r["route"])] += 1

            results[model][arm] = {
                "overall": disc.summarise(rows),
                "by_type": by_type,
                "discrimination_gap": round(acc_o - acc_c, 4),
                "balanced_accuracy_critical": round((acc_o + acc_c) / 2, 4),
                "balanced_accuracy_ci95_family_clustered": _cluster_ci(
                    rows, lambda c, o: (c + o) / 2),
                "discrimination_gap_ci95_family_clustered": _cluster_ci(
                    rows, lambda c, o: o - c),
                "stage1_route_accuracy": round(route_acc, 4),
                "stage1_route_accuracy_critical_types": by_type_route,
                "route_distribution": dict(route_dist),
            }
            r = results[model][arm]
            print(f"{model:16} {arm:16} stage1={route_acc:.3f} "
                  f"constraint={acc_c:.3f} override={acc_o:.3f} "
                  f"D={acc_o-acc_c:+.3f} bal={r['balanced_accuracy_critical']:.3f} "
                  f"bal95={r['balanced_accuracy_ci95_family_clustered']}")

    payload = {
        "note": (
            "PROSPECTIVE hypothesis H4 of ANALYSIS_PLAN_ADDENDUM.md P1. Two-stage policy: "
            "classify the stored entry as standing or ordinary, then apply the matching rule "
            "(local NLI contradiction for standing, confidence rule for ordinary, abstain when "
            "unclear). Stage two is API-free, so a local classifier yields a fully local "
            "pipeline. Stage-one accuracy is reported because it is the policy's ceiling and "
            "the bank records each item's true type. Balanced accuracy over the two critical "
            "types is the comparison H4 was stated about."
        ),
        "bank": str(disc.BANK.relative_to(ROOT)),
        "bank_sha256": hashlib.sha256(disc.BANK.read_bytes()).hexdigest(),
        "expected_route_by_type": EXPECTED_ROUTE,
        "n_items": len(items),
        "results": results,
    }
    (OUT / args.out).write_text(json.dumps(payload, indent=2))
    print("\nWrote", OUT / args.out)


if __name__ == "__main__":
    main()
