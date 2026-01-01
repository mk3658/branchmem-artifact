#!/usr/bin/env python3
"""Aggregate the prospective discrimination experiment and judge its hypotheses.

Reads every `discrimination_*.json` shard, pools them, and reports the
cross-model table with intervals that resample TEMPLATE FAMILIES rather than
items. It then evaluates the four hypotheses exactly as they were written in
ANALYSIS_PLAN_ADDENDUM.md P1, including the two that can embarrass us, and
prints which held.

Usage: PYTHONPATH=. python scripts/analyze_discrimination.py
"""
from __future__ import annotations

import glob
import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "final"
SEED = 7717
CRIT = ("immutable_constraint", "legitimate_override")
OPEN_WEIGHT_PROVIDER = {
    "llama3.1:8b": "Meta", "qwen2.5:7b": "Alibaba", "mistral:7b": "Mistral AI",
    "deepseek-r1:8b": "DeepSeek", "qwen3:8b": "Alibaba",
}
AUTHOR_MODEL = "gpt-5.4"


def load_shards() -> tuple[list[dict], dict]:
    rows, meta = [], {}
    for path in sorted(glob.glob(str(RES / "discrimination_*.json"))):
        if path.endswith("discrimination_summary.json"):
            continue
        d = json.loads(Path(path).read_text())
        meta[Path(path).name] = {"bank_sha256": d["bank_sha256"], "n_items": d["n_items"]}
        for r in d["raw"]:
            rows.append(r)
    return rows, meta


def acc(rows: list[dict]) -> float | None:
    return round(sum(r["correct"] for r in rows) / len(rows), 4) if rows else None


def gap_ci(rows: list[dict], n_boot: int = 2000) -> tuple[float | None, list[float] | None]:
    """D and its interval, resampling families so the two types move together."""
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


def main() -> None:
    rows, meta = load_shards()
    if not rows:
        raise SystemExit("no discrimination_*.json shards found")
    hashes = {m["bank_sha256"] for m in meta.values()}
    if len(hashes) != 1:
        raise SystemExit(f"shards disagree on the bank hash: {hashes}")
    print(f"{len(rows)} decisions from {len(meta)} shards, bank {hashes.pop()[:12]}")

    policies = sorted({r["policy"] for r in rows})
    arms = sorted({r["arm"] for r in rows})
    table: dict[str, dict] = {}
    for pol in policies:
        table[pol] = {}
        for arm in arms:
            sub = [r for r in rows if r["policy"] == pol and r["arm"] == arm]
            if not sub:
                continue
            per_type = {t: acc([r for r in sub if r["type"] == t])
                        for t in sorted({r["type"] for r in sub})}
            d, ci = gap_ci(sub)
            ac, ao = per_type.get("immutable_constraint"), per_type.get("legitimate_override")
            table[pol][arm] = {
                "n": len(sub),
                "by_type": per_type,
                "discrimination_gap": d,
                "discrimination_gap_ci95_family_clustered": ci,
                "balanced_accuracy_critical": (
                    round((ac + ao) / 2, 4) if ac is not None and ao is not None else None),
                "coverage": round(sum(not r["abstained"] for r in sub) / len(sub), 4),
                "took_update_rate": round(sum(r["took_update"] for r in sub) / len(sub), 4),
            }

    print(f"\n{'policy':30} {'arm':16} {'constr':>7} {'overr':>7} {'D':>7} "
          f"{'D 95% CI':>18} {'bal':>6}")
    for pol in policies:
        for arm in arms:
            e = table[pol].get(arm)
            if not e or e["discrimination_gap"] is None:
                continue
            bt = e["by_type"]
            print(f"{pol:30} {arm:16} {bt['immutable_constraint']:7.3f} "
                  f"{bt['legitimate_override']:7.3f} {e['discrimination_gap']:+7.3f} "
                  f"{str(e['discrimination_gap_ci95_family_clustered']):>18} "
                  f"{e['balanced_accuracy_critical']:6.3f}")

    # ---- the hypotheses, as written, including the ones that can fail
    primary = "update_favoured"
    llm = [p for p in policies if p.startswith("ThreeWayLLM")]
    h1 = {}
    for p in llm:
        e = table[p].get(primary)
        if e and e["discrimination_gap"] is not None:
            h1[p] = {"D": e["discrimination_gap"],
                     "ci": e["discrimination_gap_ci95_family_clustered"],
                     "holds": e["discrimination_gap"] > 0.5}
    open_models = [p for p in h1 if any(m in p for m in OPEN_WEIGHT_PROVIDER)]
    h2 = {p: h1[p] for p in open_models}
    h3 = {}
    for p in policies:
        e = table[p].get(primary)
        if e and e["balanced_accuracy_critical"] is not None:
            h3[p] = {"balanced": e["balanced_accuracy_critical"],
                     "below_0.75": e["balanced_accuracy_critical"] < 0.75}

    verdicts = {
        "H1_every_model_D_above_0.5": {
            "per_model": h1,
            "holds": bool(h1) and all(v["holds"] for v in h1.values()),
            "exceptions": [p for p, v in h1.items() if not v["holds"]],
        },
        "H2_holds_for_an_open_weight_model_from_another_provider": {
            "per_model": h2,
            "holds": any(v["holds"] for v in h2.values()),
            "providers_tested": sorted({OPEN_WEIGHT_PROVIDER[m] for m in OPEN_WEIGHT_PROVIDER
                                        if any(m in p for p in h2)}),
        },
        "H3_single_rule_balanced_accuracy_below_0.75": {
            "per_policy": h3,
            "holds": bool(h3) and all(v["below_0.75"] for v in h3.values()),
            "exceptions": [p for p, v in h3.items() if not v["below_0.75"]],
        },
    }

    router_path = RES / "constraint_aware_router.json"
    if router_path.exists():
        rd = json.loads(router_path.read_text())["results"]
        h4 = {}
        for model, arms_d in rd.items():
            e = arms_d.get(primary)
            if not e:
                continue
            baseline = max(
                (table[p][primary]["balanced_accuracy_critical"]
                 for p in policies if table[p].get(primary)
                 and table[p][primary]["balanced_accuracy_critical"] is not None),
                default=None)
            h4[model] = {
                "router_balanced": e["balanced_accuracy_critical"],
                "stage1_route_accuracy": e["stage1_route_accuracy"],
                "best_single_rule_balanced": baseline,
                "beats_every_single_rule": (
                    baseline is not None
                    and e["balanced_accuracy_critical"] > baseline),
            }
        verdicts["H4_two_stage_router_beats_single_rule_policies"] = {
            "per_classifier": h4,
            "holds": bool(h4) and all(v["beats_every_single_rule"] for v in h4.values()),
        }

    print("\nhypotheses as preregistered:")
    for k, v in verdicts.items():
        print(f"  {k}: {'HELD' if v.get('holds') else 'FAILED'}")
        if v.get("exceptions"):
            print(f"      exceptions: {v['exceptions']}")

    payload = {
        "note": (
            "Pooled result of the PROSPECTIVE discrimination experiment "
            "(ANALYSIS_PLAN_ADDENDUM.md P1). Every shard was produced against the same frozen, "
            "hashed bank. Intervals resample template families, not items, because items "
            "authored together are not independent. The hypothesis verdicts are computed from "
            "the thresholds written in the preregistration, not chosen afterwards."
        ),
        "shards": meta,
        "n_decisions": len(rows),
        "primary_arm": primary,
        "open_weight_providers": OPEN_WEIGHT_PROVIDER,
        "author_model_is_also_a_solver": AUTHOR_MODEL,
        "table": table,
        "hypothesis_verdicts": verdicts,
    }
    (RES / "discrimination_summary.json").write_text(json.dumps(payload, indent=2))
    print("\nWrote", RES / "discrimination_summary.json")


if __name__ == "__main__":
    main()
