#!/usr/bin/env python3
"""Does the discrimination result depend on a contestable labelling norm?

Post-hoc and exploratory: written after the preregistered runs, and in
neither ANALYSIS_PLAN.md nor its addendum.

One objection to the gold label for `immutable_constraint` -- keep the standing
entry even against a confident, user-sourced update -- is that it is a
normative choice, because users often have authority over their own
constraints. If that objection holds, a merger that takes the user's update is
not failing to discriminate; it is applying a different and defensible norm,
and the paper's claim dissolves into a disagreement about labels.

This script tests the objection instead of asserting against it, in three parts.

1. WHERE DOES THE AUTHORITY SIT? Every constraint ancestor is classified by who
   can lift the constraint: an external party (a lease, a building rule, a
   contract, an insurer, a venue, an employer, a union, a library), a physical
   or capacity limit (a shelf width, a voltage, a towing mass, an upload cap),
   or the user's own standing commitment. The objection applies to the third
   class. Classification is by explicit surface patterns, every match
   is recorded, and the whole table is written to CSV so a reader can audit it
   rather than take our word for it.

2. WHAT IF THE OBJECTION IS GRANTED OUTRIGHT? We re-score every model under the
   opposite norm -- a user-sourced, higher-confidence update always supersedes
   the stored entry -- and report what happens. Under that norm the
   field-reading policy is correct by construction and the result inverts. We
   report this plainly: it is the sensitivity of the claim to its convention,
   not a defence of it.

3. WHERE IS THE OBJECTION UNTENABLE? A subset of updates asserts a state that
   is not merely irregular but physically infeasible given the ancestor (a
   24 cm container on a shelf that fits 24 cm, 16 mm chains in a 9 mm arch).
   No reading of user authority makes "accept the update" correct there,
   because the user cannot authorise a physical impossibility. We report the
   primary measure restricted to that subset.

Usage: PYTHONPATH=. python scripts/analyze_label_norms.py
"""
from __future__ import annotations

import csv
import glob
import json
import random
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "final"
BANK = ROOT / "data" / "discrimination_bank.json"
SEED = 7717
CRIT = ("immutable_constraint", "legitimate_override")

# ---------------------------------------------------------------------------
# 1. Authority locus. Each pattern names a party other than the user who would
#    have to agree before the constraint could be lifted. Matching is on the
#    ancestor value only -- never on the update, the label or the metadata.
# ---------------------------------------------------------------------------
EXTERNAL = {
    "lease/landlord": r"\blease\b|\blandlord\b",
    "building/management": r"\bbuilding\b|\bbuilding management\b|\bapartment\b(?!.*\buser's own\b)",
    "contract/policy": r"\bcontract(ual|ually)?\b|\bpolicy\b|\bpolicies\b|\bterms\b",
    "insurer/carrier": r"\binsurer\b|\bcarrier\b|\bcoverage\b|\bclaim\b|\bexcess\b",
    "venue/studio/room": r"\bvenue\b|\bstudio\b|\brehearsal\b|\breading rooms?\b|\bbanquet\b|\bpractice rooms?\b",
    "employer/union": r"\bcompany\b|\bunion\b|\bemployer\b|\bwork laptop\b|\bevent uniform\b|\bsponsor\b",
    "library/archive": r"\blibrary\b|\barchive\b|\binterlibrary\b|\bcourse reserves?\b|\blate fee\b",
    "rental/hire": r"\brental\b|\brented\b|\bhired\b|\bon file\b|\bbilling (address|cycle)\b|\breplacement cards?\b",
    "club/league/membership": r"\bleague\b|\bmembership\b|\bclub\b|\bpool policy\b|\btrack meets?\b|\bgarden policy\b|\bcommunity garden\b",
    "transit/courier/branch": r"\bcourier\b|\bbranch\b|\bstation\b|\bbus\b|\bcarpool\b|\bloading dock\b|\bkiosk\b|\btesting cent(er|re)\b|\bportal\b|\bstore policy\b|\bseason ticket\b|\btoll tag\b|\bservice contract\b",
}

# Physical or capacity limits: a measured quantity the world enforces.
_WORDNUM = (
    r"(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"fifteen hundred|twenty[- ]?five|twenty|thirty|forty|fifty|sixty|"
    r"seventy[- ]?two|seventy[- ]?six|seventy|eighty|ninety|hundred|thousand)"
)
UNIT = (
    r"\b(?:\d+(?:[.,]\d+)?|" + _WORDNUM + r")\s*"
    r"(millimet(?:er|re)s?|mm|centimet(?:er|re)s?|cm|met(?:er|re)s?|inch(?:es)?|feet|foot|"
    r"kilowatts?|kW|volts?|psi|megabytes?|MB|kilograms?|kg|pounds?|lbs?|sheets?|"
    r"minutes?|hours?|degrees?|seats?|people|guests?|students?|riders?|passengers?|"
    r"baskets?|pots?|books?|copies|items?|tickets?|liners?|logos?|waivers?|"
    r"size \d+|inch(?:es)? of hem)"
)
CAPACITY = r"\bmaximum occupancy\b|\bno more than\b|\bup to\b|\bonly fits\b|\bonly accepts\b|\bonly come[s]? up to\b|\bcaps?\b|\blimit(s|ed)?\b|\btowing limit\b|\bhem allowance\b"


def classify_authority(ancestor: str) -> tuple[str, str]:
    """Return (locus, the phrase that decided it)."""
    low = ancestor.lower()
    for name, pat in EXTERNAL.items():
        m = re.search(pat, low)
        if m:
            return f"external:{name}", m.group(0)
    m = re.search(UNIT, low)
    if m and re.search(CAPACITY, low):
        return "physical_or_capacity", m.group(0)
    if m:
        return "physical_or_capacity", m.group(0)
    # No external party named and no measured ceiling found. The ABSENCE of an
    # external marker is not evidence that the constraint is the user's own to
    # waive, so this bucket is reported as unclassified and counted for neither
    # side of the argument.
    return "unclassified", ""


# ---------------------------------------------------------------------------
# 3. Physical infeasibility: the ancestor states a measured ceiling in some
#    unit and the update states a larger quantity in the SAME unit. Both
#    numbers are extracted and compared; nothing is inferred.
# ---------------------------------------------------------------------------
_NUM_UNIT = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*(millimet(?:er|re)s?|mm|centimet(?:er|re)s?|cm|met(?:er|re)s?|"
    r"inch(?:es)?|feet|foot|kilowatts?|kw|volts?|psi|megabytes?|mb|kilograms?|kg|"
    r"pounds?|lbs?|sheets?|seats?|guests?|students?|people|minutes?|hours?|degrees?)",
    re.I,
)
_UNIT_CANON = {
    "millimeter": "mm", "millimetre": "mm", "millimeters": "mm", "millimetres": "mm", "mm": "mm",
    "centimeter": "cm", "centimetre": "cm", "centimeters": "cm", "centimetres": "cm", "cm": "cm",
    "meter": "m", "metre": "m", "meters": "m", "metres": "m",
    "inch": "in", "inches": "in", "feet": "ft", "foot": "ft",
    "kilowatt": "kw", "kilowatts": "kw", "kw": "kw",
    "volt": "v", "volts": "v", "psi": "psi",
    "megabyte": "mb", "megabytes": "mb", "mb": "mb",
    "kilogram": "kg", "kilograms": "kg", "kg": "kg",
    "pound": "lb", "pounds": "lb", "lb": "lb", "lbs": "lb",
    "sheet": "sheets", "sheets": "sheets", "seat": "seats", "seats": "seats",
    "guest": "guests", "guests": "guests", "student": "students", "students": "students",
    "people": "people", "minute": "min", "minutes": "min", "hour": "hr", "hours": "hr",
    "degree": "deg", "degrees": "deg",
}


def _quantities(text: str) -> dict[str, list[float]]:
    out: dict[str, list[float]] = defaultdict(list)
    for num, unit in _NUM_UNIT.findall(text):
        u = _UNIT_CANON.get(unit.lower())
        if u:
            out[u].append(float(num.replace(",", ".")))
    return out


def physically_infeasible(ancestor: str, update: str) -> tuple[bool, str]:
    """True when the update names a larger quantity than the ancestor's ceiling
    in the same unit, and the ancestor is phrased as a ceiling."""
    if not re.search(CAPACITY, ancestor.lower()):
        return False, ""
    a, u = _quantities(ancestor), _quantities(update)
    for unit, avals in a.items():
        if unit in u and max(u[unit]) > max(avals):
            return True, f"{max(u[unit])}{unit} > ceiling {max(avals)}{unit}"
    return False, ""


# ---------------------------------------------------------------------------
# scoring helpers (family-clustered, matching the preregistered analysis)
# ---------------------------------------------------------------------------
def balanced_and_gap(rows: list[dict]) -> dict:
    by_type = {}
    for t in CRIT:
        sub = [r for r in rows if r["type"] == t]
        by_type[t] = round(sum(r["correct"] for r in sub) / len(sub), 4) if sub else None
    ac, ao = by_type["immutable_constraint"], by_type["legitimate_override"]
    if ac is None or ao is None:
        return {"by_type": by_type, "discrimination_gap": None, "balanced_accuracy": None}

    by_fam: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["type"] in CRIT:
            by_fam[r["family"]][r["type"]].append(int(bool(r["correct"])))
    fams = [f for f, d in by_fam.items() if all(t in d for t in CRIT)]
    ci = None
    if len(fams) >= 3:
        def bal(sample):
            c = [v for f in sample for v in by_fam[f]["immutable_constraint"]]
            o = [v for f in sample for v in by_fam[f]["legitimate_override"]]
            return (sum(c) / len(c) + sum(o) / len(o)) / 2
        rng = random.Random(SEED)
        b = sorted(bal([rng.choice(fams) for _ in fams]) for _ in range(2000))
        ci = [round(b[50], 4), round(b[1950], 4)]
    return {
        "by_type": by_type,
        "discrimination_gap": round(ao - ac, 4),
        "balanced_accuracy": round((ac + ao) / 2, 4),
        "balanced_accuracy_ci95_family_clustered": ci,
        "n_families_complete": len(fams),
    }


def main() -> None:
    bank = json.loads(BANK.read_text())
    items = {it["predicate"]: it for f in bank["families"] for it in f["items"]}

    # ---- part 1: authority locus over the 80 constraint items
    rows_csv = []
    locus_counts: dict[str, int] = defaultdict(int)
    infeasible: set[str] = set()
    for it in items.values():
        if it["type"] != "immutable_constraint":
            continue
        locus, phrase = classify_authority(it["ancestor_value"])
        bucket = locus.split(":")[0]
        locus_counts[bucket] += 1
        if locus != bucket:
            locus_counts[locus] += 1
        infeas, why = physically_infeasible(it["ancestor_value"], it["update_value"])
        if infeas:
            infeasible.add(it["predicate"])
        rows_csv.append({
            "predicate": it["predicate"], "domain": it["domain"], "family": it["family"],
            "ancestor_value": it["ancestor_value"], "update_value": it["update_value"],
            "authority_locus": locus, "matched_phrase": phrase,
            "update_physically_infeasible": int(infeas), "infeasibility_evidence": why,
        })
    out_csv = ROOT / "annotation" / "discrimination_authority_locus.csv"
    with out_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_csv[0]))
        w.writeheader()
        w.writerows(rows_csv)

    n_con = sum(1 for i in items.values() if i["type"] == "immutable_constraint")
    print(f"authority locus over {n_con} constraint items")
    for k in sorted(locus_counts):
        if ":" not in k:
            print(f"  {k:24} {locus_counts[k]:3}")
    print(f"  (physically infeasible update: {len(infeasible)})")
    print(f"  wrote {out_csv.relative_to(ROOT)}")

    # ---- parts 2 and 3: re-score every model on the primary arm
    raw: list[dict] = []
    for path in sorted(glob.glob(str(RES / "discrimination_*.json"))):
        if path.endswith("discrimination_summary.json"):
            continue
        payload = json.loads(Path(path).read_text())
        # The glob also catches outputs that are not per-item model runs --
        # the human adjudication's scored summary, for one -- which carry no
        # "raw" rows. Skipping them by shape rather than by name means a later
        # addition does not break this script, which is how it broke before.
        if "raw" not in payload:
            continue
        raw.extend(payload["raw"])
    raw = [r for r in raw if r["arm"] == "update_favoured"]
    if not raw:
        raise SystemExit("no discrimination_*.json raw rows for the primary arm")

    # attach the predicate back out of the scenario id, which is
    # f"{family}_{type}_{predicate}"
    for r in raw:
        r["predicate"] = r["scenario"].split(f"_{r['type']}_", 1)[1]

    models = sorted({r["policy"] for r in raw})
    external_preds = {
        r["predicate"] for r in rows_csv_preds(rows_csv, "external")
    }
    per_model: dict[str, dict] = {}
    for pol in models:
        rows = [r for r in raw if r["policy"] == pol]
        as_labelled = balanced_and_gap(rows)

        # part 2: grant the objection -- the user-sourced update always wins.
        # In the update_favoured arm the update carries source=user/conf=.95, so
        # "correct" becomes "took the update" for BOTH critical types.
        flipped = [dict(r, correct=bool(r["took_update"])) for r in rows]
        under_user_authority = balanced_and_gap(flipped)

        # part 3: restrict to items where the objection cannot apply
        infeas_rows = [r for r in rows
                       if r["type"] == "legitimate_override" or r["predicate"] in infeasible]
        ext_rows = [r for r in rows
                    if r["type"] == "legitimate_override" or r["predicate"] in external_preds]
        per_model[pol] = {
            "as_labelled": as_labelled,
            "under_user_authority_norm": under_user_authority,
            "physically_infeasible_constraints_only": balanced_and_gap(infeas_rows),
            "externally_held_constraints_only": balanced_and_gap(ext_rows),
        }

    print("\nmodel                          as-labelled   user-authority   infeasible-only")
    for pol in models:
        e = per_model[pol]
        f = lambda d: ("  n/a " if d["balanced_accuracy"] is None  # noqa: E731
                       else f"{d['balanced_accuracy']:6.3f}")
        print(f"{pol:30} {f(e['as_labelled'])}        "
              f"{f(e['under_user_authority_norm'])}          "
              f"{f(e['physically_infeasible_constraints_only'])}")

    payload = {
        "note": (
            "POST-HOC, EXPLORATORY (in neither ANALYSIS_PLAN.md nor its "
            "addendum). Tests whether the discrimination claim depends on the labelling "
            "convention that a standing entry survives a confident user-sourced update. "
            "`authority_locus` classifies each constraint ancestor by who could lift the "
            "constraint, using surface patterns recorded alongside every decision in "
            "annotation/discrimination_authority_locus.csv. "
            "`under_user_authority_norm` grants that objection outright by "
            "re-scoring with 'take the user-sourced update' as correct for BOTH critical "
            "types; under that norm a field-reading policy is correct by construction, so "
            "this row is the sensitivity of the claim to its convention, not a defence of "
            "it. `physically_infeasible_constraints_only` restricts the constraint type to "
            "items where the update asserts a larger quantity than the ancestor's stated "
            "ceiling in the same unit, which no account of user authority can license. "
            "This script does not use human labels. The adjudication study has "
            "since been run: see annotation/discrimination_sheets/ for the filled "
            "sheets and results/final/discrimination_human_adjudication.json for its "
            "outcome, which upheld the convention (0 of 25 constraints judged the "
            "user's alone to lift)."
        ),
        "bank_sha256": __import__("hashlib").sha256(BANK.read_bytes()).hexdigest(),
        "n_constraint_items": n_con,
        "authority_locus_counts": dict(locus_counts),
        "n_physically_infeasible_updates": len(infeasible),
        "physically_infeasible_predicates": sorted(infeasible),
        "n_externally_held": len(external_preds),
        "per_model": per_model,
    }
    (RES / "label_norm_sensitivity.json").write_text(json.dumps(payload, indent=2))
    print("\nWrote", RES / "label_norm_sensitivity.json")


def rows_csv_preds(rows_csv: list[dict], prefix: str) -> list[dict]:
    return [r for r in rows_csv if r["authority_locus"].startswith(prefix)]


if __name__ == "__main__":
    main()
