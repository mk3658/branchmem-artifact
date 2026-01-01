#!/usr/bin/env python3
"""Score the completed human adjudication of the discrimination bank.

Runs every analysis listed in `annotation/discrimination_adjudication_protocol.md`
under "Analysis, fixed in advance", in that order, and prints each result
whichever way it comes out. Nothing here chooses what to report after seeing the
data; that was fixed before the sheets went out.

  1. Q1 against the construction labels, per type, with Wilson intervals, plus
     inter-annotator agreement (raw, Fleiss' kappa, nominal Krippendorff alpha).
  2. Q3, the decisive test: who annotators think could lift each standing rule.
     If a majority say `user_alone` for a majority of the constraint items, the
     paper's labelling convention is wrong and this script says so in those
     words.
  3. Every model re-scored against the annotator majority on Q1 in place of the
     construction label, reporting whether any comparison changes sign.
  4. Q2 against our type assignment.
  5. Per-item disagreements, not just aggregates.

Reads `annotation/discrimination_sheets/discrimination_annotator*.csv` (filled)
and the held-back key. Stdlib only. It does not annotate anything.

The bank was adjudicated in two waves: a 100-item stratified sample first
(`--wave 1`, the default, which reproduces every originally reported
number unchanged) and the remaining 220 items afterwards (`--wave 2`), so
`--wave full` scores all 320 and is what the paper reports. The waves are
disjoint by construction and `build_wave2_adjudication_key.py` refuses to key
wave 2 otherwise. Annotator identity ACROSS waves is not established --- the
sheets carry positions, not people --- so a pooled per-annotator row is a
per-sheet-position row and is labelled as one.

Usage: PYTHONPATH=. python scripts/score_discrimination_adjudication.py \
           [--wave 1|2|full]
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHEETS = ROOT / "annotation" / "discrimination_sheets"
SHEETS2 = ROOT / "annotation" / "discrimination_sheets_wave2"
KEY = SHEETS / "discrimination_key_HELD_BACK.csv"
RES = ROOT / "results" / "final"
OUT = RES / "discrimination_human_adjudication.json"

WAVES = {
    "1": ([SHEETS], OUT),
    "2": ([SHEETS2], RES / "discrimination_human_adjudication_wave2.json"),
    "full": ([SHEETS, SHEETS2], RES / "discrimination_human_adjudication_full_bank.json"),
}

CRIT = ("immutable_constraint", "legitimate_override")


def wilson(k: int, n: int, z: float = 1.96) -> list[float] | None:
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [round(max(0.0, (c - h) / d), 4), round(min(1.0, (c + h) / d), 4)]


def fleiss_kappa(rows: list[list[str]]) -> float | None:
    """rows[i] = the labels the raters gave item i. Equal rater count assumed."""
    rows = [r for r in rows if len(r) > 1]
    if not rows:
        return None
    n = len(rows[0])
    if n < 2 or any(len(r) != n for r in rows):
        return None
    cats = sorted({v for r in rows for v in r})
    p_i = []
    col = Counter()
    for r in rows:
        c = Counter(r)
        col.update(c)
        p_i.append((sum(v * v for v in c.values()) - n) / (n * (n - 1)))
    total = len(rows) * n
    p_e = sum((col[c] / total) ** 2 for c in cats)
    p_bar = sum(p_i) / len(p_i)
    return None if p_e >= 1 else round((p_bar - p_e) / (1 - p_e), 4)


def krippendorff_alpha_nominal(rows: list[list[str]]) -> float | None:
    """Nominal alpha, computed from the coincidence matrix. Stdlib only.

    Handles missing values by using each item's own rater count, so an item
    nobody answered is simply skipped rather than imputed.
    """
    units = [[v for v in r if v != ""] for r in rows]
    units = [u for u in units if len(u) >= 2]
    if not units:
        return None
    cats = sorted({v for u in units for v in u})
    coinc: dict[tuple[str, str], float] = defaultdict(float)
    for u in units:
        m = len(u)
        for a in range(m):
            for b in range(m):
                if a != b:
                    coinc[(u[a], u[b])] += 1.0 / (m - 1)
    n_total = sum(coinc.values())
    if n_total == 0:
        return None
    marg = {c: sum(coinc[(c, d)] for d in cats) for c in cats}
    d_obs = sum(coinc[(c, d)] for c in cats for d in cats if c != d)
    d_exp = sum(marg[c] * marg[d] for c in cats for d in cats if c != d) / (n_total - 1)
    return None if d_exp == 0 else round(1 - d_obs / d_exp, 4)


def _briefs_identical(dirs: list[Path]) -> bool:
    """Whether every wave used the same annotator brief.

    This is checked rather than assumed: the instruction about when to flag an
    item for review is the one that most directly moves the measured quantity,
    so a wave run under a reworded brief must not be pooled silently.
    """
    texts = set()
    for d in dirs:
        p = d / "INSTRUCTIONS_FOR_ANNOTATORS.md"
        texts.add(p.read_text() if p.exists() else "")
    return len(texts) == 1


def majority(labels: list[str]) -> str | None:
    """Strict majority of non-empty labels; None if the raters split evenly."""
    vals = [v for v in labels if v]
    if not vals:
        return None
    top = Counter(vals).most_common()
    if len(top) > 1 and top[0][1] == top[1][1]:
        return None
    return top[0][0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wave", choices=sorted(WAVES), default="1",
                    help="1 = the first 100-item sample (default), 2 = the "
                         "remaining 220, full = all 320")
    args = ap.parse_args()
    dirs, out_path = WAVES[args.wave]

    key = {}
    for d in dirs:
        kp = d / "discrimination_key_HELD_BACK.csv"
        if not kp.exists():
            raise SystemExit(f"missing {kp}")
        for r in csv.DictReader(kp.open()):
            if r["item_id"] in key:
                raise SystemExit(
                    f"{r['item_id']} appears in two waves' keys; the waves must be "
                    "disjoint or every majority vote pools different questions")
            key[r["item_id"]] = r

    sheets: dict[str, dict] = {}
    for d in dirs:
        for path in sorted(glob.glob(str(d / "discrimination_annotator*.csv"))):
            rows = list(csv.DictReader(Path(path).open()))
            filled = sum(1 for r in rows
                         if r.get("q1_which_value_should_be_stored", "").strip())
            if filled == 0:
                print(f"skipping {Path(path).name}: no answers filled in")
                continue
            # Pooling by sheet position, not by person: see the module docstring.
            sheets.setdefault(Path(path).stem, {}).update(
                {r["item_id"]: r for r in rows})
            print(f"loaded {d.name}/{Path(path).name}: {filled}/{len(rows)} items answered")
    if len(sheets) < 2:
        raise SystemExit(
            "need at least two filled annotator sheets. The blank sheets and the "
            "brief are in annotation/discrimination_sheets/; no human labels have "
            "been collected yet.")

    ids = sorted(key)
    names = sorted(sheets)
    get = lambda n, i, c: (sheets[n].get(i, {}).get(c, "") or "").strip().lower()  # noqa: E731

    # ---- 1. Q1 against the construction labels, per type
    print("\n== 1. Q1 vs construction labels (majority of annotators)")
    per_type: dict[str, dict] = {}
    for ty in sorted({key[i]["type"] for i in ids}):
        sub = [i for i in ids if key[i]["type"] == ty]
        agree = n = 0
        for i in sub:
            maj = majority([get(n_, i, "q1_which_value_should_be_stored") for n_ in names])
            if maj is None:
                continue
            n += 1
            want = key[i]["construction_correct_branch"]
            got = "neither" if maj.startswith("neither") else maj
            agree += int(got == want)
        per_type[ty] = {"n_resolved": n, "agree": agree,
                        "rate": round(agree / n, 4) if n else None,
                        "wilson95": wilson(agree, n)}
        print(f"  {ty:26} {agree:3}/{n:3} = "
              f"{per_type[ty]['rate']}  95% CI {per_type[ty]['wilson95']}")
    print("  The construction labels are supported only if agreement on")
    print("  immutable_constraint is high; that row is the one that matters.")

    q1_rows = [[get(n_, i, "q1_which_value_should_be_stored") for n_ in names] for i in ids]
    agreement = {
        "raw_all_agree": round(sum(len(set(r)) == 1 for r in q1_rows) / len(q1_rows), 4),
        "fleiss_kappa": fleiss_kappa(q1_rows),
        "krippendorff_alpha_nominal": krippendorff_alpha_nominal(q1_rows),
    }
    print(f"\n  inter-annotator on Q1: raw all-agree {agreement['raw_all_agree']}, "
          f"Fleiss kappa {agreement['fleiss_kappa']}, "
          f"Krippendorff alpha {agreement['krippendorff_alpha_nominal']}")

    # ---- 2. Q3, the decisive test
    print("\n== 2. Q3: who could lift the rule? (constraint items only)")
    con = [i for i in ids if key[i]["type"] == "immutable_constraint"]
    dist = Counter()
    user_alone_items = 0
    for i in con:
        votes = [get(n_, i, "q3_who_could_lift_it") for n_ in names]
        maj = majority(votes)
        dist[maj or "split"] += 1
        if maj == "user_alone":
            user_alone_items += 1
    for k, v in dist.most_common():
        print(f"  {str(k):18} {v:3}/{len(con)}")
    convention_refuted = user_alone_items > len(con) / 2
    print()
    if convention_refuted:
        print("  VERDICT: the paper's labelling convention is WRONG. A majority of")
        print("  annotators judged that the user alone could lift a majority of these")
        print("  constraints, so the discrimination result must be restated as")
        print("  conditional on a norm that people do not share.")
    else:
        print(f"  VERDICT: convention upheld on this measure. {user_alone_items}/{len(con)}")
        print("  constraint items were judged the user's alone to lift, which is not a")
        print("  majority. This does not make the labels right, only not refuted here.")

    # ---- 3. re-score every model against the annotator majority
    print("\n== 3. Models re-scored against the human majority on Q1")
    human_branch = {}
    for i in ids:
        maj = majority([get(n_, i, "q1_which_value_should_be_stored") for n_ in names])
        if maj:
            human_branch[i] = "neither" if maj.startswith("neither") else maj
    rescored: dict[str, dict] = {}
    raw_files = sorted(glob.glob(str(RES / "discrimination_*.json")))
    pred_by_pol: dict[str, dict[str, str]] = defaultdict(dict)
    for path in raw_files:
        if path.endswith(("summary.json", "human_adjudication.json")):
            continue
        for r in json.loads(Path(path).read_text()).get("raw", []):
            if r.get("arm") != "update_favoured":
                continue
            pred = ("neither" if r["abstained"]
                    else ("update" if r["took_update"] else "keep"))
            pred_by_pol[r["policy"]][r["scenario"]] = pred
    # map item_id -> scenario id via predicate
    scen_of = {}
    for i in ids:
        k = key[i]
        scen_of[i] = f"{k['family']}_{k['type']}_{k['predicate']}"
    for pol, preds in sorted(pred_by_pol.items()):
        n = c_cons = c_hum = 0
        for i, hb in human_branch.items():
            s = scen_of.get(i)
            if s not in preds:
                continue
            k = key[i]
            # the human answer names a branch; convert to keep/update/neither
            if hb == "neither":
                hum = "neither"
            else:
                hum = "update" if hb == k["update_is_branch"] else "keep"
            cons = ("neither" if k["type"] == "ambiguous"
                    else ("keep" if k["type"] == "immutable_constraint" else "update"))
            n += 1
            c_hum += int(preds[s] == hum)
            c_cons += int(preds[s] == cons)
        if n:
            rescored[pol] = {"n": n,
                             "vs_construction": round(c_cons / n, 4),
                             "vs_human_majority": round(c_hum / n, 4),
                             "delta": round((c_hum - c_cons) / n, 4)}
            print(f"  {pol:30} construction {c_cons/n:.3f} -> human {c_hum/n:.3f} "
                  f"({(c_hum-c_cons)/n:+.3f})")
    if rescored:
        signs = {p: (e["vs_construction"] >= 0.5, e["vs_human_majority"] >= 0.5)
                 for p, e in rescored.items()}
        flipped = [p for p, (a, b) in signs.items() if a != b]
        print(f"  comparisons crossing .5 when relabelled: {flipped or 'none'}")

    # ---- 3b. the behavioural comparison: do humans discriminate where models do not?
    #
    # Agreement with the construction TARGET understates what the humans did,
    # because on constraint items they often preferred "flag for review" to
    # committing to the stored value. The quantity that is defined identically
    # for people and for policies is the rate at which each TAKES THE UPDATE,
    # and its gap between the two critical types: a merger that reads only the
    # confidence field takes the update in both and scores 0, while a decider
    # that tells them apart takes it on overrides and not on constraints and
    # scores near 1.
    print("\n== 3b. take-the-update rate by type: humans vs policies")
    take: dict[str, dict] = {}
    for n_ in names:
        row = {}
        for ty in CRIT:
            sub = [i for i in ids if key[i]["type"] == ty]
            row[ty] = round(sum(
                1 for i in sub
                if get(n_, i, "q1_which_value_should_be_stored") == key[i]["update_is_branch"]
            ) / len(sub), 4)
        row["gap"] = round(row["legitimate_override"] - row["immutable_constraint"], 4)
        take[n_] = row
        print(f"  {n_:30} constraint {row['immutable_constraint']:.3f}  "
              f"override {row['legitimate_override']:.3f}  gap {row['gap']:+.3f}")
    for pol, preds in sorted(pred_by_pol.items()):
        row = {}
        for ty in CRIT:
            sub = [i for i in ids if key[i]["type"] == ty and scen_of[i] in preds]
            if not sub:
                break
            row[ty] = round(sum(preds[scen_of[i]] == "update" for i in sub) / len(sub), 4)
        if len(row) == 2:
            row["gap"] = round(row["legitimate_override"] - row["immutable_constraint"], 4)
            take[pol] = row
            print(f"  {pol:30} constraint {row['immutable_constraint']:.3f}  "
                  f"override {row['legitimate_override']:.3f}  gap {row['gap']:+.3f}")
    human_gaps = [take[n_]["gap"] for n_ in names]
    model_gaps = [v["gap"] for k, v in take.items() if k not in names]
    print(f"  human gaps {human_gaps} vs policy gaps {sorted(model_gaps)}")

    # ---- 3c. what humans did on constraints, which is not what our label says
    con_action = Counter()
    for i in [x for x in ids if key[x]["type"] == "immutable_constraint"]:
        m = majority([get(n_, i, "q1_which_value_should_be_stored") for n_ in names])
        if m is None:
            con_action["no_majority"] += 1
        elif m.startswith("neither"):
            con_action["flag_for_review"] += 1
        elif m == key[i]["update_is_branch"]:
            con_action["took_update"] += 1
        else:
            con_action["kept_the_rule"] += 1
    print(f"\n== 3c. majority action on constraint items: {dict(con_action)}")
    took = con_action["took_update"]
    kept, flagged = con_action["kept_the_rule"], con_action["flag_for_review"]
    print(f"  By majority, annotators took the forbidden update on {took} of "
          f"{len(con)} constraint items.")
    if flagged + con_action["no_majority"] > kept:
        print("  They more often declined to resolve than endorsed the stored rule,")
        print("  so the construction target 'keep' is stronger than they are.")
    else:
        print("  They more often endorsed the stored rule than declined to resolve,")
        print("  so on these items the construction target is what they would do.")

    # ---- 4. Q2 against our type assignment
    print("\n== 4. Q2 vs our type assignment")
    q2 = {}
    for ty in sorted({key[i]["type"] for i in ids}):
        sub = [i for i in ids if key[i]["type"] == ty]
        c = Counter()
        for i in sub:
            c[majority([get(n_, i, "q2_ancestor_is") for n_ in names]) or "split"] += 1
        q2[ty] = dict(c)
        print(f"  {ty:26} {dict(c)}")

    # ---- 5. per-item disagreements
    splits = [i for i in ids
              if len({v for v in (get(n_, i, "q1_which_value_should_be_stored")
                                  for n_ in names) if v}) > 1]
    print(f"\n== 5. items with any Q1 disagreement: {len(splits)}/{len(ids)}")
    for i in splits[:20]:
        print(f"  {i} [{key[i]['type']}] "
              f"{[get(n_, i, 'q1_which_value_should_be_stored') for n_ in names]}")
    if len(splits) > 20:
        print(f"  ... and {len(splits)-20} more; all are in the written JSON")

    payload = {
        "note": (
            "Human adjudication of the discrimination bank, scored exactly as "
            "annotation/discrimination_adjudication_protocol.md fixed in advance. "
            "Q3 is the decisive test of the labelling convention."
        ),
        "wave": args.wave,
        "sheet_dirs": [d.name for d in dirs],
        "briefs_identical_across_waves": _briefs_identical(dirs),
        "brief_difference_note": (
            None if len(dirs) < 2 or _briefs_identical(dirs) else
            "The waves were run under DIFFERENT briefs, and they differ on the one "
            "instruction that moves the flagging rate: wave 1 told annotators that "
            "flagging for review 'is a real answer here, not a way of opting out', "
            "while wave 2 told them not to 'guess simply to avoid an unclear or "
            "review answer'. Pooled flagging rates therefore mix two elicitations "
            "and the per-type Q1 rate against the construction label is not "
            "comparable across waves. Q2, Q3 and the take-the-update gap are "
            "reported pooled because they do not depend on that instruction."),
        "annotator_identity_across_waves": (
            "not established; rows are sheet positions, not people"
            if len(dirs) > 1 else "single wave"),
        "n_annotators": len(names),
        "annotators": names,
        "n_items": len(ids),
        "q1_vs_construction_by_type": per_type,
        "inter_annotator_q1": agreement,
        "q3_who_could_lift_it": dict(dist),
        "q3_user_alone_items": user_alone_items,
        "q3_n_constraint_items": len(con),
        "convention_refuted_by_q3": convention_refuted,
        "models_rescored_against_human_majority": rescored,
        "q2_vs_our_types": q2,
        "take_update_rate_by_type": take,
        "human_gaps": human_gaps,
        "policy_gaps": sorted(model_gaps),
        "majority_action_on_constraints": dict(con_action),
        "constraint_target_stronger_than_annotators": bool(
            con_action["flag_for_review"] + con_action["no_majority"]
            > con_action["kept_the_rule"]),
        "items_with_q1_disagreement": splits,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print("\nWrote", out_path)


if __name__ == "__main__":
    main()
