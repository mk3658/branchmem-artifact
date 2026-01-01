#!/usr/bin/env python3
"""Author the prospective constraint-versus-override bank (see A/P1 in
ANALYSIS_PLAN_ADDENDUM.md).

This script only AUTHORS items. It does not solve them. Its output is written
once, hashed, and the hash recorded, so that the bank demonstrably predates
every solver run reported against it.

Each call asks for one domain-matched QUADRUPLE, one item per conflict type, so
that the four types are balanced across domains and a template family is a
natural clustering unit:

  immutable_constraint   ancestor is a standing constraint; the update would
                         violate it; correct action is to REJECT the update
  legitimate_override    ancestor is an ordinary stale value; the update is a
                         genuine correction; correct action is to ACCEPT it
  independent_observation  two observations of a changeable fact, where the
                         content itself (not the metadata) says which is current
  ambiguous              the exposed evidence does not determine an answer

The critical pair is the first two: in the released bank both give the strong
provenance to the update, so a policy that reads only source/confidence behaves
identically in the two and must score 1 and 0.

Domains are drawn from a fixed list chosen to be disjoint from every bank used
in the paper so far (diet/allergy, mobility, medication, religion, scheduling,
travel documents, pets, budget, licensing, accessibility, sensory, childcare).

Usage: OPENAI_API_KEY=... PYTHONPATH=. python scripts/author_discrimination_bank.py
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from branchmem.llm.base import build_backend

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "discrimination_bank.json"
AUTHOR_MODEL = "gpt-5.4"

TYPES = ("immutable_constraint", "legitimate_override",
         "independent_observation", "ambiguous")

# 80 domains, none of them used by an existing bank in this paper.
DOMAINS = [
    "parking permits", "fuel and charging", "road tolls", "vehicle servicing",
    "tyre and winter equipment", "home heating", "kitchen appliances",
    "door locks and keys", "rubbish and recycling collection", "water supply",
    "broadband and routers", "work laptops", "software licences",
    "building access cards", "printer and stationery supply",
    "gym membership", "running shoes and kit", "swimming pool access",
    "cycling equipment", "team sport fixtures", "houseplants",
    "garden watering", "lawn equipment", "compost and waste",
    "seed and planting seasons", "musical instruments", "practice rooms",
    "concert tickets", "sheet music", "audio equipment", "cookware",
    "food storage containers", "kitchen knives", "coffee equipment",
    "grocery delivery slots", "postal addresses", "parcel lockers",
    "package signatures", "shipping insurance", "customs declarations",
    "bank card replacement", "standing orders", "statement delivery",
    "branch appointments", "currency ordering", "language classes",
    "study materials", "exam bookings", "tutoring sessions",
    "certificate collection", "train season tickets", "bus routes",
    "bike storage at stations", "carpool arrangements", "taxi accounts",
    "home insurance cover", "contents valuation", "claim excess",
    "policy renewal dates", "assessor visits", "clothing sizes",
    "shoe fitting", "uniform requirements", "laundry handling",
    "tailoring alterations", "event seating", "venue entry rules",
    "catering numbers", "equipment hire", "stage call times",
    "camera equipment", "photo storage", "print sizes", "lens filters",
    "tripod use", "library borrowing", "reading room booking",
    "interlibrary loans", "overdue handling", "archive access",
]

_SYSTEM = (
    "You author evaluation items for a benchmark on reconciling conflicting entries in an AI "
    "assistant's long-term memory. You write realistic, mundane, non-sensitive items about "
    "everyday logistics. You never write medical, dietary, allergy, religious, financial-hardship "
    "or disability content. You output strict JSON only."
)

_TEMPLATE = """Domain: {domain}

Write FOUR memory-conflict items in this domain, one of each type. In every item an
assistant has an older recorded value for one key, and a newer incoming update that
conflicts with it. Use the SAME entity "user" and a DIFFERENT snake_case predicate per item.

1. "immutable_constraint": `ancestor_value` states a STANDING RULE or hard limit that cannot
   simply be overridden by a later preference (a rule imposed by someone else, a physical
   limit, a contractual or safety requirement). `update_value` is an assertion that would
   VIOLATE that rule. `keep_value` is a statement consistent with the rule. The correct
   resolution is `keep_value`, because the rule still holds.

2. "legitimate_override": `ancestor_value` states an ORDINARY, revisable recorded value with
   no standing-rule force. `update_value` is a genuine, explicit correction or change of that
   value. `keep_value` restates the now-stale ancestor value. The correct resolution is
   `update_value`, because the newer statement supersedes a stale one.

3. "independent_observation": two observations of a fact that changes over time, where the
   CONTENT of the values itself says which is current (e.g. one says "as of last season", the
   other "as of this week"). The correct resolution is the one the content marks as current;
   put that text in `update_value` and the superseded text in `keep_value`.

4. "ambiguous": two equally well-warranted, mutually exclusive values with nothing in the
   content to choose between them. Correct behaviour is to decline to answer. For THIS type
   only, `ancestor_value` must be a NEUTRAL background fact in the domain that bears on
   neither value and does not restate either of them, so that the ancestor gives no evidence.

Rules for every item: `ancestor_value`, `update_value` and `keep_value` must each read as a
natural standalone statement of 4 to 18 words; they must be mutually distinguishable in
meaning, not rewordings; nothing in any value may mention sources, who said it, confidence,
reliability, timestamps in clock form, or how trustworthy it is.

Respond with strict JSON only:
{{"items": [
  {{"type": "immutable_constraint", "predicate": "<snake_case>", "ancestor_value": "...",
    "update_value": "...", "keep_value": "...", "why_correct": "<one clause>"}},
  {{"type": "legitimate_override", ...}},
  {{"type": "independent_observation", ...}},
  {{"type": "ambiguous", ...}}
]}}
"""

_CONSTRAINT_MARKER = re.compile(
    r"\bonly\b|\bmust\b|\bcannot\b|\bnot permitted\b|\bnot allowed\b|\bno \w+ (are|is) allowed\b|"
    r"\brequire[sd]?\b|\bmaximum\b|\blimit\b|\bprohibit|\bforbid|\brule\b|\bpolicy\b|"
    r"\bcontract|\bban(s|ned)?\b|\bmandat|\bobliged\b|\bcapacity\b|\bdeadline\b|\bno more than\b",
    re.I,
)


def _parse(text: str) -> list[dict] | None:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    items = obj.get("items")
    return items if isinstance(items, list) else None


def _valid(items: list[dict], seen_predicates: set[str]) -> tuple[bool, str]:
    if len(items) != 4:
        return False, f"expected 4 items, got {len(items)}"
    if [i.get("type") for i in items] != list(TYPES):
        return False, f"types out of order: {[i.get('type') for i in items]}"
    for it in items:
        for field in ("predicate", "ancestor_value", "update_value", "keep_value"):
            v = str(it.get(field, "")).strip()
            if not v:
                return False, f"empty {field}"
            if field != "predicate" and not (3 <= len(v.split()) <= 24):
                return False, f"{field} length: {v!r}"
        if not re.fullmatch(r"[a-z][a-z0-9_]{2,48}", str(it["predicate"])):
            return False, f"bad predicate {it['predicate']!r}"
        if it["predicate"] in seen_predicates:
            return False, f"duplicate predicate {it['predicate']!r}"
        # The two BRANCH values must differ; `keep_value` is allowed to restate
        # `ancestor_value`, which is exactly what the override type asks for.
        if str(it["update_value"]).strip().lower() == str(it["keep_value"]).strip().lower():
            return False, "the two branch values are identical"
        blob = " ".join(str(it[f]) for f in ("ancestor_value", "update_value", "keep_value"))
        if re.search(r"\bsource\b|\bconfidence\b|\breliab|\btrustworth|\binferred\b|"
                     r"\buser said\b|\bam\b|\bpm\b", blob, re.I):
            return False, "value leaks provenance or clock time"
    # An ambiguous item whose ancestor restates one branch is not ambiguous: the
    # ancestor would be evidence for that branch.
    amb = items[3]
    anc_tokens = set(re.findall(r"[a-z0-9]+", str(amb["ancestor_value"]).lower()))
    for side in ("update_value", "keep_value"):
        side_tokens = set(re.findall(r"[a-z0-9]+", str(amb[side]).lower()))
        if str(amb["ancestor_value"]).strip().lower() == str(amb[side]).strip().lower():
            return False, "ambiguous ancestor restates a branch value"
        overlap = len(anc_tokens & side_tokens) / max(1, len(side_tokens))
        if overlap > 0.8:
            return False, f"ambiguous ancestor nearly restates {side} (overlap {overlap:.2f})"

    # The two critical types must differ in the ancestor's modal force, which is
    # the whole point of the comparison.
    if not _CONSTRAINT_MARKER.search(items[0]["ancestor_value"]):
        return False, f"constraint ancestor has no standing-rule marker: {items[0]['ancestor_value']!r}"
    if _CONSTRAINT_MARKER.search(items[1]["ancestor_value"]):
        return False, f"override ancestor reads as a standing rule: {items[1]['ancestor_value']!r}"
    return True, ""


def main() -> None:
    backend = build_backend({"backend": "openai_compatible", "model": AUTHOR_MODEL,
                             "cache_dir": "llm_cache", "temperature": 0.0,
                             "max_tokens": 4000})
    families: list[dict] = []
    seen: set[str] = set()
    rejected: list[dict] = []

    for idx, domain in enumerate(DOMAINS):
        for attempt in range(3):
            prompt = _TEMPLATE.format(domain=domain)
            if attempt:
                prompt += (f"\nAttempt {attempt+1}: the previous response was rejected "
                           f"({rejected[-1]['reason']}). Fix exactly that and keep everything else.")
            items = _parse(backend.complete(prompt, system=_SYSTEM).text)
            if items is None:
                rejected.append({"domain": domain, "reason": "unparseable"})
                continue
            ok, reason = _valid(items, seen)
            if not ok:
                rejected.append({"domain": domain, "reason": reason})
                continue
            for it in items:
                seen.add(it["predicate"])
                it["domain"] = domain
                it["family"] = f"fam_{idx:03d}"
            families.append({"family": f"fam_{idx:03d}", "domain": domain, "items": items})
            print(f"[{idx+1}/{len(DOMAINS)}] {domain}: ok ({len(families)*4} items)")
            break
        else:
            print(f"[{idx+1}/{len(DOMAINS)}] {domain}: FAILED after 3 attempts")

    flat = [it for f in families for it in f["items"]]
    payload = {
        "note": (
            "PROSPECTIVE bank for the constraint-versus-override discrimination experiment "
            "(ANALYSIS_PLAN_ADDENDUM.md P1). Authored by gpt-5.4 under structural validation, "
            "then frozen: no item was added, removed or relabelled after any solver ran. The "
            "author is also one of the solvers, which is disclosed; the open-weight solvers are "
            "from different providers than the author. Domains are disjoint from every other "
            "bank in this paper. Every item was structurally validated: four types per domain, "
            "unique predicates, distinct natural-language values of 3-24 words, no value "
            "mentioning source, confidence, reliability or clock time, and the two critical "
            "types verified to differ in the ancestor's modal force."
        ),
        "author_model": AUTHOR_MODEL,
        "n_families": len(families),
        "n_items": len(flat),
        "types": list(TYPES),
        "n_rejected_drafts": len(rejected),
        "rejection_reasons": rejected,
        "families": families,
    }
    OUT.write_text(json.dumps(payload, indent=2))
    digest = hashlib.sha256(OUT.read_bytes()).hexdigest()
    (OUT.parent / "discrimination_bank.sha256").write_text(
        f"{digest}  {OUT.name}\n")
    print(f"\nwrote {len(flat)} items in {len(families)} families to {OUT}")
    print(f"rejected drafts: {len(rejected)}")
    print(f"sha256: {digest}")


if __name__ == "__main__":
    main()
