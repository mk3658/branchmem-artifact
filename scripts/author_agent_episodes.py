#!/usr/bin/env python3
"""Author the offline two-agent divergence episodes (P2 in ANALYSIS_PLAN_ADDENDUM.md).

This script authors ENVIRONMENTS only: a shared starting memory and two
independent dialogues. It does not decide what either agent stores --- that is
the agent's own job at run time, and it is the whole point of the experiment.
The output is frozen and hashed before any agent runs against it.

Each episode has:

  * `ancestor`: three facts the assistant already holds, one of which is a
    STANDING CONSTRAINT imposed from outside (a building rule, a contract, a
    physical limit) and two of which are ordinary revisable values.
  * `dialogue_a`: three user turns. One states a DEFINITE change to the first
    ordinary key. One is neutral chatter. One touches the constraint key in a
    way that respects it.
  * `dialogue_b`: three user turns, independent of A. One SPECULATES about the
    same ordinary key without deciding ("I might switch to ..."). One makes a
    confident request that would VIOLATE the constraint. One is neutral.

That gives two conflict opportunities per episode with known environment truth:
on the ordinary key a definite statement should beat a speculation, and on the
constraint key the constraint should survive a confident request. We know the
truth because we wrote the environment; we do not know which conflicts will
actually arise, because that depends on what each agent chooses to record.

Usage: OPENAI_API_KEY=... PYTHONPATH=. python scripts/author_agent_episodes.py
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from branchmem.llm.base import build_backend

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "agent_episodes.json"
AUTHOR_MODEL = "gpt-5.4"
N_EPISODES = 150

SETTINGS = [
    "a shared flat", "a university hall of residence", "a co-working space",
    "a community allotment", "a sailing club", "an amateur orchestra",
    "a local library", "a climbing gym", "a car club", "a camera club",
    "a village hall committee", "a tool-lending library", "a choir",
    "a makerspace", "a cycling club", "a swimming club", "a book group",
    "a model railway club", "a pottery studio", "a dance school",
    "a language exchange", "a chess club", "a scout group", "a food co-op",
    "a repair cafe", "a darkroom collective", "a radio club",
    "a rowing club", "a bridge club", "a gardening society",
]

_SYSTEM = (
    "You author environments for an evaluation of how an AI assistant reconciles memory that "
    "diverged across two offline sessions. You write mundane, non-sensitive, everyday logistics. "
    "No medical, dietary, allergy, religious, financial-hardship or disability content. "
    "You output strict JSON only."
)

_TEMPLATE = """Setting: {setting}

Write ONE episode. The assistant already holds three facts about "user". Then two separate
conversations happen, on different devices, with no shared clock, and neither conversation
knows about the other.

Facts already held:
  * one STANDING CONSTRAINT imposed from outside the user's control (a rule of the
    organisation, a contract, a physical limit). Give it predicate `constraint_predicate`,
    value `constraint_value`, and a `constraint_compatible_value`: a statement of what the
    user does that respects the rule.
  * one ORDINARY revisable value, predicate `ordinary_predicate`, value `ordinary_value`.
  * one BACKGROUND fact, predicate `background_predicate`, value `background_value`, which
    neither conversation touches.

Conversation A, three user turns in `dialogue_a`:
  * one turn that DEFINITELY changes the ordinary value; put the resulting new value in
    `ordinary_new_value`;
  * one turn that mentions the constraint area while respecting the rule;
  * one neutral turn about the setting that implies no memory change.

Conversation B, three user turns in `dialogue_b`, independent of A:
  * one turn that only SPECULATES about the same ordinary key without deciding; put the
    speculated value in `ordinary_speculated_value`;
  * one turn that confidently asks for something the standing constraint does NOT allow; put
    the value that request would imply in `constraint_violating_value`;
  * one neutral turn.

Every turn must be something a person would plausibly say, 8 to 30 words, first person, and
must not mention memory, sources, confidence, reliability or how trustworthy anything is.
All predicates are snake_case and distinct. All values are short statements of 3 to 16 words.

Respond with strict JSON only:
{{"constraint_predicate": "...", "constraint_value": "...", "constraint_compatible_value": "...",
  "constraint_violating_value": "...",
  "ordinary_predicate": "...", "ordinary_value": "...", "ordinary_new_value": "...",
  "ordinary_speculated_value": "...",
  "background_predicate": "...", "background_value": "...",
  "dialogue_a": ["...", "...", "..."], "dialogue_b": ["...", "...", "..."]}}
"""

_FIELDS = ("constraint_predicate", "constraint_value", "constraint_compatible_value",
           "constraint_violating_value", "ordinary_predicate", "ordinary_value",
           "ordinary_new_value", "ordinary_speculated_value",
           "background_predicate", "background_value")
_LEAK = re.compile(r"\bsource\b|\bconfidence\b|\breliab|\btrustworth|\bmemory\b|\binferred\b", re.I)


def _parse(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def _valid(ep: dict, seen: set[str]) -> tuple[bool, str]:
    for f in _FIELDS:
        if not str(ep.get(f, "")).strip():
            return False, f"empty {f}"
    for f in ("constraint_predicate", "ordinary_predicate", "background_predicate"):
        if not re.fullmatch(r"[a-z][a-z0-9_]{2,48}", str(ep[f])):
            return False, f"bad predicate {ep[f]!r}"
    preds = {ep["constraint_predicate"], ep["ordinary_predicate"], ep["background_predicate"]}
    if len(preds) != 3:
        return False, "predicates not distinct"
    if preds & seen:
        return False, f"predicate reused: {sorted(preds & seen)}"
    for f in ("constraint_value", "constraint_compatible_value", "constraint_violating_value",
              "ordinary_value", "ordinary_new_value", "ordinary_speculated_value",
              "background_value"):
        n = len(str(ep[f]).split())
        if not 2 <= n <= 20:
            return False, f"{f} length {n}"
    if str(ep["constraint_compatible_value"]).lower() == str(ep["constraint_violating_value"]).lower():
        return False, "compatible and violating values identical"
    ordv = {str(ep[f]).lower() for f in
            ("ordinary_value", "ordinary_new_value", "ordinary_speculated_value")}
    if len(ordv) != 3:
        return False, "ordinary values not distinct"
    for side in ("dialogue_a", "dialogue_b"):
        turns = ep.get(side)
        if not isinstance(turns, list) or len(turns) != 3:
            return False, f"{side} must have 3 turns"
        for t in turns:
            n = len(str(t).split())
            if not 6 <= n <= 40:
                return False, f"{side} turn length {n}"
            if _LEAK.search(str(t)):
                return False, f"{side} turn leaks provenance vocabulary"
    return True, ""


def main() -> None:
    backend = build_backend({"backend": "openai_compatible", "model": AUTHOR_MODEL,
                             "cache_dir": "llm_cache", "temperature": 0.0,
                             "max_tokens": 4000})
    episodes: list[dict] = []
    seen: set[str] = set()
    rejected: list[dict] = []
    variants_per_setting = -(-N_EPISODES // len(SETTINGS))

    for s_i, setting in enumerate(SETTINGS):
        for v in range(variants_per_setting):
            if len(episodes) >= N_EPISODES:
                break
            tag = f"ep_{len(episodes):03d}"
            for attempt in range(3):
                prompt = _TEMPLATE.format(setting=setting)
                if v or attempt:
                    prompt += (f"\nThis is variant {v+1} for this setting; make the specific "
                               f"objects, predicates and values different from any obvious "
                               f"first choice.")
                if attempt:
                    prompt += f"\nThe previous attempt was rejected ({rejected[-1]['reason']}). Fix that."
                ep = _parse(backend.complete(prompt, system=_SYSTEM).text)
                if ep is None:
                    rejected.append({"setting": setting, "reason": "unparseable"})
                    continue
                ok, reason = _valid(ep, seen)
                if not ok:
                    rejected.append({"setting": setting, "reason": reason})
                    continue
                ep["episode_id"] = tag
                ep["setting"] = setting
                seen.update({ep["constraint_predicate"], ep["ordinary_predicate"],
                             ep["background_predicate"]})
                episodes.append(ep)
                print(f"[{len(episodes)}/{N_EPISODES}] {setting} v{v+1}: ok")
                break
            else:
                print(f"{setting} v{v+1}: FAILED after 3 attempts")

    payload = {
        "note": (
            "PROSPECTIVE environments for the offline two-agent divergence experiment "
            "(ANALYSIS_PLAN_ADDENDUM.md P2). These files contain ENVIRONMENTS ONLY: a shared "
            "starting memory and two independent dialogues. What each agent stores, with what "
            "source and confidence, is decided by the agent at run time and is not scripted "
            "here. Environment truth is therefore known to us (we wrote the dialogues) while "
            "the branch contents are not. Authored by gpt-5.4 under structural validation and "
            "frozen before any agent ran."
        ),
        "author_model": AUTHOR_MODEL,
        "n_episodes": len(episodes),
        "n_rejected_drafts": len(rejected),
        "rejection_reasons": rejected[:50],
        "episodes": episodes,
    }
    OUT.write_text(json.dumps(payload, indent=2))
    digest = hashlib.sha256(OUT.read_bytes()).hexdigest()
    (OUT.parent / "agent_episodes.sha256").write_text(f"{digest}  {OUT.name}\n")
    print(f"\nwrote {len(episodes)} episodes to {OUT}\nsha256: {digest}")


if __name__ == "__main__":
    main()
