"""ThreeWayLLMMerge: the proposed method. Gives an LLM the common ancestor
explicitly (the "three-way" part, analogous to a three-way file merge) plus
each branch's conflicting current values, and asks it to reconcile each
conflict with a stated justification, or explicitly flag it as unresolvable.

Non-conflicting keys (only one branch touched them, or both branches agree)
are resolved deterministically in code without an LLM call — the LLM is only
invoked for genuine collisions, batched into a single call per merge for cost
control. Every decision is logged with the model's stated reasoning.
"""

from __future__ import annotations

import json
import re

from branchmem.llm.base import LLMBackend
from branchmem.memory.schemas import MemoryBranch, MergeResult, Resolution, ResolvedFact
from branchmem.merge.base import MergeStrategy, current_facts_by_key, find_collisions
from branchmem.utils.logging import get_logger

logger = get_logger(__name__)

_SYSTEM_PROMPT = (
    "You are reconciling two independently-updated branches of an agent's memory "
    "that diverged from a common ancestor while disconnected from each other. For "
    "each conflicting (entity, predicate) key, you are given the common ancestor's "
    "value (if any) and each branch's current value, source, and confidence. "
    "'source' is how that update was recorded: \"user\" means the user explicitly "
    "stated or corrected it; \"inference\" means the agent inferred or guessed it; "
    "\"observation\" means the agent observed it indirectly. Resolve each conflict "
    "using whatever signal actually distinguishes the two updates — e.g. a "
    "user-sourced, higher-confidence update is ordinarily more trustworthy than an "
    "inferred, lower-confidence one; the ancestor value can indicate which branch's "
    "update is more likely a genuine correction vs. a stale duplicate. Do NOT use "
    "timestamps to decide, even if you can infer relative recency — two branches "
    "that were disconnected from each other have no shared clock, so 'later' does "
    "not mean 'more correct' here. If source, confidence, and ancestor context give "
    "no real basis to prefer one branch over the other (e.g. both are equally "
    "plausible, equally-sourced preference changes), explicitly flag it as "
    "unresolvable rather than guessing."
)

_PROMPT_TEMPLATE = """Common ancestor value for each key (null if the key didn't exist yet):
{ancestor_json}

Branch A's current value/source/confidence for each conflicting key:
{branch_a_json}

Branch B's current value/source/confidence for each conflicting key:
{branch_b_json}

For EACH key listed above, decide how to reconcile it. Respond with strict JSON only,
no other text, in this exact shape:

{{"resolutions": [
  {{"entity": "...", "predicate": "...", "resolution": "kept_from_a" | "kept_from_b" | "merged" | "flagged_unresolved",
    "value": "<the reconciled value, or null if flagged_unresolved>", "justification": "<one sentence>"}}
]}}
"""


class ThreeWayLLMMerge(MergeStrategy):
    name = "three_way_llm"

    def __init__(
        self,
        backend: LLMBackend,
        system_prompt: str | None = None,
        prompt_suffix: str | None = None,
    ) -> None:
        self.backend = backend
        # Overriding the system prompt changes the cache key, so an ablation
        # never collides with the locked run's cached calls.
        self.system_prompt = system_prompt or _SYSTEM_PROMPT
        # Appended verbatim to the user template when set. Left None for every
        # locked and preregistered run, which keeps the prompt byte-identical
        # and so keeps those runs cache-served.
        self.prompt_suffix = prompt_suffix
        # Replies that could not be parsed at all. Kept separate from genuine
        # abstentions so a run can report them instead of crediting an
        # unreadable reply as a calibrated decline.
        self.n_calls = 0
        self.n_parse_failures = 0

    def merge(
        self, ancestor: MemoryBranch, branch_a: MemoryBranch, branch_b: MemoryBranch, use_cache: bool = True
    ) -> MergeResult:
        ancestor_facts = current_facts_by_key(ancestor)
        a_facts = current_facts_by_key(branch_a)
        b_facts = current_facts_by_key(branch_b)
        collisions = find_collisions(branch_a, branch_b)

        resolved: list[ResolvedFact] = []

        # Non-conflicting keys: resolved deterministically, no LLM call needed.
        for key in sorted((set(a_facts) | set(b_facts)) - set(collisions)):
            fact_a, fact_b = a_facts.get(key), b_facts.get(key)
            if fact_a is not None and fact_b is not None:  # both branches agree
                resolved.append(
                    ResolvedFact(
                        fact=fact_a,
                        resolution=Resolution.KEPT,
                        source_branch_ids=[branch_a.branch_id, branch_b.branch_id],
                        justification="both branches agree",
                    )
                )
            else:
                fact = fact_a or fact_b
                resolved.append(
                    ResolvedFact(fact=fact, resolution=Resolution.KEPT, source_branch_ids=[fact.branch_id])
                )

        if not collisions:
            return MergeResult(strategy_name=self.name, resulting_facts=resolved)

        def _fact_json(fact) -> dict:
            return {"value": fact.value, "source": fact.source, "confidence": fact.confidence}

        ancestor_json = json.dumps(
            {f"{k[0]}|{k[1]}": ancestor_facts[k].value for k in collisions if k in ancestor_facts}, indent=2
        )
        branch_a_json = json.dumps({f"{k[0]}|{k[1]}": _fact_json(v[0]) for k, v in collisions.items()}, indent=2)
        branch_b_json = json.dumps({f"{k[0]}|{k[1]}": _fact_json(v[1]) for k, v in collisions.items()}, indent=2)
        prompt = _PROMPT_TEMPLATE.format(
            ancestor_json=ancestor_json, branch_a_json=branch_a_json, branch_b_json=branch_b_json
        )
        if self.prompt_suffix:
            prompt = prompt + self.prompt_suffix
        response = self.backend.complete(prompt, system=self.system_prompt, use_cache=use_cache)
        resolutions_by_key = _parse_response(response.text)
        self.n_calls += 1
        if not resolutions_by_key:
            self.n_parse_failures += 1
            logger.warning(
                "unparseable reply from %s (%d chars); the affected keys are "
                "flagged unresolved and counted as parse failures, not abstentions",
                getattr(self.backend, "model", "?"), len(response.text or ""),
            )

        unresolved = []
        for key, (fact_a, fact_b) in collisions.items():
            key_str = f"{key[0]}|{key[1]}"
            decision = resolutions_by_key.get(key_str)
            if decision is None:
                logger.warning("no LLM resolution for key %s; flagging unresolved", key_str)
                decision = {"resolution": "flagged_unresolved", "value": None, "justification": "unparsed LLM response"}

            res_type = decision.get("resolution", "flagged_unresolved")
            justification = decision.get("justification", "")
            if res_type == "flagged_unresolved" or not decision.get("value"):
                resolved.append(
                    ResolvedFact(
                        fact=fact_a,
                        resolution=Resolution.FLAGGED_UNRESOLVED,
                        source_branch_ids=[branch_a.branch_id, branch_b.branch_id],
                        justification=justification or "flagged unresolvable by LLM",
                    )
                )
                from branchmem.memory.schemas import ConflictPair

                unresolved.append(
                    ConflictPair(fact_a_id=fact_a.fact_id, fact_b_id=fact_b.fact_id, is_conflict=True)
                )
                continue

            value = decision["value"]
            resolution = (
                Resolution.KEPT
                if (res_type == "kept_from_a" and value == fact_a.value)
                or (res_type == "kept_from_b" and value == fact_b.value)
                else Resolution.MERGED
            )
            source_fact = fact_a if res_type == "kept_from_a" else fact_b
            merged_fact = source_fact.model_copy(update={"value": value})
            resolved.append(
                ResolvedFact(
                    fact=merged_fact,
                    resolution=resolution,
                    source_branch_ids=[branch_a.branch_id, branch_b.branch_id],
                    justification=justification,
                )
            )

        return MergeResult(strategy_name=self.name, resulting_facts=resolved, unresolved_conflicts=unresolved)


_THINK_BLOCK = re.compile(r"<think\b[^>]*>.*?</think\s*>", re.S | re.I)
_THINK_OPEN = re.compile(r"<think\b[^>]*>.*\Z", re.S | re.I)


def _strip_reasoning(text: str) -> str:
    """Remove a reasoning model's visible scratchpad before JSON extraction.

    Reasoning models such as DeepSeek-R1 emit a `<think>...</think>` block
    ahead of the answer. That prose routinely contains braces, so the
    first-brace-to-last-brace heuristic below would slice across it and fail to
    parse. A truncated reply can also leave `<think>` unclosed, so an unclosed
    block running to the end of the string is dropped too. This is a no-op for
    models that emit no such block, which keeps every cached result identical.
    """
    text = _THINK_BLOCK.sub("", text)
    return _THINK_OPEN.sub("", text)


def _balanced_json_objects(text: str):
    """Yield every balanced top-level {...} span, outermost first."""
    depth = start = 0
    in_str = esc = False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                yield text[start : i + 1]
            elif depth < 0:
                depth = 0


def _parse_response(text: str) -> dict[str, dict]:
    """Extract the resolution objects, or {} if the reply cannot be parsed.

    An empty return means PARSE FAILURE, which callers must not confuse with a
    model that deliberately declined: `ThreeWayLLMMerge` counts these in
    `n_parse_failures` so a run can report them rather than scoring an
    unreadable reply as a calibrated abstention.
    """
    text = _strip_reasoning(text)
    candidates = []
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])
    # Fall back to properly balanced spans, preferring ones that look like the
    # requested schema. Needed when prose around the JSON carries stray braces.
    candidates.extend(
        sorted(_balanced_json_objects(text), key=lambda s: ("resolutions" not in s, -len(s)))
    )
    for cand in candidates:
        try:
            parsed = json.loads(cand)
        except json.JSONDecodeError:
            continue
        if not isinstance(parsed, dict) or "resolutions" not in parsed:
            continue
        out = {}
        for item in parsed.get("resolutions", []):
            try:
                key = f"{item['entity']}|{item['predicate']}"
                out[key] = item
            except (KeyError, TypeError):
                continue
        if out:
            return out
    return {}
