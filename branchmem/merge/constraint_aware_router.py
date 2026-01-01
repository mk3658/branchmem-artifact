"""ConstraintAwareRouterMerge: decide WHICH signal to read, then read it.

The paper's operational-baselines appendix argues that provenance-gated merging
should not be "rule versus LLM" but a router over signal *type*, because the
metadata rule and the semantic gate have different exposures: the rule is
perfect where provenance is informative and catastrophic where it is inverted,
while the gate is unmoved by the metadata in either direction. This class is
that router.

The routing question is not "which policy is better" but a property of the
conflict: does the previously recorded value carry STANDING force that a later
update cannot simply override, or is it an ordinary value a correction may
supersede? Those two cases want opposite behaviour from the same metadata:

* standing constraint -> the update must not be allowed to override it, so
  decide on ancestor semantics and ignore the metadata;
* ordinary revisable value -> a newer, better-sourced statement is exactly what
  should win, so apply the provenance rule;
* unclear -> abstain, because the two readings imply opposite actions.

Stage one is a single classification call on the ancestor value. Stage two is
deterministic and API-free: a local NLI contradiction comparison for the
constraint case, the confidence rule for the revisable case. The classifier can
itself be a small local model, in which case the whole policy runs without an
API.

This is a mitigation, not a guarantee. Its ceiling is stage-one accuracy, which
is reported separately wherever this policy is evaluated.
"""

from __future__ import annotations

import re
from typing import Optional

from branchmem.conflict.nli_detector import NLIConflictDetector
from branchmem.memory.schemas import MemoryBranch, MergeResult, Resolution, ResolvedFact
from branchmem.merge.base import MergeStrategy, current_facts_by_key, find_collisions

_SYSTEM = (
    "You classify what kind of thing a stored memory entry is. You do not decide which of two "
    "competing values is correct, and you are not told which one a system prefers. Answer with "
    "one word."
)

_TEMPLATE = """An assistant has this entry stored about the user:

  {ancestor}

A later conversation produced a statement that conflicts with it.

Question: is the stored entry a STANDING requirement, rule or limit, which a later preference
cannot simply override, or an ORDINARY recorded value, which a later correction can legitimately
replace?

Answer with exactly one word: "standing" or "ordinary". If the entry could plausibly be either,
answer "unclear".
"""


class ConstraintAwareRouterMerge(MergeStrategy):
    name = "constraint_aware_router"

    def __init__(
        self,
        backend,
        detector: Optional[NLIConflictDetector] = None,
        margin: float = 0.10,
    ) -> None:
        self.backend = backend
        self.detector = detector or NLIConflictDetector()
        self.margin = margin
        # Routing decisions, so stage-one behaviour can be audited separately
        # from the merge outcome it produces.
        self.routes: dict[tuple[str, str], str] = {}
        self.n_standing = 0
        self.n_ordinary = 0
        self.n_unclear = 0

    def classify(self, ancestor_value: str) -> str:
        text = self.backend.complete(
            _TEMPLATE.format(ancestor=ancestor_value), system=_SYSTEM
        ).text.strip().lower()
        if re.search(r"\bstanding\b", text):
            return "standing"
        if re.search(r"\bordinary\b", text):
            return "ordinary"
        return "unclear"

    def _contradiction(self, premise: str, hypothesis: str) -> float:
        return self.detector._predict([(premise, hypothesis)])[0]["contradiction"]

    def merge(
        self, ancestor: MemoryBranch, branch_a: MemoryBranch, branch_b: MemoryBranch
    ) -> MergeResult:
        anc = current_facts_by_key(ancestor)
        a_facts = current_facts_by_key(branch_a)
        b_facts = current_facts_by_key(branch_b)
        collisions = find_collisions(branch_a, branch_b)

        resolved: list[ResolvedFact] = []
        for key in sorted((set(a_facts) | set(b_facts)) - set(collisions)):
            fact = a_facts.get(key) or b_facts.get(key)
            resolved.append(ResolvedFact(fact=fact, resolution=Resolution.KEPT,
                                         source_branch_ids=[fact.branch_id]))

        for key, (fa, fb) in collisions.items():
            anc_fact = anc.get(key)
            if anc_fact is None:
                # Nothing to classify: no standing entry can be at stake.
                route = "ordinary"
            else:
                route = self.classify(anc_fact.value)
            self.routes[key] = route

            if route == "standing" and anc_fact is not None:
                self.n_standing += 1
                premise = f"{anc_fact.entity} {anc_fact.value}."
                ca = self._contradiction(premise, f"{fa.entity} {fa.value}.")
                cb = self._contradiction(premise, f"{fb.entity} {fb.value}.")
                if abs(ca - cb) < self.margin:
                    resolved.append(ResolvedFact(
                        fact=fa, resolution=Resolution.FLAGGED_UNRESOLVED,
                        source_branch_ids=[branch_a.branch_id, branch_b.branch_id],
                        justification="standing entry, but neither value is clearly compatible"))
                else:
                    win = fa if ca < cb else fb
                    resolved.append(ResolvedFact(
                        fact=win, resolution=Resolution.KEPT,
                        source_branch_ids=[win.branch_id],
                        justification=f"standing entry: kept the less contradicted value "
                                      f"({min(ca, cb):.2f} vs {max(ca, cb):.2f})"))
                continue

            if route == "unclear":
                self.n_unclear += 1
                resolved.append(ResolvedFact(
                    fact=fa, resolution=Resolution.FLAGGED_UNRESOLVED,
                    source_branch_ids=[branch_a.branch_id, branch_b.branch_id],
                    justification="could not tell whether the stored entry has standing force"))
                continue

            self.n_ordinary += 1
            if fa.confidence == fb.confidence:
                resolved.append(ResolvedFact(
                    fact=fa, resolution=Resolution.FLAGGED_UNRESOLVED,
                    source_branch_ids=[branch_a.branch_id, branch_b.branch_id],
                    justification="ordinary entry with no provenance signal"))
            else:
                win = fa if fa.confidence > fb.confidence else fb
                resolved.append(ResolvedFact(
                    fact=win, resolution=Resolution.KEPT,
                    source_branch_ids=[win.branch_id],
                    justification=f"ordinary entry: higher confidence ({win.confidence:.2f})"))

        return MergeResult(strategy_name=self.name, resulting_facts=resolved)
