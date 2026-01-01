"""HybridMerge: a deployable cascade that routes each conflict by whether a
usable metadata signal is present.

Motivation. Two results in this paper point in opposite directions. Where a
provenance signal exists (differing source/confidence), the deterministic
``ConfidenceRuleMerge`` matches or beats the LLM -- an ``if``-statement is
enough. Where no such signal exists (equal source, equal confidence; the
``semantic_resolvable`` regime), the rule can only abstain, yet a
capable LLM is a *wrong-commit-free selective resolver*. Neither strategy
dominates on its own.

HybridMerge composes them without an oracle: run ``ConfidenceRuleMerge``
first; wherever it *abstains* (a tied-confidence collision it cannot decide),
escalate that key -- and only that key -- to ``ThreeWayLLMMerge``. The
escalation trigger ("the rule abstained") is directly observable at merge
time, so this is a policy one could actually deploy, not a post-hoc oracle.

By construction the cascade inherits the rule's decision wherever the rule
commits, and the LLM's decision (which may itself be a calibrated abstention)
wherever the rule does not. Whether that escalation *helps* on every category
-- in particular whether the LLM wrongly commits on genuinely ``ambiguous``
conflicts that the rule correctly flagged -- is an empirical question the
accompanying experiment measures rather than assumes.
"""

from __future__ import annotations

from branchmem.llm.base import LLMBackend
from branchmem.memory.schemas import MemoryBranch, MergeResult, Resolution
from branchmem.merge.base import MergeStrategy
from branchmem.merge.confidence_rule import ConfidenceRuleMerge
from branchmem.merge.three_way_llm import ThreeWayLLMMerge


class HybridMerge(MergeStrategy):
    name = "hybrid"

    def __init__(self, backend: LLMBackend) -> None:
        self.rule = ConfidenceRuleMerge()
        self.llm = ThreeWayLLMMerge(backend=backend)

    def merge(
        self, ancestor: MemoryBranch, branch_a: MemoryBranch, branch_b: MemoryBranch, use_cache: bool = True
    ) -> MergeResult:
        rule_result = self.rule.merge(ancestor, branch_a, branch_b)

        # Keys the deterministic rule could not decide (tied confidence).
        escalate = {
            rf.fact.key()
            for rf in rule_result.resulting_facts
            if rf.resolution == Resolution.FLAGGED_UNRESOLVED
        }
        if not escalate:
            return MergeResult(strategy_name=self.name, resulting_facts=rule_result.resulting_facts)

        # One batched LLM call for the scenario (same prompt as a standalone
        # ThreeWayLLMMerge, so it hits the existing content-hash cache); we then
        # adopt its decision only for the rule-abstained keys.
        llm_result = self.llm.merge(ancestor, branch_a, branch_b, use_cache=use_cache)
        llm_by_key = {rf.fact.key(): rf for rf in llm_result.resulting_facts}

        merged = []
        for rf in rule_result.resulting_facts:
            key = rf.fact.key()
            if key in escalate and key in llm_by_key:
                merged.append(llm_by_key[key])
            else:
                merged.append(rf)
        return MergeResult(strategy_name=self.name, resulting_facts=merged)
