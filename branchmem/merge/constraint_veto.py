"""ConstraintVetoMerge: check for an explicit ancestor contradiction, then fall
back to provenance.

Two of the paper's policies are complementary rather than ordered.
``ConfidenceRuleMerge`` is strong where provenance is informative and
catastrophic where it is inverted; the NLI gate of
``scripts/run_operational_baselines.py`` is the mirror image, surviving
inverted provenance but weak when provenance is the only signal (it decides on
*relative* contradiction even when neither value actually violates anything).

This strategy composes them in the order a safety-minded deployment would use.
For each conflict:

1. Score each branch value for contradiction against the ancestor value.
2. If exactly one value's contradiction clears an ABSOLUTE threshold while the
   other stays below it, that is a detected constraint violation: veto the
   violating branch and keep the other, whatever provenance says.
3. Otherwise no constraint is at stake, so apply the provenance rule: keep the
   higher-confidence branch, abstaining on a tie.

Step 2 is deliberately conservative. A relative comparison fires on every
conflict and therefore imports the NLI model's noise into cases it has nothing
to say about; requiring one side to cross an absolute threshold and the other
to stay under it means the veto fires only on a positively detected violation.

Cost: one local cross-encoder pass per branch value, on CPU, with no API call.
"""

from __future__ import annotations

from branchmem.conflict.nli_detector import NLIConflictDetector
from branchmem.memory.schemas import MemoryBranch, MergeResult, Resolution, ResolvedFact
from branchmem.merge.base import MergeStrategy, current_facts_by_key, find_collisions


class ConstraintVetoMerge(MergeStrategy):
    name = "constraint_veto"

    def __init__(
        self,
        detector: NLIConflictDetector | None = None,
        veto_threshold: float = 0.60,
        clear_threshold: float = 0.30,
    ) -> None:
        if not 0.0 < clear_threshold < veto_threshold < 1.0:
            raise ValueError("require 0 < clear_threshold < veto_threshold < 1")
        self.detector = detector or NLIConflictDetector()
        self.veto_threshold = veto_threshold
        self.clear_threshold = clear_threshold
        self.n_vetoes = 0
        self.n_provenance = 0
        self.n_abstained = 0

    def _contradiction(self, premise: str, hypothesis: str) -> float:
        return self.detector._predict([(premise, hypothesis)])[0]["contradiction"]

    def merge(self, ancestor: MemoryBranch, branch_a: MemoryBranch, branch_b: MemoryBranch) -> MergeResult:
        anc = current_facts_by_key(ancestor)
        a_facts = current_facts_by_key(branch_a)
        b_facts = current_facts_by_key(branch_b)
        collisions = find_collisions(branch_a, branch_b)

        resolved: list[ResolvedFact] = []
        for key in sorted((set(a_facts) | set(b_facts)) - set(collisions)):
            fact = a_facts.get(key) or b_facts.get(key)
            resolved.append(
                ResolvedFact(fact=fact, resolution=Resolution.KEPT, source_branch_ids=[fact.branch_id])
            )

        for key, (fa, fb) in collisions.items():
            anc_fact = anc.get(key)
            veto_winner = None
            if anc_fact is not None:
                premise = f"{anc_fact.entity} {anc_fact.value}."
                ca = self._contradiction(premise, f"{fa.entity} {fa.value}.")
                cb = self._contradiction(premise, f"{fb.entity} {fb.value}.")
                if ca >= self.veto_threshold and cb <= self.clear_threshold:
                    veto_winner, reason = fb, f"branch A contradicts the ancestor ({ca:.2f})"
                elif cb >= self.veto_threshold and ca <= self.clear_threshold:
                    veto_winner, reason = fa, f"branch B contradicts the ancestor ({cb:.2f})"

            if veto_winner is not None:
                self.n_vetoes += 1
                resolved.append(
                    ResolvedFact(
                        fact=veto_winner,
                        resolution=Resolution.KEPT,
                        source_branch_ids=[veto_winner.branch_id],
                        justification=f"constraint veto: {reason}",
                    )
                )
                continue

            if fa.confidence == fb.confidence:
                self.n_abstained += 1
                resolved.append(
                    ResolvedFact(
                        fact=fa,
                        resolution=Resolution.FLAGGED_UNRESOLVED,
                        source_branch_ids=[branch_a.branch_id, branch_b.branch_id],
                        justification="no constraint violation detected and no provenance signal",
                    )
                )
                continue

            self.n_provenance += 1
            win = fa if fa.confidence > fb.confidence else fb
            resolved.append(
                ResolvedFact(
                    fact=win,
                    resolution=Resolution.KEPT,
                    source_branch_ids=[win.branch_id],
                    justification=f"no constraint violation detected; higher confidence ({win.confidence:.2f})",
                )
            )

        return MergeResult(strategy_name=self.name, resulting_facts=resolved)
