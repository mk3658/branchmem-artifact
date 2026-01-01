"""Generator for the `misleading_provenance` condition (post-hoc, exploratory).

Motivation. Every prior category is solvable (or unsolvable) by reading one
metadata field: `resolvable` ties ground truth *to* source/confidence, so a
one-line rule wins; `semantic_resolvable` removes the field entirely, so the
rule can only abstain. Neither separates *reading a field* from *semantic
reasoning*, because in neither does the field ever point the wrong way.

This condition does. Each item carries BOTH signals, pointing in OPPOSITE
directions:
  - a strict ancestor constraint (e.g. "is completely blind and cannot read
    standard printed text visually"), and
  - a provenance signal that favours the value CONTRADICTING that constraint
    (the incompatible branch is user-sourced/high-confidence; the compatible
    branch is inference-sourced/low-confidence).

Ground truth is the value semantically compatible with the ancestor
constraint. Therefore:
  - ConfidenceRuleMerge (and any confidence-following strategy) is WRONG by
    construction: it scores exactly 0.000, not by abstaining but by committing
    to the constraint-violating value;
  - a strategy that genuinely reasons over ancestor semantics can score above
    0 by OVERRIDING the misleading field.

This is the discriminative test the benchmark previously lacked: the answer is
not recoverable from the metadata field, and the field is an active trap. It is
informative in both directions. If the LLM overrides the field, LLM merging
supplies something no field-reading rule can. If it follows the field off the
cliff, then LLM merging on this benchmark is provenance-following rather than
semantic reasoning, which is itself a sharper finding than "a rule matches it".

Templates are reused verbatim from the realistic strict-constraint bank so the
content is held fixed and ONLY the provenance assignment changes relative to
Appendix M.11's realism check.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

from branchmem.benchmark.scenario_generator import Scenario
from branchmem.memory.branch_sim import BranchSimulator, UpdateSpec

_DEFAULT_TEMPLATES_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "realistic_semantic_templates.json"
)


@dataclass
class MisleadingProvenanceConfig:
    entity: str = "user"
    # The branch that CONTRADICTS the ancestor constraint gets the strong signal.
    misleading_source: str = "user"
    misleading_confidence: float = 0.95
    # The branch COMPATIBLE with the constraint (ground truth) gets the weak one.
    truthful_source: str = "inference"
    truthful_confidence: float = 0.55
    divergence_span: float = 10.0


def load_templates(path: Path | str = _DEFAULT_TEMPLATES_PATH) -> list[dict]:
    with open(path) as f:
        return json.load(f)["templates"]


def generate_misleading_provenance_scenarios(
    n_scenarios: int,
    seed: int,
    config: MisleadingProvenanceConfig | None = None,
    templates_path: Path | str = _DEFAULT_TEMPLATES_PATH,
) -> list[Scenario]:
    """One misleading-provenance conflict per scenario.

    Template convention (inherited): `branch_a_value` STRICTLY CONTRADICTS
    `ancestor_value`; `branch_b_value` is compatible with it. Ground truth is
    therefore always the b-side text, regardless of which physical branch
    carries it after the seeded swap, and the misleading high-confidence signal
    always rides on the a-side (contradicting) text.
    """
    config = config or MisleadingProvenanceConfig()
    templates = load_templates(templates_path)
    sim = BranchSimulator()
    scenarios: list[Scenario] = []

    for i in range(n_scenarios):
        rng = random.Random(f"mislead:{seed}:{i}")
        template = templates[i % len(templates)]

        # Randomize which physical branch carries the contradicting text, so a
        # strategy cannot exploit a positional shortcut.
        swap = rng.random() < 0.5
        contradicting = template["branch_a_value"]  # violates the constraint
        compatible = template["branch_b_value"]  # ground truth
        value_a = compatible if swap else contradicting
        value_b = contradicting if swap else compatible
        # The misleading (high-confidence) signal follows the CONTRADICTING text.
        a_is_contradicting = not swap

        ancestor = sim.create_common_ancestor(
            facts=[(config.entity, template["ancestor_predicate"], template["ancestor_value"])],
            timestamp=0.0,
        )

        def fields(is_contradicting: bool):
            if is_contradicting:
                return config.misleading_source, config.misleading_confidence
            return config.truthful_source, config.truthful_confidence

        src_a, conf_a = fields(a_is_contradicting)
        src_b, conf_b = fields(not a_is_contradicting)

        # Timestamps drawn independently of correctness: uninformative by design.
        ts_a = config.divergence_span * rng.uniform(0.1, 1.0)
        ts_b = config.divergence_span * rng.uniform(0.1, 1.0)

        updates_a = [
            UpdateSpec(
                entity=config.entity, predicate=template["predicate"], value=value_a,
                timestamp=1.0 + ts_a, source=src_a, confidence=conf_a,
                divergence_type="resolvable_conflict", ground_truth_value=compatible,
            )
        ]
        updates_b = [
            UpdateSpec(
                entity=config.entity, predicate=template["predicate"], value=value_b,
                timestamp=1.0 + ts_b, source=src_b, confidence=conf_b,
                divergence_type="resolvable_conflict", ground_truth_value=compatible,
            )
        ]

        branch_a, branch_b, conflict_pairs = sim.diverge(
            ancestor, updates_a, updates_b, fork_point_timestamp=1.0
        )
        scenarios.append(
            Scenario(
                scenario_id=f"mislead_{seed}_{i:04d}",
                ancestor=ancestor,
                branch_a=branch_a,
                branch_b=branch_b,
                conflict_pairs=conflict_pairs,
                source="synthetic",
                metadata={
                    "category": "misleading_provenance",
                    "template_predicate": template["predicate"],
                    "contradicting_is_a": a_is_contradicting,
                    "divergence_span": config.divergence_span,
                },
            )
        )
    return scenarios
