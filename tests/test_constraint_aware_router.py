"""The two-stage router must read the signal its own classification selects.

These tests pin the behaviour the mitigation claim rests on: when the stored
entry is classified as standing, the metadata must not decide the merge; when
it is classified as ordinary, the metadata must decide it; and when the
classification is unclear, nothing is committed. A mock classifier is used so
the routing branch under test is the only variable.
"""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from branchmem.memory.branch_sim import BranchSimulator, UpdateSpec
from branchmem.memory.schemas import Resolution
from branchmem.merge.constraint_aware_router import ConstraintAwareRouterMerge


@dataclass
class _Reply:
    text: str


class _FixedClassifier:
    """Returns one canned verdict, and records what it was asked."""

    def __init__(self, verdict: str) -> None:
        self.verdict = verdict
        self.prompts: list[str] = []

    def complete(self, prompt, system=None, use_cache=True):  # noqa: D102
        self.prompts.append(prompt)
        return _Reply(self.verdict)


@pytest.fixture(scope="module")
def detector():
    from branchmem.conflict.nli_detector import NLIConflictDetector
    return NLIConflictDetector()


def _scenario(ancestor_value: str, strong_value: str, weak_value: str):
    """One conflict where `strong_value` carries the better provenance."""
    sim = BranchSimulator()
    ancestor = sim.create_common_ancestor(
        facts=[("user", "test_key", ancestor_value)], timestamp=0.0)
    # Both branches must declare the SAME ground truth; these tests are about
    # routing behaviour, so the declared truth is irrelevant and held fixed.
    mk = lambda v, src, conf: [UpdateSpec(  # noqa: E731
        entity="user", predicate="test_key", value=v, timestamp=2.0,
        source=src, confidence=conf, divergence_type="resolvable_conflict",
        ground_truth_value=strong_value)]
    a, b, _ = sim.diverge(ancestor, mk(strong_value, "user", 0.95),
                          mk(weak_value, "inference", 0.55), fork_point_timestamp=1.0)
    return ancestor, a, b


def _committed(result):
    for rf in result.resulting_facts:
        if rf.fact.key() == ("user", "test_key") and rf.resolution != Resolution.DROPPED:
            return (None if rf.resolution == Resolution.FLAGGED_UNRESOLVED
                    else rf.fact.value)
    return None


def test_ordinary_route_follows_the_metadata(detector):
    clf = _FixedClassifier("ordinary")
    strat = ConstraintAwareRouterMerge(backend=clf, detector=detector)
    anc, a, b = _scenario("The user parks in bay 14.",
                          "The user now parks in bay 31.",
                          "The user still parks in bay 14.")
    assert _committed(strat.merge(anc, a, b)) == "The user now parks in bay 31."
    assert strat.n_ordinary == 1 and strat.n_standing == 0


def test_standing_route_ignores_the_metadata(detector):
    """The strong-provenance value contradicts the ancestor, so it must lose."""
    clf = _FixedClassifier("standing")
    strat = ConstraintAwareRouterMerge(backend=clf, detector=detector)
    anc, a, b = _scenario("The user is not allowed to park in bay 1.",
                          "The user parks in bay 1.",
                          "The user does not park in bay 1.")
    assert _committed(strat.merge(anc, a, b)) != "The user parks in bay 1."
    assert strat.n_standing == 1 and strat.n_ordinary == 0


def test_unclear_route_abstains(detector):
    clf = _FixedClassifier("unclear")
    strat = ConstraintAwareRouterMerge(backend=clf, detector=detector)
    anc, a, b = _scenario("The user parks in bay 14.",
                          "The user now parks in bay 31.",
                          "The user still parks in bay 14.")
    assert _committed(strat.merge(anc, a, b)) is None
    assert strat.n_unclear == 1


def test_unparseable_classification_abstains_rather_than_guessing(detector):
    clf = _FixedClassifier("I am not sure what you mean.")
    strat = ConstraintAwareRouterMerge(backend=clf, detector=detector)
    anc, a, b = _scenario("The user parks in bay 14.",
                          "The user now parks in bay 31.",
                          "The user still parks in bay 14.")
    assert _committed(strat.merge(anc, a, b)) is None


def test_classifier_is_never_shown_the_metadata_or_the_candidates(detector):
    """Stage one must classify the stored entry, not pick a winner."""
    clf = _FixedClassifier("standing")
    strat = ConstraintAwareRouterMerge(backend=clf, detector=detector)
    anc, a, b = _scenario("The user is not allowed to park in bay 1.",
                          "The user parks in bay 1.",
                          "The user does not park in bay 1.")
    strat.merge(anc, a, b)
    prompt = clf.prompts[0]
    assert "The user is not allowed to park in bay 1." in prompt
    for leak in ("bay 1.", "confidence", "0.95", "inference", "user-sourced"):
        if leak == "bay 1.":
            continue          # appears inside the ancestor itself
        assert leak not in prompt, f"stage-one prompt leaks {leak!r}"
    assert "The user does not park in bay 1." not in prompt
