"""The specification ablation's claim rests on what its prompts do NOT say.

The paper reports that a prompt which merely raises the standing-rule question,
naming no categories and marking no items, takes the frontier model from .500
to .938. That claim is only meaningful if P4 and P5 really withhold what P3
supplies. These tests pin that difference, so an edit to the prompts fails the
suite rather than silently changing what the reported number means.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from branchmem.merge.three_way_llm import _PROMPT_TEMPLATE, _SYSTEM_PROMPT, ThreeWayLLMMerge
from run_specification_ablation import (  # type: ignore
    _SHIPPED_ABSTAIN,
    _SHIPPED_CLAUSE,
    _SHIPPED_INPUT_DESC,
    build_prompts,
)

# Words that would hand the model the classification instead of asking for it.
CATEGORY_WORDS = (
    "medical", "allergy", "allergies", "physical limitation", "restriction",
    "lease", "contract", "insurer", "venue", "employer", "union", "library",
    "building", "landlord",
)


@pytest.fixture(scope="module")
def prompts() -> dict[str, str]:
    return build_prompts()


def test_all_seven_conditions_present(prompts):
    assert set(prompts) == {
        "P0_unprimed", "P1_shipped", "P2_neutral", "P3_marked",
        "P4_unmarked_rule", "P5_unmarked_norule", "P6_elicited_cot",
    }


def test_p0_mentions_no_metadata_ordering_or_standing_rule(prompts):
    """P0 is the unprimed control: the number reported for it only answers
    "what does the model do when nothing points at the fields?" if the prompt
    names none of them. The shipped prompt names them in three places, so all
    three must be gone."""
    low = prompts["P0_unprimed"].lower().replace("unresolvable", "")
    for banned in ("source", "confidence", "trust", "standing", "rule",
                   "trustworthy", "user-sourced"):
        assert banned not in low, f"P0 mentions {banned!r}"
    for span in (_SHIPPED_INPUT_DESC, _SHIPPED_CLAUSE, _SHIPPED_ABSTAIN):
        assert span not in prompts["P0_unprimed"]


def test_p0_keeps_the_task_and_the_abstention_option(prompts):
    """What P0 removes is the priming, not the task. The no-timestamp rule and
    the flag-as-unresolvable instruction are shared by every condition; without
    them P0's items would not be the same items."""
    p0 = prompts["P0_unprimed"]
    assert "Do NOT use timestamps to decide" in p0
    assert "flag it as unresolvable" in p0
    assert p0.startswith("You are reconciling two independently-updated branches")


def test_p0_is_the_most_austere_condition(prompts):
    """Every other condition says strictly more than P0 about how to decide."""
    n0 = len(prompts["P0_unprimed"])
    for pid, p in prompts.items():
        if pid != "P0_unprimed":
            assert len(p) > n0, f"{pid} is no longer than the unprimed control"


def test_p1_is_the_shipped_prompt_verbatim(prompts):
    """P1 must be byte-identical to the locked run's prompt, or it would not be
    cache-served and could not reproduce the preregistered row."""
    assert prompts["P1_shipped"] == _SYSTEM_PROMPT


def test_conditions_differ_only_in_the_specification_clause(prompts):
    """Every condition is the shipped prompt with one clause swapped, so the
    only moving part between any two is how the signals are to be weighed."""
    for pid in ("P2_neutral", "P3_marked", "P4_unmarked_rule", "P5_unmarked_norule"):
        p = prompts[pid]  # P0 is deliberately excluded: it swaps three spans
        assert _SHIPPED_CLAUSE not in p, f"{pid} still contains the shipped clause"
        # the surrounding text survives intact on both sides
        head = _SYSTEM_PROMPT.split(_SHIPPED_CLAUSE)[0]
        tail = _SYSTEM_PROMPT.split(_SHIPPED_CLAUSE)[1]
        assert p.startswith(head), pid
        assert p.endswith(tail), pid


def test_p4_and_p5_name_no_qualifying_categories(prompts):
    """This is the control's whole point: P4/P5 may raise the question but must
    not tell the model which kinds of entry count, which P3 does."""
    for pid in ("P4_unmarked_rule", "P5_unmarked_norule"):
        low = prompts[pid].lower()
        hits = [w for w in CATEGORY_WORDS if w in low]
        assert hits == [], f"{pid} leaks qualifying categories: {hits}"


def test_p3_does_supply_categories(prompts):
    """The contrast the experiment relies on: P3 is the pre-solved condition."""
    low = prompts["P3_marked"].lower()
    assert any(w in low for w in CATEGORY_WORDS)


def _inserted_clause(prompt: str) -> str:
    """The clause this condition put where the shipped trust ordering was.

    Scoped so these tests judge the swapped clause only; "even if" also occurs
    in the shipped no-timestamp sentence, which every condition inherits.
    """
    head, tail = _SYSTEM_PROMPT.split(_SHIPPED_CLAUSE)
    assert prompt.startswith(head) and prompt.endswith(tail)
    return prompt[len(head):len(prompt) - len(tail)].lower()


def test_p5_states_no_decision_rule(prompts):
    """P5 raises the question and stops; its clause must not say what follows
    from the answer, or it would be P4."""
    clause = _inserted_clause(prompts["P5_unmarked_norule"])
    for giveaway in ("keep the compatible value", "even if", "incompatible with it"):
        assert giveaway not in clause, f"P5 states a decision rule: {giveaway!r}"


def test_p4_does_state_a_decision_rule(prompts):
    clause = _inserted_clause(prompts["P4_unmarked_rule"])
    assert "even if" in clause and "standing commitment" in clause


def test_prompt_suffix_defaults_to_an_unchanged_user_template():
    """The CoT condition appends to the user template. Every other run must
    leave it byte-identical, or the locked caches would miss."""
    assert ThreeWayLLMMerge(backend=None).prompt_suffix is None
    assert ThreeWayLLMMerge(backend=None, prompt_suffix="X").prompt_suffix == "X"
    assert "{ancestor_json}" in _PROMPT_TEMPLATE
