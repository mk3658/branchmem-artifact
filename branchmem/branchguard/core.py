"""Calibration-certified routing for two-branch semantic memory conflicts.

This standalone reference implementation accompanies the revised BranchMem
manuscript. It does not depend on the missing original experiment repository.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math


class Action(str, Enum):
    FOLLOW_PROVENANCE = "follow_provenance"
    INVERT_PROVENANCE = "invert_provenance"
    ABSTAIN = "abstain"


@dataclass(frozen=True)
class Certificate:
    action: Action
    empirical_reliability: float
    lower: float
    upper: float
    epsilon: float
    n: int
    correct: int


def hoeffding_interval(correct: int, n: int, delta: float = 0.05) -> tuple[float, float, float]:
    """Return a two-sided Hoeffding interval for Bernoulli reliability."""
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 <= correct <= n:
        raise ValueError("correct must lie in [0, n]")
    if not 0 < delta < 1:
        raise ValueError("delta must lie in (0, 1)")
    p_hat = correct / n
    epsilon = math.sqrt(math.log(2.0 / delta) / (2.0 * n))
    return max(0.0, p_hat - epsilon), min(1.0, p_hat + epsilon), epsilon


def certify_provenance(
    correct: int,
    n: int,
    abstention_loss: float = 0.25,
    delta: float = 0.05,
) -> Certificate:
    """Certify whether to follow, invert, or reject a provenance channel.

    Wrong-commit loss is normalized to 1 and correct-commit loss to 0.
    Under the i.i.d./stationarity assumptions in the paper, a certified
    provenance action has expected wrong-commit probability at most
    ``abstention_loss`` with probability at least ``1-delta`` over the
    calibration sample.
    """
    if not 0 < abstention_loss < 0.5:
        raise ValueError("abstention_loss must lie in (0, 0.5)")
    lower, upper, epsilon = hoeffding_interval(correct, n, delta)
    p_hat = correct / n
    if lower >= 1.0 - abstention_loss:
        action = Action.FOLLOW_PROVENANCE
    elif upper <= abstention_loss:
        action = Action.INVERT_PROVENANCE
    else:
        action = Action.ABSTAIN
    return Certificate(action, p_hat, lower, upper, epsilon, n, correct)


def bayes_action(p_branch_a: float, abstention_loss: float = 0.25) -> str:
    """Bayes-optimal action for posterior P(Y=A|O)."""
    if not 0 <= p_branch_a <= 1:
        raise ValueError("p_branch_a must lie in [0, 1]")
    if not 0 < abstention_loss < 0.5:
        raise ValueError("abstention_loss must lie in (0, 0.5)")
    if p_branch_a <= abstention_loss:
        return "B"
    if p_branch_a >= 1.0 - abstention_loss:
        return "A"
    return "abstain"
