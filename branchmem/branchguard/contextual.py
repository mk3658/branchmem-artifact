"""Context-conditional and drift-monitored BranchGuard.

This module provides the reference implementation for the revised BranchMem
paper.  It uses exact Clopper-Pearson intervals for finite-sample certification
and an anytime-valid likelihood-ratio e-process for post-deployment revocation.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Hashable, Mapping

from scipy.stats import beta


class Action(str, Enum):
    FOLLOW = "follow_provenance"
    INVERT = "invert_provenance"
    ABSTAIN = "abstain"


@dataclass(frozen=True)
class ChannelCertificate:
    channel: Hashable
    action: Action
    correct: int
    n: int
    empirical_reliability: float
    lower: float
    upper: float
    alpha: float


def clopper_pearson_interval(correct: int, n: int, alpha: float) -> tuple[float, float]:
    """Two-sided exact binomial confidence interval.

    ``alpha`` is the total tail probability for this interval.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 <= correct <= n:
        raise ValueError("correct must lie in [0, n]")
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0, 1)")
    lower = 0.0 if correct == 0 else float(beta.ppf(alpha / 2.0, correct, n - correct + 1))
    upper = 1.0 if correct == n else float(beta.ppf(1.0 - alpha / 2.0, correct + 1, n - correct))
    return lower, upper


def certify_channel(
    channel: Hashable,
    correct: int,
    n: int,
    *,
    abstention_loss: float = 0.25,
    alpha: float = 0.05,
    drift_budget: float = 0.0,
) -> ChannelCertificate:
    """Certify a single provenance channel.

    A drift budget ``d`` tightens the stationary threshold from ``lambda`` to
    ``lambda-d``.  Under an absolute reliability change at most ``d``, the
    resulting deployment wrong-commit risk remains at most ``lambda``.
    """
    if not 0 < abstention_loss < 0.5:
        raise ValueError("abstention_loss must lie in (0, 0.5)")
    if not 0 <= drift_budget < abstention_loss:
        raise ValueError("drift_budget must lie in [0, abstention_loss)")
    lower, upper = clopper_pearson_interval(correct, n, alpha)
    effective_loss = abstention_loss - drift_budget
    if lower >= 1.0 - effective_loss:
        action = Action.FOLLOW
    elif upper <= effective_loss:
        action = Action.INVERT
    else:
        action = Action.ABSTAIN
    return ChannelCertificate(
        channel=channel,
        action=action,
        correct=correct,
        n=n,
        empirical_reliability=correct / n,
        lower=lower,
        upper=upper,
        alpha=alpha,
    )


def certify_channels(
    counts: Mapping[Hashable, tuple[int, int]],
    *,
    abstention_loss: float = 0.25,
    delta: float = 0.05,
    drift_budget: float = 0.0,
) -> dict[Hashable, ChannelCertificate]:
    """Simultaneously certify a pre-specified finite family of channels.

    Uses a Bonferroni correction, so all returned intervals cover jointly with
    probability at least ``1-delta`` under independent or dependent channels.
    Empty channels are represented by abstention certificates.
    """
    if not counts:
        raise ValueError("counts must contain at least one channel")
    if not 0 < delta < 1:
        raise ValueError("delta must lie in (0, 1)")
    per_channel_alpha = delta / len(counts)
    result: dict[Hashable, ChannelCertificate] = {}
    for channel, (correct, n) in counts.items():
        if n == 0:
            result[channel] = ChannelCertificate(
                channel=channel,
                action=Action.ABSTAIN,
                correct=0,
                n=0,
                empirical_reliability=float("nan"),
                lower=0.0,
                upper=1.0,
                alpha=per_channel_alpha,
            )
        else:
            result[channel] = certify_channel(
                channel,
                correct,
                n,
                abstention_loss=abstention_loss,
                alpha=per_channel_alpha,
                drift_budget=drift_budget,
            )
    return result


def bayesian_source_action(
    correct: int,
    n: int,
    *,
    abstention_loss: float = 0.25,
    prior_a: float = 1.0,
    prior_b: float = 1.0,
) -> Action:
    """Posterior-mean BayesianSourceMerge baseline.

    This is deliberately not a certificate: it commits when the beta posterior
    mean crosses the Bayes threshold, without controlling finite-sample error.
    """
    if n < 0 or not 0 <= correct <= n:
        raise ValueError("invalid binomial count")
    posterior_mean = (correct + prior_a) / (n + prior_a + prior_b)
    if posterior_mean >= 1.0 - abstention_loss:
        return Action.FOLLOW
    if posterior_mean <= abstention_loss:
        return Action.INVERT
    return Action.ABSTAIN


@dataclass
class AnytimeRiskMonitor:
    """Anytime-valid revocation monitor for one committed channel.

    The monitor observes audited error indicators.  It tests the composite null
    ``p_error <= target_risk`` against a fixed alternative ``p_error = alt_risk``.
    The likelihood ratio is a nonnegative supermartingale under the null, so
    crossing ``1/alpha`` controls the probability of a false revocation by
    ``alpha`` at any audit time.
    """

    target_risk: float = 0.25
    alt_risk: float = 0.45
    alpha: float = 0.0125
    log_e_value: float = 0.0
    audited: int = 0
    errors: int = 0
    revoked: bool = False

    def __post_init__(self) -> None:
        if not 0 < self.target_risk < self.alt_risk < 1:
            raise ValueError("require 0 < target_risk < alt_risk < 1")
        if not 0 < self.alpha < 1:
            raise ValueError("alpha must lie in (0, 1)")

    @property
    def threshold(self) -> float:
        return -math.log(self.alpha)

    @property
    def e_value(self) -> float:
        return math.exp(min(self.log_e_value, 700.0))

    def update(self, error: bool) -> bool:
        """Add one audited outcome and return the revocation state."""
        if self.revoked:
            return True
        self.audited += 1
        self.errors += int(error)
        if error:
            self.log_e_value += math.log(self.alt_risk / self.target_risk)
        else:
            self.log_e_value += math.log((1.0 - self.alt_risk) / (1.0 - self.target_risk))
        if self.log_e_value >= self.threshold:
            self.revoked = True
        return self.revoked

@dataclass
class BlockwiseRiskMonitor:
    """Anytime-safe drift monitor with alpha-spending periodic restarts.

    Each block runs a likelihood-ratio test of ``p_error <= target_risk``
    against ``p_error = alt_risk``. Block ``j`` receives
    ``alpha_j = total_alpha * 6/(pi^2 j^2)``; a union bound therefore controls
    the probability of any false revocation across all blocks by
    ``total_alpha``. Restarts prevent a long safe prefix from hiding a later
    regime change.

    Block *length* must grow with the block index. The block-``j`` rejection
    threshold is ``-log(alpha_j) = 2 log j + log(pi^2 / (6 * total_alpha))``,
    while the largest log-likelihood-ratio attainable inside a block of fixed
    length ``L`` is ``L * log(alt_risk / target_risk)``. With a FIXED ``L``
    those two cross: for ``L = 20`` at ``(0.25, 0.55, 0.0125)`` every block
    from ``j = 232`` on is structurally undetectable, because even an
    all-error block cannot reach the threshold. Expected detection delay is
    then infinite, since the alternative survives undetected with positive
    probability through the finitely many detectable blocks.

    We therefore size block ``j`` as

        ``L_j = max(min_block_size, ceil(margin * threshold_j / KL))``

    where ``KL = D(Bern(alt_risk) || Bern(target_risk))`` is the expected
    log-evidence gained per audit under the alternative. With ``margin > 1``
    the expected evidence in block ``j`` exceeds its threshold by a factor
    ``margin``, so by Hoeffding the per-block miss probability is bounded by a
    constant ``rho < 1`` uniformly in ``j`` while ``L_j`` grows only
    logarithmically; expected detection delay is therefore finite. The false
    revocation guarantee is unaffected: it depends on the per-block
    supermartingale and the alpha budget, not on block length.
    """

    target_risk: float = 0.25
    alt_risk: float = 0.55
    total_alpha: float = 0.0125
    min_block_size: int = 20
    margin: float = 2.0
    block_index: int = 1
    within_block: int = 0
    audited: int = 0
    errors: int = 0
    log_lr: float = 0.0
    revoked: bool = False

    def __post_init__(self) -> None:
        if not 0 < self.target_risk < self.alt_risk < 1:
            raise ValueError("require 0 < target_risk < alt_risk < 1")
        if not 0 < self.total_alpha < 1:
            raise ValueError("total_alpha must lie in (0, 1)")
        if self.min_block_size <= 0:
            raise ValueError("min_block_size must be positive")
        if self.margin <= 1.0:
            raise ValueError("margin must exceed 1 for finite expected delay")

    @property
    def block_alpha(self) -> float:
        return self.total_alpha * 6.0 / (math.pi**2 * self.block_index**2)

    @property
    def threshold(self) -> float:
        return -math.log(self.block_alpha)

    @property
    def kl_rate(self) -> float:
        """Expected log-evidence per audit under the unsafe alternative."""
        q, lam = self.alt_risk, self.target_risk
        return q * math.log(q / lam) + (1.0 - q) * math.log((1.0 - q) / (1.0 - lam))

    @property
    def block_size(self) -> int:
        """Length of the current block; grows like ``log(block_index)``."""
        needed = math.ceil(self.margin * self.threshold / self.kl_rate)
        return max(self.min_block_size, int(needed))

    def update(self, error: bool) -> bool:
        if self.revoked:
            return True
        self.audited += 1
        self.within_block += 1
        self.errors += int(error)
        if error:
            self.log_lr += math.log(self.alt_risk / self.target_risk)
        else:
            self.log_lr += math.log((1.0 - self.alt_risk) / (1.0 - self.target_risk))
        if self.log_lr >= self.threshold:
            self.revoked = True
            return True
        if self.within_block >= self.block_size:
            self.block_index += 1
            self.within_block = 0
            self.log_lr = 0.0
        return False
