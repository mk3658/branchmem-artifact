from branchmem.branchguard.contextual import (
    Action,
    AnytimeRiskMonitor,
    bayesian_source_action,
    certify_channel,
    certify_channels,
    clopper_pearson_interval,
)


def test_exact_interval_extremes():
    lo, hi = clopper_pearson_interval(0, 20, 0.05)
    assert lo == 0.0 and 0.0 < hi < 0.2
    lo, hi = clopper_pearson_interval(20, 20, 0.05)
    assert 0.8 < lo < 1.0 and hi == 1.0


def test_stationary_follow_and_invert():
    assert certify_channel("good", 190, 200).action == Action.FOLLOW
    assert certify_channel("bad", 10, 200).action == Action.INVERT


def test_simultaneous_small_channel_abstains():
    certs = certify_channels({"large": (190, 200), "small": (9, 10)}, delta=0.05)
    assert certs["large"].action == Action.FOLLOW
    assert certs["small"].action == Action.ABSTAIN


def test_drift_budget_tightens_certificate():
    ordinary = certify_channel("c", 90, 100, drift_budget=0.0)
    robust = certify_channel("c", 90, 100, drift_budget=0.10)
    assert ordinary.action == Action.FOLLOW
    assert robust.action == Action.ABSTAIN


def test_bayesian_baseline_is_more_aggressive():
    assert bayesian_source_action(8, 10) == Action.FOLLOW
    assert certify_channel("c", 8, 10).action == Action.ABSTAIN


def test_anytime_monitor_revokes_under_high_error():
    monitor = AnytimeRiskMonitor(target_risk=0.25, alt_risk=0.60, alpha=0.05)
    for _ in range(20):
        monitor.update(True)
        if monitor.revoked:
            break
    assert monitor.revoked


def test_blockwise_monitor_revokes_after_restart():
    from branchmem.branchguard.contextual import BlockwiseRiskMonitor
    monitor = BlockwiseRiskMonitor(target_risk=0.25, alt_risk=0.65, total_alpha=0.05, min_block_size=12)
    # One safe block should not poison the later change detector.
    for _ in range(12):
        monitor.update(False)
    for _ in range(30):
        monitor.update(True)
        if monitor.revoked:
            break
    assert monitor.revoked


def test_blockwise_monitor_is_detectable_at_every_block_index():
    """Regression for a fixed-block-length bug.

    With a FIXED block length the block-j threshold (-log alpha_j, growing
    like 2 log j) eventually exceeds the largest log-likelihood ratio an
    all-error block can produce (L * log(alt/target)), so late blocks cannot
    reject at all and expected detection delay is infinite. Block length must
    grow with the block index.
    """
    import math
    from branchmem.branchguard.contextual import BlockwiseRiskMonitor

    monitor = BlockwiseRiskMonitor()
    step = math.log(monitor.alt_risk / monitor.target_risk)
    for j in (1, 2, 10, 100, 232, 10_000, 1_000_000):
        monitor.block_index = j
        attainable = monitor.block_size * step
        assert attainable > monitor.threshold, f"block {j} structurally undetectable"
        # expected evidence under the alternative clears the threshold with margin
        assert monitor.block_size * monitor.kl_rate >= monitor.margin * monitor.threshold


def test_blockwise_monitor_rejects_non_growing_margin():
    import pytest
    from branchmem.branchguard.contextual import BlockwiseRiskMonitor

    with pytest.raises(ValueError):
        BlockwiseRiskMonitor(margin=1.0)
