from branchmem.branchguard.core import Action, bayes_action, certify_provenance


def test_bayes_regions():
    assert bayes_action(0.10) == "B"
    assert bayes_action(0.50) == "abstain"
    assert bayes_action(0.90) == "A"


def test_perfect_channel_certified_with_enough_data():
    cert = certify_provenance(correct=100, n=100)
    assert cert.action == Action.FOLLOW_PROVENANCE


def test_inverted_channel_certified_with_enough_data():
    cert = certify_provenance(correct=0, n=100)
    assert cert.action == Action.INVERT_PROVENANCE


def test_ambiguous_channel_abstains():
    cert = certify_provenance(correct=50, n=100)
    assert cert.action == Action.ABSTAIN


def test_small_sample_is_conservative():
    cert = certify_provenance(correct=25, n=25)
    assert cert.action == Action.ABSTAIN
