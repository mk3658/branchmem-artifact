"""Tests for the WikiDivergence extraction helpers.

These are the parts of the external-validity collection that can silently
corrupt it: an infobox parser that mis-splits fields, a normaliser that leaves
markup behind or invents truncated values, and the filter that decides whether
two values actually disagree. Network access is not needed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_wikidivergence import (  # noqa: E402
    channel_for,
    confidence_for,
    is_formatting_variant,
    normalise,
    parse_infobox,
    source_for,
)


def test_parse_infobox_splits_top_level_fields_only():
    wikitext = """{{Short description|An airline}}
{{Infobox airline
| airline = Delta Air Lines, Inc.
| hubs = {{ubl | [[Atlanta]] | [[Detroit|DTW]] }}
| num_employees = 100,000
}}
Body text here."""
    fields = parse_infobox(wikitext)
    assert fields["airline"] == "Delta Air Lines, Inc."
    assert fields["num_employees"] == "100,000"
    # The pipes inside the nested template and the link must not create fields.
    assert "[[atlanta]]" not in fields
    assert set(fields) == {"airline", "hubs", "num_employees"}


def test_parse_infobox_returns_empty_without_an_infobox():
    assert parse_infobox("{{Short description|No box here}}\nJust prose.") == {}


def test_parse_infobox_handles_the_first_infobox_when_nested_templates_precede():
    wikitext = "{{pp-semi-indef}}\n{{Infobox company\n| name = Acme\n| type = Public\n}}"
    assert parse_infobox(wikitext) == {"name": "Acme", "type": "Public"}


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("[[Atlanta, Georgia|Atlanta]]", "Atlanta"),
        ("Public<ref name=x>{{cite web|url=http://e.com}}</ref>", "Public"),
        ("'''Acme''' Corporation", "Acme Corporation"),
        ("12,700&nbsp;(2024)", "12,700 (2024)"),
        ("Leeds <!-- do not change -->", "Leeds"),
        ("{{increase}} $4.079 billion (2023)", "$4.079 billion (2023)"),
    ],
)
def test_normalise_strips_markup_without_losing_the_value(raw, expected):
    assert normalise(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "{{Unbulleted list| |S&P 500 component}}",   # template residue
        "{{US$|4.079 billion}} (2023)",               # substance inside a template
        "(2025)",                                     # only a parenthetical survives
        "|",                                          # pure markup
        "",
    ],
)
def test_normalise_rejects_values_it_cannot_clean(raw):
    assert normalise(raw) == ""


def test_normalise_is_idempotent():
    once = normalise("[[Molineux|Molineux Stadium]]<ref>x</ref>")
    assert once == "Molineux Stadium"
    assert normalise(once) == once


@pytest.mark.parametrize(
    "a,b",
    [
        ("Public", "Public company"),
        ("3M", "3M Company"),
        ("Greensboro, North Carolina, U.S", "Greensboro, North Carolina, United States"),
        ("AFC Bournemouth", "Bournemouth"),
        ("Acme Inc", "Acme Incorporated"),
    ],
)
def test_formatting_variants_are_not_conflicts(a, b):
    assert is_formatting_variant(a, b)
    assert is_formatting_variant(b, a)


@pytest.mark.parametrize(
    "a,b",
    [
        ("32,050", "31,750"),
        ("Arthur Oliver Smith", "Charles Jeremiah Smith"),
        ("Paint and coating manufacturing", "Chemicals"),
        ("0.49", "0.44"),
    ],
)
def test_substantive_disagreements_are_kept(a, b):
    assert not is_formatting_variant(a, b)


def test_channel_router_reads_the_field_name():
    assert channel_for("leader_name") == "leadership"
    assert channel_for("num_employees") == "quantity"
    assert channel_for("headquarters") == "location"
    assert channel_for("clubname") == "identity"
    assert channel_for("aoc") == "other"


def test_confidence_is_monotone_in_edit_count_and_bounded():
    confs = [confidence_for(n) for n in (0, 10, 1000, 100_000, 10_000_000)]
    assert confs == sorted(confs)
    assert all(0.50 <= c <= 0.98 for c in confs)
    assert confidence_for(0) == 0.50


def test_source_mapping_follows_account_type_and_groups():
    assert source_for(True, []) == "inference"
    assert source_for(False, ["user"]) == "observation"
    assert source_for(False, ["extendedconfirmed", "user"]) == "user"
    assert source_for(False, ["sysop"]) == "user"
