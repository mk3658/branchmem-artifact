"""The manifest's cache table is a claim about the artifact, so it is checked.

ARTIFACT_MANIFEST.md tells a reader that every reported number is regenerable
without an API key, and backs that with a per-model count of `llm_cache/`. That
table was hand-maintained and drifted badly -- it claimed 6,105 calls across
five models when the cache held 15,040 across ten, omitting `o3`, `qwen3:8b`,
`qwen2.5:7b`, `mistral:7b` and `deepseek-r1:8b` outright. A reader checking the
artifact's honesty would have found the first number they tested to be wrong.

So the table is recounted here rather than trusted. The manifest is withheld
from the anonymized artifact by design, so its absence skips rather than fails.
"""
from __future__ import annotations

import collections
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "ARTIFACT_MANIFEST.md"
CACHE = ROOT / "llm_cache"

# | `model-id` | 1,234 | covers... |
ROW = re.compile(r"^\|\s*`([^`]+)`\s*\|\s*([\d,]+)\s*\|", re.M)
TOTAL = re.compile(r"^\|\s*\*\*Total\*\*\s*\|\s*\*\*([\d,]+)\*\*\s*\|", re.M)


def _table() -> str:
    """The cache-coverage section, isolated from the rest of the manifest."""
    if not MANIFEST.exists():
        pytest.skip("ARTIFACT_MANIFEST.md is withheld from the packaged artifact")
    text = MANIFEST.read_text()
    start = text.index("## Model-call cache coverage")
    end = text.index("\n## ", start + 1)
    return text[start:end]


@pytest.fixture(scope="module")
def counted() -> dict[str, int]:
    if not CACHE.is_dir():
        pytest.skip("llm_cache/ is not present")
    c: collections.Counter[str] = collections.Counter()
    for f in CACHE.glob("*.json"):
        try:
            c[json.loads(f.read_text()).get("model", "<no model field>")] += 1
        except (json.JSONDecodeError, OSError) as exc:  # a corrupt entry is a finding
            pytest.fail(f"unreadable cache entry {f.name}: {exc}")
    return dict(c)


def _claimed() -> dict[str, int]:
    return {m: int(n.replace(",", "")) for m, n in ROW.findall(_table())}


def test_every_cached_model_appears_in_the_table(counted):
    """An omitted model is the failure that actually happened: five were missing,
    and a reader would have had no way to know the table was partial."""
    missing = sorted(set(counted) - set(_claimed()))
    assert not missing, f"models in llm_cache/ but absent from the table: {missing}"


def test_the_table_invents_no_models(counted):
    extra = sorted(set(_claimed()) - set(counted))
    assert not extra, f"table lists models with no cache entries: {extra}"


def test_per_model_counts_match_the_cache(counted):
    claimed = _claimed()
    wrong = {m: (n, counted[m]) for m, n in claimed.items()
             if m in counted and n != counted[m]}
    assert not wrong, f"table vs cache, as {{model: (claimed, actual)}}: {wrong}"


def test_the_total_is_the_sum_of_the_cache(counted):
    m = TOTAL.search(_table())
    assert m, "the cache-coverage table has no **Total** row"
    assert int(m.group(1).replace(",", "")) == sum(counted.values())


def test_only_one_figure_generator():
    """paper/make_figures.py was a byte-identical duplicate of the scripts/ copy,
    and each wrote its PNGs beside itself -- so which one you ran decided where
    the paper's figures came from. One copy, one destination."""
    copies = sorted(p.relative_to(ROOT).as_posix()
                    for p in ROOT.rglob("make_figures.py")
                    if ".git" not in p.parts and ".venv" not in p.parts)
    assert copies == ["scripts/make_figures.py"], f"found {copies}"
