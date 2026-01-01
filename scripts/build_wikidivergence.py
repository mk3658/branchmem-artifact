#!/usr/bin/env python3
"""Build WikiDivergence: an independently sourced collection of naturally
occurring semantic conflicts, with real provenance metadata and labels we did
not author.

Why this exists. Every other collection in the paper is constructed: the
synthetic generators script the divergence, and the MemoryAgentBench extension
derives two branches from one pre-existing history. Neither observes two
histories that diverged on their own. Wikipedia's revision history does: an
infobox field is a key, each editor who changes it writes a value without
coordinating with the editor who changes it next, and the community later
settles on one value. That is an asynchronous branch merge that happened
without us.

Construction (one item = one three-way merge):
  * ancestor  V0 -- the field's value before either edit;
  * branch A  V1 -- the value editor E1 wrote, with E1's real metadata;
  * branch B  V2 -- the value editor E2 wrote next, with E2's real metadata;
  * label        -- the value the field holds in the article's CURRENT
                    revision, kept only when it equals V1 or V2.

The label is the community's eventual reconciliation. It is independent of our
provenance mapping, of any model, and of us: nobody involved in this paper
chose it, and it was fixed before we looked.

Provenance mapping (pre-declared, derived from real metadata, never invented):
  * source = "user" for an editor in a trusted Wikipedia group
    (extendedconfirmed/sysop/autoreviewer/rollbacker), "observation" for any
    other registered editor, "inference" for an anonymous (IP) editor;
  * confidence = clip(0.50 + 0.08*log10(1+editcount), 0.50, 0.98), i.e.
    monotone in the editor's lifetime edit count, Wikipedia's standard
    experience proxy;
  * channel = a family read off the FIELD NAME, so it is available before any
    decision: leadership / quantity / location / identity / other.

Privacy. Editors are pseudonymised to sequential ids (E0001...) in the
released CSV; only the derived features above survive. Article content is
public encyclopedic infobox text (CC BY-SA 4.0); the page list and items ship
with the artifact so a rerun does not depend on live Wikipedia.

Usage: PYTHONPATH=. python scripts/build_wikidivergence.py [--pages-per-cat 15]
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_ITEMS = ROOT / "data" / "wikidivergence_items.csv"
OUT_PAGES = ROOT / "data" / "wikidivergence_pages.csv"
CACHE = ROOT / ".wiki_cache"

API = "https://en.wikipedia.org/w/api.php"
UA = "BranchMem-research/1.0 (academic study of memory reconciliation; serial requests)"

# Four pre-declared categories spanning organisations, places and sports clubs.
# Biographies are excluded on purpose: living-person articles raise privacy
# questions the study does not need.
CATEGORIES = [
    "Category:Capital cities in Europe",
    "Category:Premier League clubs",
    "Category:Airlines of the United States",
    "Category:Companies in the S&P 500 Dividend Aristocrats",
]
REV_WINDOW = 300        # revisions of history examined per page (newest first)
BATCH = 50              # revisions whose section-0 wikitext we fetch per request
THROTTLE = 0.6          # seconds between API requests
SEED = 2026             # page-sample seed
MAX_PER_FIELD = 3       # items drawn from one (page, field) chain
MAX_PER_PAGE = 10       # items drawn from one article
MAX_VALUE_CHARS = 60

_SKIP_FIELD = re.compile(
    r"logo|image|caption|alt$|^alt|photo|map|pushpin|footnote|ref|url|website|"
    r"colou?r|width|size_|_size|module|embed|align|style|border|label|flag|seal|"
    r"coordinates|latd|longd|blank|list|signature|nickname|"
    r"pattern|shorts|socks|leftarm|rightarm|^body|^kit|shirt|sponsor_",
    re.I,
)
_CHANNELS = (
    ("leadership", r"leader|manager|ceo|chairman|chairperson|president|owner|"
                   r"governor|mayor|director|founder|head_|_head|coach"),
    ("quantity",   r"population|employees|revenue|capacity|fleet|area|elevation|"
                   r"num_|_total|_num|assets|income|seats|destinations|members"),
    ("location",   r"hq|headquarters|location|ground|stadium|city|country|region|"
                   r"district|hubs|base|county|state|subdivision"),
    ("identity",   r"name|league|type|industry|currency|language|timezone|"
                   r"iata|icao|callsign|motto|founded|established|website_name"),
)


def _request(params: dict) -> dict:
    params = dict(params)
    params.update({"format": "json", "formatversion": "2", "maxlag": "5"})
    url = API + "?" + urllib.parse.urlencode(params)
    key = hashlib.sha256(url.encode()).hexdigest()
    CACHE.mkdir(exist_ok=True)
    path = CACHE / f"{key}.json.gz"
    if path.exists():
        with gzip.open(path, "rt") as f:
            return json.load(f)
    for attempt in range(6):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": UA, "Api-User-Agent": UA, "Accept-Encoding": "gzip"}
            )
            with urllib.request.urlopen(req, timeout=90) as resp:
                raw = resp.read()
                if resp.headers.get("Content-Encoding") == "gzip":
                    raw = gzip.decompress(raw)
            data = json.loads(raw.decode())
            if "error" in data and data["error"].get("code") == "maxlag":
                time.sleep(5 * (attempt + 1))
                continue
            with gzip.open(path, "wt") as f:
                json.dump(data, f)
            time.sleep(THROTTLE)
            return data
        except urllib.error.HTTPError as exc:
            if attempt == 5:
                raise
            wait = int(exc.headers.get("Retry-After", 0) or 0) or 8 * (attempt + 1)
            print(f"  HTTP {exc.code}; sleeping {wait}s")
            time.sleep(wait)
        except Exception as exc:
            if attempt == 5:
                raise
            print(f"  retry {attempt+1} after {type(exc).__name__}")
            time.sleep(3 * (attempt + 1))
    raise RuntimeError("unreachable")


# ---------------------------------------------------------------- infobox parse
def parse_infobox(wikitext: str) -> dict[str, str]:
    """Extract top-level `| field = value` pairs from the first infobox."""
    m = re.search(r"\{\{\s*Infobox", wikitext, re.I)
    if not m:
        return {}
    i, depth, start = m.start(), 0, None
    while i < len(wikitext):
        if wikitext.startswith("{{", i):
            depth += 1
            if start is None:
                start = i + 2
            i += 2
            continue
        if wikitext.startswith("}}", i):
            depth -= 1
            if depth == 0:
                body = wikitext[start:i]
                break
            i += 2
            continue
        i += 1
    else:
        return {}

    fields, depth_t, depth_b, buf = {}, 0, 0, []
    for ch_i, ch in enumerate(body):
        two = body[ch_i:ch_i + 2]
        if two == "{{":
            depth_t += 1
        elif two == "}}":
            depth_t = max(0, depth_t - 1)
        elif two == "[[":
            depth_b += 1
        elif two == "]]":
            depth_b = max(0, depth_b - 1)
        if ch == "|" and depth_t == 0 and depth_b == 0:
            buf.append("\x00")
        else:
            buf.append(ch)
    for chunk in "".join(buf).split("\x00")[1:]:
        if "=" not in chunk:
            continue
        name, _, value = chunk.partition("=")
        fields[name.strip().lower()] = value.strip()
    return fields


def normalise(value: str) -> str:
    """Strip markup so two spellings of the same value compare equal.

    Returns "" for anything still carrying markup after stripping, so a value
    mangled by template removal never becomes a benchmark item.
    """
    v = re.sub(r"<ref[^>]*>.*?</ref>", " ", value, flags=re.S | re.I)
    v = re.sub(r"<ref[^>]*/>", " ", v, flags=re.I)
    v = re.sub(r"<!--.*?-->", " ", v, flags=re.S)
    for _ in range(6):                        # templates and links nest
        nxt = re.sub(r"\{\{[^{}]*\}\}", " ", v)
        nxt = re.sub(r"\[\[([^\[\]|]*\|)?([^\[\]]*)\]\]", r"\2", nxt)
        if nxt == v:
            break
        v = nxt
    v = re.sub(r"<[^>]+>", " ", v)
    v = v.replace("&nbsp;", " ").replace("'''", "").replace("''", "")
    v = re.sub(r"\s+", " ", v).strip(" .,;|")
    if re.search(r"[{}|<>=\[\]]", v):
        return ""
    if len(v) < 2 or not re.search(r"[A-Za-z0-9]", v):
        return ""
    # A value whose substance sat inside a stripped template survives only as a
    # parenthetical ("$4.079 billion (2023)" -> "(2023)"); reject those.
    if len(re.sub(r"\([^)]*\)", "", v).strip(" .,;")) < 2:
        return ""
    return v


_ABBREV = {"u.s": "united states", "us": "united states", "u.s.a": "united states",
           "usa": "united states", "uk": "united kingdom", "st": "saint",
           "&": "and", "inc": "incorporated", "corp": "corporation",
           "co": "company", "ltd": "limited", "fc": "football club"}


def _tokens(value: str) -> set[str]:
    out: set[str] = set()
    for tok in re.split(r"[^A-Za-z0-9.&]+", value.lower()):
        tok = tok.strip(".")
        if not tok:
            continue
        out.update(_ABBREV.get(tok, tok).split())
    return out


def is_formatting_variant(v1: str, v2: str) -> bool:
    """True when two values differ only in abbreviation or verbosity.

    One token set contained in the other ("Public" vs "Public company", "3M"
    vs "3M Company", "U.S" vs "United States") is a presentation difference
    rather than a disagreement about the world, so it is not a semantic
    conflict and is excluded.
    """
    t1, t2 = _tokens(v1), _tokens(v2)
    if not t1 or not t2:
        return True
    return t1 <= t2 or t2 <= t1


def channel_for(field: str) -> str:
    for name, pattern in _CHANNELS:
        if re.search(pattern, field, re.I):
            return name
    return "other"


def confidence_for(editcount: int) -> float:
    return round(min(0.98, max(0.50, 0.50 + 0.08 * math.log10(1 + max(0, editcount)))), 3)


def source_for(anon: bool, groups: list[str]) -> str:
    if anon:
        return "inference"
    trusted = {"extendedconfirmed", "sysop", "autoreviewer", "rollbacker", "patroller"}
    return "user" if trusted & set(groups or []) else "observation"


# ------------------------------------------------------------------ API helpers
def category_pages(category: str, limit: int, seed: int = SEED) -> list[str]:
    """A deterministic sample of the category's articles.

    Ordering by a seeded hash of the title rather than alphabetically avoids
    the bias of taking whatever sorts first, which in these categories means
    stub articles with almost no revision history.
    """
    data = _request({"action": "query", "list": "categorymembers", "cmtitle": category,
                     "cmlimit": "500", "cmnamespace": "0", "cmtype": "page"})
    titles = sorted(m["title"] for m in data["query"]["categorymembers"])
    titles.sort(key=lambda t: hashlib.sha1(f"{seed}:{t}".encode()).hexdigest())
    return titles[:limit]


def revisions(title: str) -> list[dict]:
    out, cont = [], None
    while len(out) < REV_WINDOW:
        params = {"action": "query", "prop": "revisions", "titles": title,
                  "rvlimit": "500", "rvprop": "ids|timestamp|user|comment|flags|tags"}
        if cont:
            params["rvcontinue"] = cont
        data = _request(params)
        pages = data.get("query", {}).get("pages", [])
        if not pages or "revisions" not in pages[0]:
            break
        out.extend(pages[0]["revisions"])
        cont = data.get("continue", {}).get("rvcontinue")
        if not cont:
            break
    return out[:REV_WINDOW]


def section0_batch(revids: list[int]) -> dict[int, str]:
    """Section-0 wikitext for up to BATCH revisions per request."""
    out: dict[int, str] = {}
    for i in range(0, len(revids), BATCH):
        chunk = revids[i:i + BATCH]
        data = _request({"action": "query", "prop": "revisions",
                         "revids": "|".join(str(r) for r in chunk),
                         "rvprop": "content|ids", "rvslots": "main", "rvsection": "0"})
        for page in data.get("query", {}).get("pages", []):
            for rev in page.get("revisions", []):
                out[rev["revid"]] = rev.get("slots", {}).get("main", {}).get("content", "") or ""
    return out


def user_info(names: list[str]) -> dict[str, dict]:
    info: dict[str, dict] = {}
    names = [n for n in dict.fromkeys(names) if n]
    for i in range(0, len(names), 40):
        chunk = names[i:i + 40]
        data = _request({"action": "query", "list": "users", "ususers": "|".join(chunk),
                         "usprop": "editcount|groups"})
        for u in data.get("query", {}).get("users", []):
            info[u["name"]] = {
                "editcount": u.get("editcount", 0),
                "groups": u.get("groups", []),
                "anon": "invalid" in u or "missing" in u,
            }
    for n in names:
        info.setdefault(n, {"editcount": 0, "groups": [], "anon": True})
    return info


# ------------------------------------------------------------------- extraction
def field_changes(revs: list[dict]) -> tuple[list[dict], dict[str, str]]:
    """Every infobox field change in the window, attributed to its exact revision.

    Returns the changes plus the newest revision's parsed infobox (the labels).
    """
    revs = list(reversed(revs))                      # oldest -> newest
    text = section0_batch([r["revid"] for r in revs])
    boxes = [{k: normalise(v) for k, v in parse_infobox(text.get(r["revid"], "")).items()}
             for r in revs]
    changes: list[dict] = []
    for i in range(1, len(revs)):
        before, after = boxes[i - 1], boxes[i]
        if not before or not after:
            continue
        for field in set(before) | set(after):
            if _SKIP_FIELD.search(field):
                continue
            v0, v1 = before.get(field, ""), after.get(field, "")
            if v0 == v1 or not v0 or not v1:
                continue
            if len(v0) > MAX_VALUE_CHARS or len(v1) > MAX_VALUE_CHARS:
                continue
            if not revs[i].get("user"):
                continue
            changes.append({"field": field, "rev_index": i, "old": v0, "new": v1,
                            "rev": revs[i]})
    return changes, (boxes[-1] if boxes else {})


def _days(a: str, b: str) -> float:
    from datetime import datetime
    fmt = "%Y-%m-%dT%H:%M:%SZ"
    return abs((datetime.strptime(b, fmt) - datetime.strptime(a, fmt)).total_seconds()) / 86400.0


def build_items(title: str) -> list[dict]:
    revs = revisions(title)
    if len(revs) < 20:
        return []
    changes, newest_box = field_changes(revs)
    by_field: dict[str, list[dict]] = {}
    for c in changes:
        by_field.setdefault(c["field"], []).append(c)

    items: list[dict] = []
    for field, seq in by_field.items():
        seq.sort(key=lambda c: c["rev_index"])
        current = newest_box.get(field, "")
        if not current:
            continue
        for c1, c2 in zip(seq, seq[1:]):
            if c1["rev"]["user"] == c2["rev"]["user"]:
                continue                       # same editor: not a divergence
            v0, v1, v2 = c1["old"], c1["new"], c2["new"]
            if v1 == v2 or not v0 or v1 != c2["old"]:
                continue                       # c2 must act on what c1 wrote
            if current not in (v1, v2):
                continue                       # no determinate label
            if is_formatting_variant(v1, v2):
                continue                       # presentation, not disagreement
            items.append({
                "page": title,
                "field": field,
                "channel": channel_for(field),
                "ancestor_value": v0,
                "branch_a_value": v1,
                "branch_b_value": v2,
                "label": "branch_a" if current == v1 else "branch_b",
                "is_revert_to_ancestor": v2 == v0,
                "n_intervening_revisions": c2["rev_index"] - c1["rev_index"],
                "divergence_days": round(_days(c1["rev"]["timestamp"], c2["rev"]["timestamp"]), 3),
                "a_user": c1["rev"]["user"],
                "a_anon": bool(c1["rev"].get("anon")),
                "a_timestamp": c1["rev"]["timestamp"],
                "a_cites_source": bool(re.search(r"<ref|cite|source", c1["rev"].get("comment", ""), re.I)),
                "b_user": c2["rev"]["user"],
                "b_anon": bool(c2["rev"].get("anon")),
                "b_timestamp": c2["rev"]["timestamp"],
                "b_cites_source": bool(re.search(r"<ref|cite|source", c2["rev"].get("comment", ""), re.I)),
            })
    per_field: dict[str, int] = {}
    kept: list[dict] = []
    for it in sorted(items, key=lambda r: (r["field"], r["a_timestamp"])):
        if per_field.get(it["field"], 0) >= MAX_PER_FIELD or len(kept) >= MAX_PER_PAGE:
            continue
        per_field[it["field"]] = per_field.get(it["field"], 0) + 1
        kept.append(it)
    return kept


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages-per-cat", type=int, default=15)
    args = ap.parse_args()

    pages: list[tuple[str, str]] = []
    for cat in CATEGORIES:
        for t in category_pages(cat, args.pages_per_cat):
            pages.append((cat, t))
    print(f"{len(pages)} pages from {len(CATEGORIES)} categories")

    raw: list[dict] = []
    for i, (cat, title) in enumerate(pages, 1):
        try:
            got = build_items(title)
        except Exception as exc:
            print(f"[{i}/{len(pages)}] {title}: FAILED {type(exc).__name__}: {exc}")
            continue
        for it in got:
            it["category"] = cat
        raw.extend(got)
        print(f"[{i}/{len(pages)}] {title}: {len(got)} conflicts (total {len(raw)})")

    names = [n for it in raw for n in (it["a_user"], it["b_user"])
             if not (it["a_anon"] if n == it["a_user"] else it["b_anon"])]
    info = user_info(names)

    pseudo: dict[str, str] = {}
    rows: list[dict] = []
    for it in sorted(raw, key=lambda r: (r["page"], r["field"], r["a_timestamp"])):
        out = dict(it)
        for side in ("a", "b"):
            user, anon = it[f"{side}_user"], it[f"{side}_anon"]
            meta = info.get(user, {"editcount": 0, "groups": []})
            ec = 0 if anon else int(meta.get("editcount", 0))
            out[f"{side}_source"] = source_for(anon, meta.get("groups", []))
            out[f"{side}_confidence"] = confidence_for(ec)
            out[f"{side}_editcount"] = ec
            out[f"{side}_editor"] = pseudo.setdefault(user, f"E{len(pseudo)+1:04d}")
            del out[f"{side}_user"]
        out["conflict_id"] = "wd_" + hashlib.sha1(
            f"{it['page']}|{it['field']}|{it['a_timestamp']}".encode()
        ).hexdigest()[:12]
        rows.append(out)

    import pandas as pd
    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset="conflict_id")
    cols = ["conflict_id", "page", "category", "field", "channel", "ancestor_value",
            "branch_a_value", "branch_b_value", "label", "is_revert_to_ancestor",
            "n_intervening_revisions", "divergence_days",
            "a_editor", "a_source", "a_confidence", "a_editcount", "a_anon",
            "a_cites_source", "a_timestamp",
            "b_editor", "b_source", "b_confidence", "b_editcount", "b_anon",
            "b_cites_source", "b_timestamp"]
    df[cols].to_csv(OUT_ITEMS, index=False)
    pd.DataFrame(pages, columns=["category", "page"]).to_csv(OUT_PAGES, index=False)
    print(f"\nwrote {len(df)} conflicts to {OUT_ITEMS}")
    print("\nby channel:\n", df.channel.value_counts().to_string())
    print("\nlabel:\n", df.label.value_counts().to_string())
    print("\nreverts to ancestor:", int(df.is_revert_to_ancestor.sum()))
    print("\nsource (A/B):\n", df.a_source.value_counts().to_string(),
          "\n", df.b_source.value_counts().to_string())


if __name__ == "__main__":
    main()
