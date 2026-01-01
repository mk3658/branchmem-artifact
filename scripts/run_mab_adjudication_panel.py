#!/usr/bin/env python3
"""Blinded independent adjudication of all 270 MAB-derived conflicts.

The released protocol (`annotation/external_validation_protocol.md`) specifies
three independent human annotators. We cannot run that here. This script runs
the SAME protocol with a panel of model adjudicators, which is a weaker
substitute and is reported as one: it tests whether an independent judge
looking only at the MemoryAgentBench evidence reaches the same resolution as
the parser-derived label, but it does not discharge the human study.

What the protocol contributes, and what is preserved here:
  * blinded presentation: no source, no confidence, no model output, no other
    adjudicator's answer;
  * independently randomised left/right order per adjudicator, with the mapping
    held in a private key;
  * one of three outcomes per item, plus a confidence and an evidence span;
  * family-stratified reporting over all eight source rows.

Panel members. Three are OpenAI models at different capability tiers, so their
errors are vendor-correlated and their agreement among themselves overstates
independence. The other three were added precisely because they do not share
that lineage: an open-weight instruction model run locally (Llama 3.1 8B via
Ollama, Meta weights, no API), and two entailment models from different
pretraining families and different NLI finetunes (DeBERTa-v3 and RoBERTa).
Those three form a VENDOR-INDEPENDENT sub-panel of the size the released human
protocol asks for, and the analysis reports it separately from the full panel.

Usage: OPENAI_API_KEY=... PYTHONPATH=. python scripts/run_mab_adjudication_panel.py [--limit N]
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import pandas as pd

from branchmem.llm.base import build_backend

ROOT = Path(__file__).resolve().parents[1]
SHEETS = Path(os.environ.get("MAB_SHEETS", ROOT / "annotation" / "mab_sheets"))
OUT = ROOT / "results" / "final"

# adjudicator id -> panel member
PANEL = {
    "ann1": {"kind": "llm", "model": "gpt-5.4", "vendor": "openai"},
    "ann2": {"kind": "llm", "model": "gpt-5.4-mini", "vendor": "openai"},
    "ann3": {"kind": "llm", "model": "gpt-5.4-nano", "vendor": "openai"},
    "ann4": {"kind": "nli", "model": "cross-encoder/nli-deberta-v3-small",
             "vendor": "local-deberta"},
    "ann5": {"kind": "llm", "model": "llama3.1:8b", "vendor": "local-llama",
             "base_url": "http://localhost:11434/v1"},
    "ann6": {"kind": "nli", "model": "roberta-large-mnli", "vendor": "local-roberta"},
}
# Adjudicators sharing no weights, no training data and no vendor with the
# OpenAI tier. The analysis reports this sub-panel on its own.
INDEPENDENT = ("ann4", "ann5", "ann6")

_SYSTEM = (
    "You are adjudicating which of two candidate values is correct for a fact, using only "
    "the numbered evidence lines you are given. The evidence is a numbered log in which "
    "LATER lines supersede EARLIER lines about the same subject and relation. Decide which "
    "candidate the evidence supports as the current value. You are not told where either "
    "candidate came from, and you must not guess from the candidates' plausibility as world "
    "knowledge: the log may contain counterfactual statements, and the log is the only "
    "authority. If the evidence genuinely does not determine an answer, say so rather than "
    "guessing."
)

_TEMPLATE = """Entity: {entity}
Relation: {predicate}
Previously recorded value (may be stale, may be empty): {ancestor}

Candidate LEFT:  {left}
Candidate RIGHT: {right}

Evidence lines (numbered, in original order):
{evidence}

Respond with strict JSON only, no other text:
{{"resolution": "left" | "right" | "ambiguous",
  "confidence_1_5": <integer 1-5>,
  "evidence_span": "<the single line number you relied on, or empty>"}}
"""


PARSE_FAILURES: dict[str, int] = {}


def _parse(text: str, who: str = "") -> dict:
    """Parse one adjudication. A malformed reply counts as an abstention, and
    is also tallied so an adjudicator's abstentions are never silently inflated
    by its own formatting failures."""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        PARSE_FAILURES[who] = PARSE_FAILURES.get(who, 0) + 1
        return {"resolution": "ambiguous", "confidence_1_5": 1, "evidence_span": ""}
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        PARSE_FAILURES[who] = PARSE_FAILURES.get(who, 0) + 1
        return {"resolution": "ambiguous", "confidence_1_5": 1, "evidence_span": ""}
    res = str(obj.get("resolution", "ambiguous")).lower()
    if res not in {"left", "right", "ambiguous"}:
        res = "ambiguous"
    return {
        "resolution": res,
        "confidence_1_5": obj.get("confidence_1_5", 3),
        "evidence_span": str(obj.get("evidence_span", ""))[:40],
    }


def llm_adjudicate(backend, row, who: str = "") -> dict:
    prompt = _TEMPLATE.format(
        entity=row.entity,
        predicate=row.predicate,
        ancestor=row.ancestor_value or "(none recorded)",
        left=row.left_value,
        right=row.right_value,
        evidence=row.evidence_context.replace(" || ", "\n"),
    )
    return _parse(backend.complete(prompt, system=_SYSTEM).text, who)


class NliAdjudicator:
    """Local entailment adjudicator: which candidate does the evidence entail?

    Deliberately simple and deliberately not an LLM. It scores each candidate
    by the entailment probability of 'entity predicate candidate' given the
    evidence, and abstains when the two scores are close.
    """

    MARGIN = 0.10

    def __init__(self, name: str = "cross-encoder/nli-deberta-v3-small") -> None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        import torch

        self.torch = torch
        self.name = name
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForSequenceClassification.from_pretrained(name).eval()
        labels = self.model.config.id2label
        self.entail_idx = next(i for i, l in labels.items() if "entail" in l.lower())

    def _entailment(self, premise: str, hypothesis: str) -> float:
        enc = self.tok(premise, hypothesis, return_tensors="pt", truncation=True, max_length=512)
        with self.torch.no_grad():
            probs = self.model(**enc).logits.softmax(-1)[0]
        return float(probs[self.entail_idx])

    def adjudicate(self, row) -> dict:
        premise = row.evidence_context.replace(" || ", " ")
        rel = str(row.predicate).replace("_", " ")
        left = self._entailment(premise, f"{row.entity} {rel} {row.left_value}.")
        right = self._entailment(premise, f"{row.entity} {rel} {row.right_value}.")
        if abs(left - right) < self.MARGIN:
            res = "ambiguous"
        else:
            res = "left" if left > right else "right"
        return {
            "resolution": res,
            "confidence_1_5": min(5, max(1, int(round(1 + 4 * abs(left - right))))),
            "evidence_span": "",
        }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="adjudicate only the first N items")
    ap.add_argument("--panel", default="", help="comma-separated adjudicator ids (default: all)")
    ap.add_argument("--out", default="mab_adjudication_responses.csv")
    ap.add_argument("--only", default="", help="CSV with a conflict_id column: adjudicate only those")
    args = ap.parse_args()

    key = pd.read_csv(SHEETS / "PRIVATE_presentation_key.csv")
    records = []
    nli_cache: dict[str, NliAdjudicator] = {}

    members = PANEL if not args.panel else {k: PANEL[k] for k in args.panel.split(',')}
    for ann, spec in members.items():
        sheet = pd.read_csv(SHEETS / f"annotator_{ann[-1]}_sheet.csv", keep_default_na=False)
        if args.limit:
            sheet = sheet.head(args.limit)
        if args.only:
            keep = set(pd.read_csv(args.only).conflict_id)
            sheet = sheet[sheet.conflict_id.isin(keep)]
        if spec["kind"] == "llm":
            backend = build_backend(
                {
                    "backend": "openai_compatible",
                    "model": spec["model"],
                    "base_url": spec.get("base_url"),
                    "cache_dir": "llm_cache",
                    "temperature": 0.0,
                    "max_tokens": 2000,
                }
            )
            who = spec["model"]
            judge = lambda r, b=backend, w=who: llm_adjudicate(b, r, w)  # noqa: E731
        else:
            nli_cache.setdefault(spec["model"], NliAdjudicator(spec["model"]))
            judge = nli_cache[spec["model"]].adjudicate
            who = f"local NLI ({spec['model']})"

        print(f"adjudicating with {ann} = {who} ({len(sheet)} items)")
        for _, r in sheet.iterrows():
            ans = judge(r)
            records.append(
                {
                    "annotator_id": ann,
                    "adjudicator": who,
                    "vendor": spec.get("vendor", "unknown"),
                    "conflict_id": r.conflict_id,
                    "presentation_id": int(r.presentation_id),
                    **ans,
                }
            )

    responses = pd.DataFrame(records)
    # De-blind: map left/right back to branch A/B through the private key.
    merged = responses.merge(key, on=["annotator_id", "presentation_id", "conflict_id"])
    merged["branch_choice"] = [
        "ambiguous"
        if res == "ambiguous"
        else ("branch_a" if (lo == "A") == (res == "left") else "branch_b")
        for res, lo in zip(merged.resolution, merged.left_original)
    ]
    merged.to_csv(OUT / args.out, index=False)
    print(f"\nwrote {len(merged)} adjudications to {OUT / args.out}")
    if PARSE_FAILURES:
        print("malformed replies counted as abstentions:", PARSE_FAILURES)
    else:
        print("no malformed replies")


if __name__ == "__main__":
    main()
