#!/usr/bin/env python3
"""Can a merger detect WHEN the standing-rule distinction is at stake?

Post-hoc, exploratory; added in revision. This is the detection problem the
rest of the paper leaves open, run on `data/hard_standing_bank.json`
(BranchMem-Hard), where it is answerable because the lexical shortcut has been
removed: all four obligation types carry an obligation-bearing ancestor and
only one of them makes rejecting the update correct.

Five policies, each scored on BOTH error directions, which is the point:

  ModalLexicon        zero cost. Fires on the ancestor's obligation wording.
                      On the old discrimination bank this is a perfect
                      detector by construction; here it should freeze most
                      legitimate updates.
  LLMStandingCheck    the general prompt a deployment could actually write,
                      applied to EVERY key: "does the stored entry carry
                      standing this update cannot override?" No categories,
                      no examples, no marking.
  TrainedTransfer     TF-IDF or MiniLM + logistic regression, trained on the
                      120 ancestors of the earlier strict/defeasible banks
                      (60 standing, 60 ordinary) and applied here with no
                      adaptation. Cheap and API-free.
  TrainedLORO         the same classifier under leave-one-family-out CV
                      WITHIN this bank: an upper bound on what a small
                      classifier can learn from 60 in-domain items.
  ConfidenceRule      what a deployed merger does today: follow the metadata.

Input scope is a variable, not a detail. Expiry, unmet conditions and waivers
are stated in the UPDATE, not the ancestor, so a detector shown only the
stored entry cannot separate the four obligation types even in principle. Every
learned and prompted policy is therefore run twice, on `ancestor` alone and on
`ancestor+update`, and the gap between them is itself a result.

Metrics, reported together:
  violations_caught  recall on standing_indefinite -- the harm the paper warns
                     about, prevented.
  freezing_rate      share of the 48 update-correct items wrongly kept -- the
                     cost the paper previously only noted.
  balanced           mean of the two, so neither can be bought with the other.

Usage:
  OPENAI_API_KEY=... PYTHONPATH=.:scripts python scripts/run_standing_detection.py \
      --llm-models gpt-5.4,gpt-5.4-nano --out standing_detection.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BANK = ROOT / "data" / "hard_standing_bank.json"
STRICT = ROOT / "data" / "realistic_semantic_templates.json"
DEFEASIBLE = ROOT / "data" / "defeasible_ancestor_templates.json"
OUT_DIR = ROOT / "results" / "final"
SEED = 4242

LOCAL_BASE_URL = "http://localhost:11434/v1"
OPEN_WEIGHT = {"llama3.1:8b", "qwen2.5:7b", "mistral:7b", "deepseek-r1:8b", "qwen3:8b"}

MARKER = re.compile(
    r"\bonly\b|\bmust\b|\bcannot\b|\bnot permitted\b|\bnot allowed\b|\brequire[sd]?\b|"
    r"\bmaximum\b|\blimit\b|\bprohibit|\bforbid|\brule\b|\bpolicy\b|\bcontract|"
    r"\bban(s|ned)?\b|\bmandat|\bobliged\b|\bcapacity\b|\bdeadline\b|\bno more than\b",
    re.I)

_SYSTEM = (
    "You decide whether a stored memory entry still binds. You are given the stored "
    "entry and a proposed update to the same key. Decide whether the stored entry "
    "carries standing that the proposed update cannot override -- so the update must "
    "be rejected -- or whether the update may legitimately replace it. Consider only "
    "the content you are shown. Answer with exactly one word: BINDING or SUPERSEDABLE."
)
_TMPL_BOTH = "Stored entry: {ancestor}\nProposed update: {update}\n\nOne word:"
_TMPL_ANC = "Stored entry: {ancestor}\n\nOne word:"


def load_bank() -> list[dict]:
    digest = hashlib.sha256(BANK.read_bytes()).hexdigest()
    pinned = (BANK.parent / "hard_standing_bank.sha256").read_text().split()[0]
    if digest != pinned:
        raise SystemExit(f"hard bank hash mismatch\n  disk {digest}\n  pin  {pinned}")
    payload = json.loads(BANK.read_text())
    return [it for f in payload["families"] for it in f["items"]]


def load_training() -> tuple[list[str], list[int]]:
    """120 ancestors from the EARLIER banks: strict -> standing, defeasible -> ordinary.

    These two files share predicates and differ only in the ancestor's modal
    force, so a classifier trained here learns the modal cue. That is the
    honest point of the transfer condition, not a flaw in it.
    """
    x, y = [], []
    for path, label in ((STRICT, 1), (DEFEASIBLE, 0)):
        for tpl in json.loads(path.read_text())["templates"]:
            x.append(tpl["ancestor_value"])
            y.append(label)
    return x, y


# ---------------------------------------------------------------- detectors
def predict_lexicon(items: list[dict], scope: str) -> list[int]:
    return [int(bool(MARKER.search(it["ancestor_value"]))) for it in items]


def _vectorisers():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    return {
        "tfidf_lr": lambda: make_pipeline(
            TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1,
                            sublinear_tf=True),
            LogisticRegression(max_iter=2000, random_state=SEED)),
    }


def _embed(texts: list[str]):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model.encode(texts, show_progress_bar=False, normalize_embeddings=True)


def _text_of(it: dict, scope: str) -> str:
    return (it["ancestor_value"] if scope == "ancestor"
            else f"{it['ancestor_value']} || {it['update_value']}")


def predict_trained_transfer(items, scope, kind):
    """Trained on the earlier banks, applied here with no adaptation.

    The training ancestors have no paired update, so when scope is
    ancestor+update the training text is the ancestor alone; that asymmetry is
    reported rather than hidden, and is why the transfer row is weak.
    """
    xtr, ytr = load_training()
    xte = [_text_of(it, scope) for it in items]
    if kind == "embed_lr":
        from sklearn.linear_model import LogisticRegression
        import numpy as np
        emb = _embed(xtr + xte)
        clf = LogisticRegression(max_iter=2000, random_state=SEED)
        clf.fit(np.asarray(emb[:len(xtr)]), ytr)
        return [int(v) for v in clf.predict(np.asarray(emb[len(xtr):]))]
    pipe = _vectorisers()["tfidf_lr"]()
    pipe.fit(xtr, ytr)
    return [int(v) for v in pipe.predict(xte)]


def predict_trained_loro(items, scope, kind):
    """Leave-one-family-out CV inside this bank: the learnability upper bound."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    texts = [_text_of(it, scope) for it in items]
    y = [int(it["correct_action"] == "keep") for it in items]
    fams = [it["family"] for it in items]
    pred = [0] * len(items)
    if kind == "embed_lr":
        emb = np.asarray(_embed(texts))
    for held in sorted(set(fams)):
        tr = [i for i, f in enumerate(fams) if f != held]
        te = [i for i, f in enumerate(fams) if f == held]
        if len(set(y[i] for i in tr)) < 2:
            continue
        if kind == "embed_lr":
            clf = LogisticRegression(max_iter=2000, random_state=SEED)
            clf.fit(emb[tr], [y[i] for i in tr])
            out = clf.predict(emb[te])
        else:
            pipe = _vectorisers()["tfidf_lr"]()
            pipe.fit([texts[i] for i in tr], [y[i] for i in tr])
            out = pipe.predict([texts[i] for i in te])
        for i, v in zip(te, out):
            pred[i] = int(v)
    return pred


def predict_llm(items, scope, model):
    from branchmem.llm.base import build_backend
    cfg = {"backend": "openai_compatible", "model": model, "cache_dir": "llm_cache",
           "temperature": 0.0, "max_tokens": 4000}
    if model in OPEN_WEIGHT:
        cfg["base_url"] = LOCAL_BASE_URL
    be = build_backend(cfg)
    tmpl = _TMPL_ANC if scope == "ancestor" else _TMPL_BOTH
    out, unparsed = [], 0
    for it in items:
        txt = be.complete(
            tmpl.format(ancestor=it["ancestor_value"], update=it["update_value"]),
            system=_SYSTEM).text
        low = re.sub(r"<think\b[^>]*>.*?</think\s*>", "", txt, flags=re.S | re.I).lower()
        if "supersedable" in low:
            out.append(0)
        elif "binding" in low:
            out.append(1)
        else:
            out.append(0)
            unparsed += 1
    return out, unparsed


# ---------------------------------------------------------------- scoring
def score(items: list[dict], pred_keep: list[int], n_boot: int = 2000) -> dict:
    """pred_keep[i] == 1 means the policy rejects the update for item i."""
    truth = [int(it["correct_action"] == "keep") for it in items]
    types = [it["type"] for it in items]
    fams = [it["family"] for it in items]

    viol = [(p, t) for p, t, ty in zip(pred_keep, truth, types) if t == 1]
    upd = [(p, t) for p, t in zip(pred_keep, truth) if t == 0]
    caught = sum(p for p, _ in viol) / len(viol)
    frozen = sum(p for p, _ in upd) / len(upd)

    by_type = {}
    for ty in sorted(set(types)):
        idx = [i for i, t in enumerate(types) if t == ty]
        by_type[ty] = round(
            sum(int(pred_keep[i] == truth[i]) for i in idx) / len(idx), 4)

    # family-clustered interval on the balanced score
    by_fam = defaultdict(list)
    for f, p, t in zip(fams, pred_keep, truth):
        by_fam[f].append((p, t))
    names = sorted(by_fam)
    rng = random.Random(SEED)

    def bal_of(sample):
        v = [(p, t) for f in sample for p, t in by_fam[f] if t == 1]
        u = [(p, t) for f in sample for p, t in by_fam[f] if t == 0]
        if not v or not u:
            return None
        return (sum(p for p, _ in v) / len(v) + 1 - sum(p for p, _ in u) / len(u)) / 2

    boots = sorted(b for b in (bal_of([rng.choice(names) for _ in names])
                               for _ in range(n_boot)) if b is not None)
    ci = [round(boots[int(.025 * len(boots))], 4),
          round(boots[int(.975 * len(boots))], 4)] if boots else None
    return {
        "n": len(items),
        "violations_caught": round(caught, 4),
        "freezing_rate": round(frozen, 4),
        "balanced": round((caught + 1 - frozen) / 2, 4),
        "balanced_ci95_family_clustered": ci,
        "accuracy": round(sum(int(p == t) for p, t in zip(pred_keep, truth)) / len(items), 4),
        "accuracy_by_type": by_type,
        "n_rejected": sum(pred_keep),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm-models", default="gpt-5.4,gpt-5.4-nano")
    ap.add_argument("--out", default="standing_detection.json")
    args = ap.parse_args()

    items = load_bank()
    print(f"{len(items)} items, {len({i['family'] for i in items})} families")
    results: dict[str, dict] = {}

    for scope in ("ancestor", "ancestor_update"):
        results[scope] = {}
        results[scope]["ModalLexicon"] = score(items, predict_lexicon(items, scope))
        for kind in ("tfidf_lr", "embed_lr"):
            results[scope][f"TrainedTransfer ({kind})"] = score(
                items, predict_trained_transfer(items, scope, kind))
            results[scope][f"TrainedLORO ({kind})"] = score(
                items, predict_trained_loro(items, scope, kind))
        for model in [m for m in args.llm_models.split(",") if m]:
            pred, unparsed = predict_llm(items, scope, model)
            e = score(items, pred)
            e["n_unparsed_replies"] = unparsed
            results[scope][f"LLMStandingCheck ({model})"] = e

        print(f"\n--- scope: {scope}")
        print(f"{'policy':34} {'caught':>7} {'frozen':>7} {'bal':>6} {'bal 95% CI':>16}")
        for name, e in results[scope].items():
            print(f"{name:34} {e['violations_caught']:7.3f} {e['freezing_rate']:7.3f} "
                  f"{e['balanced']:6.3f} {str(e['balanced_ci95_family_clustered']):>16}")

    payload = {
        "note": (
            "POST-HOC, EXPLORATORY (added in revision; in neither preregistration). The "
            "detection problem on BranchMem-Hard, where all four obligation types carry "
            "an obligation-bearing ancestor and only standing_indefinite makes rejecting "
            "the update correct, so no lexical shortcut recovers the answer. Both error "
            "directions are reported: violations_caught is recall on the type that must "
            "be rejected, freezing_rate is the share of legitimate updates wrongly "
            "rejected. 'ancestor' scope shows only the stored entry; 'ancestor_update' "
            "also shows the proposed update, which is where expiry, unmet conditions and "
            "waivers are stated. TrainedTransfer is trained on the 120 ancestors of the "
            "earlier strict/defeasible banks, whose two halves differ only in modal "
            "force, so it learns that cue; TrainedLORO is leave-one-family-out inside "
            "this bank and is the learnability upper bound. Intervals resample families."
        ),
        "bank": str(BANK.relative_to(ROOT)),
        "bank_sha256": hashlib.sha256(BANK.read_bytes()).hexdigest(),
        "seed": SEED,
        "n_items": len(items),
        "detection_prompt": {"system": _SYSTEM, "ancestor_only": _TMPL_ANC,
                             "ancestor_and_update": _TMPL_BOTH},
        "results": results,
    }
    (OUT_DIR / args.out).write_text(json.dumps(payload, indent=2))
    print("\nWrote", OUT_DIR / args.out)


if __name__ == "__main__":
    main()
