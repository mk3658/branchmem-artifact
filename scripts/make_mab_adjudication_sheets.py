"""Create independently randomized, blinded MAB adjudication sheets."""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

REQUIRED = {
    "conflict_id", "source_row", "channel", "predicate_family", "entity",
    "predicate", "ancestor_value", "branch_a_value", "branch_b_value",
    "evidence_context", "derived_label", "provenance_choice",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--annotators", type=int, default=3)
    ap.add_argument("--seed", type=int, default=20260728)
    args = ap.parse_args()
    if args.annotators < 3:
        raise SystemExit("Use at least three independent annotators.")
    items = pd.read_csv(args.items, keep_default_na=False)
    missing = REQUIRED - set(items.columns)
    if missing:
        raise SystemExit(f"Missing required columns: {sorted(missing)}")
    if items.conflict_id.duplicated().any():
        raise SystemExit("conflict_id must be unique")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    keys = []
    for a in range(1, args.annotators + 1):
        rows = []
        order = rng.permutation(len(items))
        for presentation_id, idx in enumerate(order, start=1):
            row = items.iloc[idx]
            swap = bool(rng.integers(0, 2))
            left = row.branch_b_value if swap else row.branch_a_value
            right = row.branch_a_value if swap else row.branch_b_value
            rows.append({
                "presentation_id": presentation_id,
                "conflict_id": row.conflict_id,
                "source_row": row.source_row,
                "channel": row.channel,
                "predicate_family": row.predicate_family,
                "entity": row.entity,
                "predicate": row.predicate,
                "ancestor_value": row.ancestor_value,
                "left_value": left,
                "right_value": right,
                "evidence_context": row.evidence_context,
                "resolution": "",
                "confidence_1_5": "",
                "evidence_span": "",
                "rationale": "",
            })
            keys.append({
                "annotator_id": f"ann{a}",
                "presentation_id": presentation_id,
                "conflict_id": row.conflict_id,
                "left_original": "B" if swap else "A",
                "right_original": "A" if swap else "B",
            })
        pd.DataFrame(rows).to_csv(out / f"annotator_{a}_sheet.csv", index=False)
    pd.DataFrame(keys).to_csv(out / "PRIVATE_presentation_key.csv", index=False)
    print(f"Wrote {args.annotators} blinded sheets for {len(items)} conflicts to {out}")


if __name__ == "__main__":
    main()
