# Independent MAB Adjudication Package

This folder is an execution-ready protocol for independently adjudicating the
270 MemoryAgentBench-derived branch conflicts discussed in the paper. It does
**not** contain completed human labels. The paper must continue to describe
those labels as missing until three independent annotators complete the sheets.

## Files

- `external_validation_protocol.md`: sampling, blinding, annotation, and
  analysis plan.
- `mab_item_master_template.csv`: schema to populate from the BranchMem MAB
  extension. The `derived_label` and `provenance_choice` columns form the hidden
  key and must not be shown to annotators.
- `mab_annotation_template.csv`: long-format annotation schema.
- `../experiments/make_mab_adjudication_sheets.py`: randomizes A/B presentation
  separately for each annotator and removes hidden-key columns.
- `../experiments/analyze_mab_adjudication.py`: validates labels, computes
  agreement and label validity, and evaluates Contextual BranchGuard versus
  BayesianSourceMerge on source-row-separated calibration/test histories.

## Minimal workflow

1. Export all derived conflicts into `mab_items.csv` using the master schema.
2. Run:

   ```bash
   python experiments/make_mab_adjudication_sheets.py \
     --items mab_items.csv --out annotation/generated --annotators 3
   ```

3. Have annotators work independently. They must see the relevant original
   context/evidence but not the derived label, provenance metadata, or another
   annotator's response.
4. Combine completed long-format labels into `mab_annotations.csv`.
5. Run:

   ```bash
   PYTHONPATH=. python experiments/analyze_mab_adjudication.py \
     --items mab_items.csv --annotations mab_annotations.csv \
     --out results/mab_adjudication
   ```

A paper revision should report the completed study only after inspecting the
script outputs and releasing the de-identified annotation files.
