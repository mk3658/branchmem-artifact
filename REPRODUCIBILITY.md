# REPRODUCIBILITY.md

Exact settings and commands needed to reproduce every number in the paper,
without spending on new API calls (the cache in `llm_cache/` is committed).

## Model and decoding parameters

| Component | Model | Temperature | Max tokens | top\_p |
|---|---|---|---|---|
| `ThreeWayLLMMerge`, `TwoWayLLMMerge`, `RawTextLLMMerge` (locked run) | `gpt-5.4-nano` | `0.0` | `8000`* | provider default (not set) |
| `LLMJudgeConflictDetector` | `gpt-5.4-nano` | `0.0` | `1024` | provider default (not set) |
| Post-hoc `gpt-5.4-mini` robustness check | `gpt-5.4-mini` | `0.0` | `1024` | provider default (not set) |
| Post-hoc cross-vendor checks (`run_cross_vendor_*.py`) | `claude-haiku-4-5-20251001` | `0.0` | `1024`/`8000` | n/a (Anthropic backend) |
| Post-hoc frontier-tier check (`run_stronger_model.py`) | `gpt-5.4` | `0.0` | `8000` | provider default (not set) |
| Post-hoc signal-coverage sweep (`run_signal_coverage.py`) | `gpt-5.4-nano` | `0.0` | `2048` | provider default (not set) |
| Post-hoc open-weight check (`run_open_weight.py`) | `llama3.1:8b` (Ollama, local) | `0.0` | `8000` | `http://localhost:11434/v1` |

\* The original locked run used `max_tokens=1024`/`2048`; raised to `8000`
for the expanded MemoryAgentBench and later ablation runs after discovering
that large batched JSON responses were silently truncated at the lower
limits. All numbers reported in the paper use the corrected `8000` limit for
any call that batches more than ~15 conflicts.

Backend: `branchmem/llm/openai_compatible_backend.py`, OpenAI's chat
completions API. `top_p` is never set explicitly (uses the API's own
default). Newer reasoning-family models sometimes reject `max_tokens` (want
`max_completion_tokens`) or a non-default `temperature`; the backend
detects this via a `BadRequestError` retry (up to 3 attempts) and remembers
the fix per backend instance — this is a client-side compatibility shim, not
a change to the requested decoding parameters themselves.

**API run dates**: locked confirmatory run and pilot, 2026 (see git history
for exact dates); post-hoc analyses (new baselines, expanded MAB,
`RawTextLLMMerge`, resolvable-category power expansion, abstention
metrics, semantic-resolvable category) run 2026-07-04, per this file's own
edit date.

## Retry policy and failure handling

- Network/API errors: handled by the OpenAI SDK's own retry behavior; the
  backend's 3-attempt loop (above) is specifically for the
  `max_tokens`-vs-`max_completion_tokens` and temperature-support
  incompatibilities, not general transient failures.
- **JSON parsing policy**: `ThreeWayLLMMerge`/`TwoWayLLMMerge`/
  `RawTextLLMMerge` expect a JSON object with a `resolutions` array. If
  `json.loads` fails on the raw response text, every conflict in that batch
  is marked `FLAGGED_UNRESOLVED` rather than guessed — this is a
  conservative fallback, not a silent skip. Historical JSON failure rate:
  zero in the current cache (verified by scanning `llm_cache/*.json` for
  entries whose `text` field fails `json.loads()`).

## Cache key construction

`branchmem/llm/cache.py`: the cache key is
`sha256(json.dumps({backend, model, system, prompt, temperature, max_tokens, ...}, sort_keys=True))`.
Any change to the prompt text, system prompt, model, or decoding parameter
produces a new key — there is no cross-configuration cache reuse. One JSON
file per unique key under `llm_cache/`, containing the full request
parameters, response text, model, and token counts. `find_collisions()` in
`branchmem/merge/base.py` iterates conflict keys in **sorted**, not raw
`set`, order specifically so the JSON built from it — and thus the cache
key — is stable across separate Python process invocations (Python
randomizes `set`/string-hash iteration order per process by default).

## Reproduce without spending anything

```bash
source .venv/bin/activate
python -m pytest tests/ -q                      # unit tests, no API key needed
python scripts/run_experiment.py                # locked Phase 6 run — reads from llm_cache/, $0 if cache populated
```

Every real LLM call across every round of this project (pilot, locked run,
post-hoc checks, new baselines, semantic-resolvable category, balanced
detector benchmark) is committed under `llm_cache/`, so none of the above
commands make a network call as long as the exact prompt/config combination
was already run once.

## Total call volume

The paper's Experimental Setup section reports the headline total for the
locked run (460 calls; 111,403 input / 45,462 output tokens);
later post-hoc rounds add further cached calls under `llm_cache/`, all
released alongside this repository.

## Where the real-content MAB numbers come from

`results/final/mab_secondary_analysis_expanded.json` is not written by a script
in this repository, which is worth saying plainly because every other result
file here is. It holds the reconciled MemoryAgentBench run --- seed 2026, 8
rows, 270 conflicts --- in the schema the figure script reads, and its numbers
are those of `results/final/mab_reconciled.json`, which supersedes an earlier
unseeded 230-conflict run whose accuracy and abstention sides contradicted each
other. Addendum item A12 records that correction and what changed. The two
files are expected to agree; if you regenerate either, check them against each
other.

The synthetic run and the reconciled MAB run do not share a strategy set: the
synthetic side has `branch_discard_fewer_updates`, the MAB side
`confidence_rule`. `scripts/make_figures.py` pairs only the strategies both
measured, so the synthetic-vs-real figure shows four, not five.


## BranchGuard revision: build and regeneration

## Compile the manuscript

```bash
pdflatex -interaction=nonstopmode -halt-on-error acl_latex.tex
bibtex8 acl_latex   # plain `bibtex acl_latex` is also valid when available
pdflatex -interaction=nonstopmode -halt-on-error acl_latex.tex
pdflatex -interaction=nonstopmode -halt-on-error acl_latex.tex
```

## Regenerate the Contextual BranchGuard study

```bash
PYTHONPATH=. python scripts/run_contextual_branchguard.py
```

This rewrites the CSVs under `results/contextual_branchguard/` and the main
`contextual_branchguard.png` figure (copy it to `paper/` to refresh Figure 2).

To reproduce the certified-risk-level sensitivity reported in the appendix,
re-run the same script with the utility-implied threshold:

```bash
PYTHONPATH=. BRANCHGUARD_LAMBDA=0.20 python scripts/run_contextual_branchguard.py
```

Results land in `results/contextual_branchguard_lambda0.2/`. Both runs are
pure simulation under seed 20260728 and make no model calls.

## Run tests

```bash
PYTHONPATH=. pytest -q tests/
```

Expected result: `135 passed` (whole suite).

## Prepare independent MAB annotation

After exporting the item-level conflicts into the supplied master schema:

```bash
PYTHONPATH=. python scripts/make_mab_adjudication_sheets.py \
  --items mab_items.csv --out annotation/generated --annotators 3

PYTHONPATH=. python scripts/analyze_mab_adjudication.py \
  --items mab_items.csv --annotations mab_annotations.csv \
  --out results/mab_adjudication
```

See `annotation/external_validation_protocol.md` before collecting labels.

## Two further ablations

Two experiments addressing objections to the main design. Neither needs new
model calls beyond the prompt ablation's two new conditions.

Prompt-priority ablation (three matched system prompts over the same 60
strict-constraint items, both provenance regimes). Condition P1 is the shipped
prompt and is served from cache; P2 and P3 issue new calls because the system
prompt is part of the content-hash cache key:

```bash
OPENAI_API_KEY=... PYTHONPATH=. python scripts/run_prompt_priority_ablation.py 60
```

End-to-end Contextual BranchGuard on observed conflicts, with every channel
reliability estimated from adjudicated labels rather than drawn from prescribed
Bernoulli masses. Fully cached:

```bash
OPENAI_API_KEY=cache-only-sentinel PYTHONPATH=. python scripts/run_empirical_channel_branchguard.py
```

## External adjudication and the hybrid resolver

Blinded independent adjudication of all 270 MAB-derived conflicts, following
`annotation/external_validation_protocol.md`. Step 1 needs network access once
to fetch the public MemoryAgentBench parquet; steps 2-4 are offline except for
the panel's model calls:

```bash
PYTHONPATH=. python scripts/make_mab_item_master.py
```

```bash
PYTHONPATH=. python scripts/make_mab_adjudication_sheets.py --items annotation/mab_item_master.csv --out annotation/mab_sheets --annotators 4
```

```bash
OPENAI_API_KEY=... PYTHONPATH=. python scripts/run_mab_adjudication_panel.py
```

```bash
PYTHONPATH=. python scripts/analyze_mab_adjudication.py --items results/final/mab_items_for_analysis.csv --annotations results/final/mab_adjudication_llm3.csv --out results/final/mab_adjudication_llm3
```

Set `MAB_EVIDENCE_LINES=30` on the first command and `MAB_SHEETS=` on the third
to reproduce the wide-evidence second pass.

### Six-member panel and the vendor-independence analysis

The panel now has six members: three OpenAI models, a local DeBERTa-v3 NLI
model, a local open-weight Llama 3.1 8B served by Ollama, and a local RoBERTa
NLI model. The last three share no vendor, weights or training data with the
first three, and the analysis reports that sub-panel separately. Step 2 needs
`ollama pull llama3.1:8b` and a running Ollama server; the two NLI checkpoints
download from the Hugging Face hub on first use.

```bash
PYTHONPATH=. python scripts/make_mab_adjudication_sheets.py --items annotation/mab_item_master.csv --out annotation/mab_sheets --annotators 6
```

```bash
PYTHONPATH=. python scripts/make_mab_adjudication_sheets.py --items annotation/mab_item_master_ev30.csv --out annotation/mab_sheets_ev30 --annotators 6
```

```bash
OPENAI_API_KEY=... PYTHONPATH=. python scripts/run_mab_adjudication_panel.py --out mab_adjudication_panel6.csv
```

```bash
MAB_SHEETS=annotation/mab_sheets_ev30 OPENAI_API_KEY=... PYTHONPATH=. python scripts/run_mab_adjudication_panel.py --out mab_adjudication_panel6_ev30.csv
```

```bash
PYTHONPATH=. python scripts/analyze_panel_independence.py --pass 6:results/final/mab_adjudication_panel6.csv --pass 30:results/final/mab_adjudication_panel6_ev30.csv --items annotation/mab_item_master.csv --out results/final/mab_adjudication_summary.json
```

Regenerating the sheets with `--annotators 6` leaves sheets 1-4 byte-identical,
so the earlier four-member responses stay valid.

## WikiDivergence: independently sourced conflicts

Three-way merges taken from English Wikipedia infobox revision history, where
the divergence, the provenance metadata and the label are all external to this
paper. Step 1 needs network access and is polite but slow (serial requests,
cached under `.wiki_cache/`); it writes `data/wikidivergence_items.csv` and
`data/wikidivergence_pages.csv`, both of which ship with the artifact, so step
2 can be run without network access.

```bash
PYTHONPATH=. python scripts/build_wikidivergence.py --pages-per-cat 25
```

```bash
OPENAI_API_KEY=... PYTHONPATH=. python scripts/run_wikidivergence_eval.py
```

Because Wikipedia keeps changing, a later rebuild will not reproduce the
released item set; the shipped CSV is the frozen collection the paper reports.

## Prospective discrimination experiment (preregistered)

The protocol, hypotheses and item bank were fixed before any solver ran; see the
PROSPECTIVE PREREGISTRATION section of `ANALYSIS_PLAN_ADDENDUM.md`. Step 1
authors the bank and writes `data/discrimination_bank.sha256`; every later step
refuses to run if that hash no longer matches, so a preregistered result cannot
be reported against an edited bank.

```bash
OPENAI_API_KEY=... PYTHONPATH=. python scripts/author_discrimination_bank.py
```

```bash
OPENAI_API_KEY=... PYTHONPATH=. python scripts/run_discrimination_experiment.py --models gpt-5.4 --out discrimination_gpt-5_4.json
```

Repeat for `gpt-5.4-mini` and `gpt-5.4-nano`. The open-weight models need a
running Ollama server with `llama3.1:8b`, `qwen2.5:7b` and `mistral:7b` pulled,
and must be run ONE AT A TIME: three 8B models cannot stay resident together and
Ollama will otherwise swap a model per call.

```bash
PYTHONPATH=. python scripts/run_discrimination_experiment.py --models llama3.1:8b --arms update_favoured --types immutable_constraint,legitimate_override --out discrimination_llama318b.json
```

The two-stage mitigation (hypothesis H4). Stage two is API-free, so a local
classifier gives a pipeline that needs no API at all:

```bash
OPENAI_API_KEY=... PYTHONPATH=. python scripts/run_constraint_aware_router.py --classifiers gpt-5.4,gpt-5.4-nano,llama3.1:8b --arms update_favoured,neutral,inverted
```

```bash
PYTHONPATH=. python scripts/analyze_discrimination.py
```

## Offline two-agent divergence (preregistered)

Environments are authored and frozen first; the branch contents are the agent's
own memory writes, decided at run time.

```bash
OPENAI_API_KEY=... PYTHONPATH=. python scripts/author_agent_episodes.py
```

```bash
OPENAI_API_KEY=... PYTHONPATH=. python scripts/run_agent_divergence.py
```

## Routing-aware certification

Oracle, predicted and conservative channel assignment, plus the purity a routed
inverted channel needs to stay certifiable. Fully cached:

```bash
OPENAI_API_KEY=cache-only-sentinel PYTHONPATH=. python scripts/run_routing_aware_certification.py
```

## Matched regime table

One item bank, three provenance assignments, six policies on identical items.
Replaces an earlier table whose columns were three different populations. Fully
cached:

```bash
OPENAI_API_KEY=cache-only-sentinel PYTHONPATH=. python scripts/run_matched_regime_table.py
```

## Ancestor-strength ablation

Matched intervention over the modal force of the ancestor: identical predicates
and branch values, hard constraint versus revisable preference, each item
merged twice with opposite provenance. The strict arm is served from cache; the
defeasible arm issues new calls.

```bash
OPENAI_API_KEY=... PYTHONPATH=. python scripts/run_ancestor_strength_ablation.py
```

## Channel governance: assignment, misassignment and cost

Label-free channel router, an independent misassignment sweep, calibration
sample complexity and audit cadence. Fully cached:

```bash
OPENAI_API_KEY=cache-only-sentinel PYTHONPATH=. python scripts/run_channel_governance.py
```

Semantic-then-provenance hybrid against both parents, fully cached:

```bash
OPENAI_API_KEY=cache-only-sentinel PYTHONPATH=. python scripts/run_constraint_veto_hybrid.py
```

Note: this script's ABSENT arm is built from
`generate_semantic_resolvable_scenarios(seed=3001)` (the 12-template bank),
while its informative and misleading arms come from the realistic
strict-constraint bank, so its three columns are not one population. The
paper's hybrid table therefore takes its absent column from
`scripts/run_matched_regime_table.py` instead, which runs all three arms on the
same bank; the informative and misleading columns agree exactly between the two
scripts.

## Specification ablation: is the failure the model's or the prompt's?

Seven matched merge system prompts over the frozen 320-item discrimination
bank. P1-P6 differ from the shipped prompt in exactly one clause, the one that
specifies how to weigh the signals: P4 and P5 raise the standing-rule question
without naming qualifying categories or marking which items are which; P3
asserts constraint primacy and enumerates categories; P6 is elicited
chain-of-thought. P1 is the shipped prompt and is served from the
preregistered run's cache.

**P0 is the exception and the only condition that is not a one-clause diff.**
The shipped prompt names the metadata in three places - the input description,
the trust-ordering clause and the abstention instruction - and P0 replaces all
three, so it mentions no metadata, no trust ordering and no standing rule at
all. The branch records still carry `source` and `confidence`, because those
are the data rather than the prompt; P0 asks only whether being *told* about
them is what produces the field-reading. An assertion in `_build_p0()` fails
the run if any of those words survives, and
`tests/test_specification_ablation.py` pins the same property.

P0 abstains far more than the shipped prompt (coverage .750 vs .956), so read
`took_update_rate` and `wrong_commit_rate` beside the accuracies rather than
the balanced accuracy alone.

P0 is simply the first condition of the `gpt-5.4` run below, so no separate
command is needed. To run it alone, beside the shipped prompt for reference:

```bash
OPENAI_API_KEY=... PYTHONPATH=.:scripts python scripts/run_specification_ablation.py --models gpt-5.4 --prompts P0_unprimed,P1_shipped --types immutable_constraint,legitimate_override --out specification_ablation_p0_only.json
```

```bash
OPENAI_API_KEY=... PYTHONPATH=.:scripts python scripts/run_specification_ablation.py --models gpt-5.4 --types immutable_constraint,legitimate_override --out specification_ablation_gpt54.json
```

```bash
OPENAI_API_KEY=... PYTHONPATH=.:scripts python scripts/run_specification_ablation.py --models llama3.1:8b --types immutable_constraint,legitimate_override --out specification_ablation_llama.json
```

Run the open-weight models one at a time: three 8B models cannot stay resident
in Ollama together, and it otherwise swaps a model per call, which slows a run
by roughly an order of magnitude. Scenario construction, arms, seeds and scoring are
imported from `run_discrimination_experiment.py`, so the items are identical to
the preregistered run and the P1 row reproduces it exactly. That reproduction
is the harness check for this experiment.

## Label-norm sensitivity: does the claim rest on our convention?

Classifies all 80 constraint ancestors by who could lift the constraint,
re-scores every model under the competing "user authority" norm, and restricts
the primary measure to the subset where that norm cannot apply. No API calls.

```bash
PYTHONPATH=. python scripts/analyze_label_norms.py
```

Writes `results/final/label_norm_sensitivity.json` and the auditable
classification table `annotation/discrimination_authority_locus.csv`.

## Human adjudication of the discrimination bank (COLLECTED, ALL 320 ITEMS)

The bank has been adjudicated by people in two disjoint waves: a stratified
100-item sample first, then the remaining 220, three annotators each. The
protocol, the questions and the analysis plan fixed in advance are in
`annotation/discrimination_adjudication_protocol.md`.

Score a wave, or all 320 together. No API calls; the answers are read from the
returned sheets.

```bash
PYTHONPATH=. python scripts/score_discrimination_adjudication.py --wave full
```

`--wave 1` scores `annotation/discrimination_sheets/` and reproduces every
number first reported from the sample; `--wave 2` scores
`annotation/discrimination_sheets_wave2/`; `--wave full` pools both and writes
`results/final/discrimination_human_adjudication_full_bank.json`.

The wave-1 sheet generator is still deterministic given the pinned bank hash
and seed 9091, and refuses to overwrite sheets that hold answers:

```bash
PYTHONPATH=. python scripts/make_discrimination_adjudication_sheets.py --manifest-only
```

Wave 2's sheets were issued by the study organizer rather than generated here,
so its held-back key is derived from the returned sheets and the frozen bank
and is reproducible from them:

```bash
PYTHONPATH=. python scripts/build_wave2_adjudication_key.py
```

The two waves used differently worded briefs, and the difference is in the
instruction about flagging an item for review, which is the one that moves how
often a person declines to resolve. The scorer checks this and the pooled
output carries the disclosure; `tests/test_human_adjudication_wave2.py` fails
if it is dropped while the briefs still differ.

## Composing the two fixes (no API key needed)

Chains the standing check and the specification-prompted merge per item on
BranchMem-Hard and scores four arms: each alone, either-may-reject, and
both-must-reject. Every call is served from `llm_cache/`.

```bash
OPENAI_API_KEY=cache-only-sentinel PYTHONPATH=.:scripts python scripts/run_composed_pipeline.py
```

Writes `results/final/composed_pipeline.json`. Re-running is byte-identical.

## BranchMem-Hard

Hand-authored bank in which an obligation-bearing ancestor no longer indicates
whether an update should be rejected. Deterministic; re-running reproduces the
pinned hash.

```bash
PYTHONPATH=. python scripts/author_hard_standing_bank.py
```

`tests/test_hard_standing_bank.py` enforces the no-cue property: the keyword
marker must fire at comparable rates on all four obligation types, so a later
edit that reintroduces a lexical shortcut fails the suite.

## Standing detection: when is the distinction at stake?

Runs one general standing check over every key, varying what it is shown.
`ancestor` scope sees only the stored entry; `ancestor_update` also sees the
proposed update, where expiry, unmet conditions and waivers are stated.

```bash
OPENAI_API_KEY=... PYTHONPATH=.:scripts python scripts/run_standing_detection.py --llm-models gpt-5.4,gpt-5.4-nano,o3 --out standing_detection.json
```

Needs `scikit-learn` and `sentence-transformers` for the trained baselines; no
API calls for those. Both error directions are reported together
(`violations_caught`, `freezing_rate`), since either can be bought with the
other.

## BranchMem-Hard end to end

Re-runs the specification prompts as merges on the harder bank, to test whether
the .938 result depends on the easier bank's keyword cue.

```bash
OPENAI_API_KEY=... PYTHONPATH=.:scripts python scripts/run_hard_bank_merge.py --models gpt-5.4,gpt-5.4-nano,o3 --out hard_bank_merge.json
```

## Reasoning models

`o3` reaches the OpenAI chat endpoint and needs no special handling: the backend
retries with `max_completion_tokens` and drops `temperature` on the first 400
and remembers. `gpt-5.4-pro` is not a chat model and cannot be used here.

```bash
OPENAI_API_KEY=... PYTHONPATH=.:scripts python scripts/run_specification_ablation.py --models o3 --prompts P1_shipped,P4_unmarked_rule --types immutable_constraint,legitimate_override --out specification_ablation_o3.json
```

Local reasoning models need `--max-tokens` below Ollama's default 4096-token
context, or they emit scratchpad until truncation and never reach the JSON:

```bash
OPENAI_API_KEY=... PYTHONPATH=.:scripts python scripts/run_specification_ablation.py --models qwen3:8b --prompts P1_shipped,P4_unmarked_rule --types immutable_constraint,legitimate_override --max-tokens 3000 --out specification_ablation_qwen3.json
```

Neither 8B thinking model completes the P4 specification condition.
`deepseek-r1:8b` spends its whole budget on scratchpad (8000 output tokens,
~400 s per call) and `qwen3:8b` does the same under P4 (~14,000 characters,
33 of 55 replies unparseable), though `qwen3:8b` completes P1 cleanly (160
items, 0 parse failures). Replies are That is reported as a failure mode, not scored. Unparseable
replies are counted in `n_parse_failures` and never credited as abstentions —
see the note in `branchmem/merge/three_way_llm.py`.
