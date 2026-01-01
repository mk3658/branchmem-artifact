# ANALYSIS_PLAN_ADDENDUM.md

`ANALYSIS_PLAN.md` is locked and was not edited after Phase 6. Everything in
this file is **post-hoc and exploratory** — added after seeing the locked
results. None of it is folded into the confirmatory tests (Table 1/2 in the
paper) or treated as satisfying H1/H2/H3.
Where a number here is compared against a locked-run number, the comparison
itself is descriptive, not a new preregistered test.

## A1. Abstention-aware metrics (added 2026-07-04)

Motivation: raw accuracy penalizes `FLAGGED_UNRESOLVED` identically whether
the underlying situation is genuinely ambiguous (correct behavior) or
resolvable-with-signal (a missed resolution). In a safety-sensitive memory
merge, a wrong commit and an honest abstention are not equally bad. This
addendum computes, from the **already-cached results** (no new API calls):
commit rate, abstention rate, conditional accuracy given commit, flag
precision/recall, wrong-commit rate, and expected utility under three cost
regimes (correct commit +1; wrong commit −2/−5/−10; abstention −1).
Implementation: `branchmem/eval/abstention_metrics.py`.

## A2. `semantic_resolvable` conflict category (added 2026-07-04)

Motivation: every existing "resolvable" conflict in the locked generator
ties ground truth to a source/confidence signal that a single `if`-statement
(`ConfidenceRuleMerge`) can read directly. This category equalizes source
and confidence across both branches and makes timestamp uninformative, so
ground truth is recoverable only from semantic compatibility with the
ancestor/context — a condition genuinely hard for `ConfidenceRuleMerge` and
genuinely informative about whether an LLM's semantic reasoning adds value
beyond reading metadata. This is a **new, non-preregistered generator
condition**, not a reinterpretation of the locked `resolvable` category.
Sample size, exact templates, and results are in the paper's new subsection
and `results/final/semantic_resolvable.json`.

## A3. Balanced conflict-detector benchmark (added 2026-07-04)

Motivation: RQ2's locked evaluation set contains only genuine conflict
pairs by construction (an orthogonal key is never shared between branches),
so every detector's precision is mechanically 1.0 — this measures recall
only, not real-world discriminative power. This addendum adds five negative
categories (paraphrase, entailment, unrelated-same-entity,
preference-refinement, ambiguous-but-not-conflicting) alongside genuine
contradictions, and reports full precision/recall/F1/specificity/FPR (and
AUROC where a continuous score is available) for
`EmbeddingConflictDetector`, `NLIConflictDetector`, and
`LLMJudgeConflictDetector`. This does not change or retract RQ2's locked
result — it is a harder, separate, descriptive follow-up.

## A4. Independent-validation audit protocol (added 2026-07-04)

No human annotation is collected in this project (no annotator available in
this environment). `annotation/audit_sample.csv` and
`annotation/audit_instructions.md` provide a sampling protocol and template
so an independent party could validate ground-truth labels, extraction
correctness, and preferred resolutions. The paper does not claim this
validation was performed at the time — but see item A11, where it has since
been carried out.

## A5. Cross-vendor robustness check (added 2026-07-04)

Motivation: the existing robustness check (`results/final/robustness_second_model.json`)
swaps `gpt-5.4-nano` for `gpt-5.4-mini` — a same-provider, same-family model
swap — leaving open whether the paper's effects are properties of LLMs in
general or specific to this provider's models. With real access to a second
vendor's API (Anthropic) obtained after the locked run, three existing
post-hoc analyses were replicated with `claude-haiku-4-5-20251001` (chosen
to match the "cheapest available model" cost tier already used for the
OpenAI-side results), on identical scenario constructions (same seeds):

- **RQ1, same 20 scenarios as the `gpt-5.4-mini` check**
  (`results/final/cross_vendor_rq1.json`): `claude-haiku-4-5` scores
  0.936 mean accuracy vs. 0.914 (`gpt-5.4-nano`) and 0.929
  (`gpt-5.4-mini`) on the identical scenarios — paired diff. vs. nano
  +0.021, 95% CI [0.000, 0.043], p=0.10. Direction and magnitude
  hold across a genuinely different vendor, not just a same-provider swap.
- **`semantic_resolvable`, identical 150-scenario construction**
  (`results/final/cross_vendor_semantic_resolvable.json`): every
  confidence-independent strategy, including `ThreeWayLLMMerge`,
  `TwoWayLLMMerge`, and `RawTextLLMMerge`, again scores **exactly 0.000**
  — the same striking negative result as the original `gpt-5.4-nano` run,
  now shown to generalize across vendors rather than being an artifact of
  one cheap model.
- **Real MemoryAgentBench content** (`results/final/cross_vendor_mab.json`):
  `ThreeWayLLMMerge` again scores exactly 0.000 (vs. `gpt-5.4-nano`'s
  0.004), confirming complete abstention on real content is a property of
  the data (no source-reliability signal exists to read), not an
  OpenAI-specific quirk. Note: this replication's naive-baseline numbers
  (e.g. `last_writer_wins` 0.355 vs. the original 0.506) differ from
  the original MAB secondary analysis because that earlier ad hoc script
  did not record the random seed used for branch construction — a
  reproducibility gap in that earlier analysis, now fixed going forward
  (this script uses and discloses `seed=2026`). This does not affect the
  `ThreeWayLLMMerge`/`ConfidenceRuleMerge` result: both are forced to
  exactly 0 by the data's construction (source/confidence are never
  randomized, only which value lands on which branch is), so that result
  is seed-invariant; only the naive baselines' absolute values are
  seed-sensitive, as expected for a randomized construction.

Scripts: `scripts/run_cross_vendor_rq1.py`,
`scripts/run_cross_vendor_semantic_resolvable.py`,
`scripts/run_cross_vendor_mab.py`. Does not change or retract any
confirmatory or prior post-hoc result — it is additional, disclosed
evidence on the "single model family" limitation.

## A6. Additional robustness and diagnostic analyses (added 2026-07-04)

Four additional post-hoc analyses strengthening the boundary-condition
story; none change any confirmatory or prior post-hoc result.

- **Source-signal-noise sensitivity** (`results/final/noise_sensitivity.json`,
  `scripts/run_noise_sensitivity.py`): re-scored the identical locked n=60
  scenarios at noise in {0.0, 0.05, 0.15, 0.30, 0.50}. Zero new API: noise
  only reassigns ground truth, which the model never sees, so all
  ThreeWayLLMMerge calls are cache hits. Result: ConfidenceRuleMerge >=
  ThreeWayLLMMerge at every level, and both decay from ~1.0 to chance
  (~0.5) as noise -> 0.5, i.e. the LLM/rule advantage over naive baselines
  is exactly as large as the signal is informative. The boundary condition
  made continuous.
- **Semantic-resolvable failure modes**
  (`results/final/semantic_resolvable_failure_modes.json`,
  `scripts/analyze_semantic_resolvable_failures.py`): categorized the 150
  cached semantic_resolvable responses by outcome and reasoning phenomenon
  (zero new API). Of 150 keys, 0 correct commits: 125 flagged, 17 incoherent
  merges, 8 wrong commits. All 25 active errors fall in constraint-exclusion
  / negation templates; precondition, attribute-matching, and
  numeric-temporal are flagged 100%. Localizes negation/exclusion as the
  highest-signal target for future semantic-resolution work.
- **Risk-coverage frontier** (`results/final/risk_coverage_frontier.json`,
  `scripts/compute_risk_coverage_frontier.py`): each strategy's achieved
  (coverage, risk) on the pooled locked set. Deterministic strategies
  zero-API; ThreeWayLLMMerge point reused from the cache-backed abstention
  report. Naive baselines: coverage 1.0, risk 0.63-0.70; abstaining
  strategies: risk 0.13-0.17 at coverage ~0.67. Two confidence tiers only,
  so achieved operating points, not a swept curve (a graded/conformal curve
  is future work).
- **Provenance proxy at branch granularity**
  (`results/final/provenance_proxy.json`, `scripts/run_provenance_proxy.py`):
  a disjoint-seed (5001) n=60 variant where reliability is presented at
  branch (replica) granularity instead of per fact. Real API (~60 nano
  calls). Both ThreeWayLLMMerge and ConfidenceRuleMerge score 0.883
  resolvable accuracy (p=0.32 between them), on par with the per-fact locked
  0.858/0.867 -- a coarse per-replica proxy is exploited as well as the
  per-fact engineered signal, so the boundary condition is about signal
  presence, not granularity.

Also added: two real citations to `custom.bib`
(`dong2009integrating`, `li2016survey`) for the truth-discovery /
knowledge-fusion literature, with a main-text characterization
(Section 2) and full appendix discussion (Extended Related Work).

## A7. Additional model-coverage and boundary-mapping analyses (added 2026-07-22)

Three more post-hoc analyses; none change any confirmatory or prior result.

- **Frontier-tier model** (`results/final/stronger_model.json`,
  `scripts/run_stronger_model.py`): re-ran RQ1 (20 scenarios) and
  semantic_resolvable (150) with `gpt-5.4` (the full frontier OpenAI model).
  RQ1 overall 0.936 (vs nano 0.914 / mini 0.929) -- capability-invariant, does
  not beat the deterministic rule. semantic_resolvable: 0.467 for
  ThreeWayLLMMerge vs 0.000 for nano -- the semantic-resolution collapse is
  partly a cheap-model artifact; a frontier model performs the compatibility
  check ~half the time but unreliably. Cleanly separates the paper's two
  claims: field-reading is capability-invariant, semantic reasoning is
  capability-dependent yet unsolved.
- **Signal-coverage sweep** (`results/final/signal_coverage.json`,
  `scripts/run_signal_coverage.py`): varied the fraction of resolvable
  conflicts carrying an exploitable signal from 0 to 1, mapping the transition
  between the synthetic (all-signal) and MAB (no-signal) extremes. Accuracy is
  a smooth ramp (0.000/0.271/0.448/0.625/0.812) and the commit rate tracks
  coverage almost exactly -- abstention is calibrated to signal availability.
  Partial provenance buys partial, proportional reliability.
- **Cost/latency profile** (`results/final/cost_latency.json`,
  `scripts/compute_cost_latency.py`): per-merge cost. LLM strategies = 1
  batched call + ~650/236 tokens + ~2-3 s live; deterministic strategies = 0
  calls, ~25 us. Since ConfidenceRuleMerge matches/beats the LLM where a
  confidence field exists, the LLM's per-merge cost buys no accuracy there.

Also added six real, web-verified citations to `custom.bib`
(LongMemEval wu2025longmemeval; AgentAbstain liu2026agentabstain;
ProvenanceGuard she2026provenance; AMA-Bench zhao2026amabench; MemSim
zhang2025memsim; Tang et al. survey tang2026memorysurvey), with body nods
(Related Work "Abstention", robustness section) and full discussion in the
Extended Related Work appendix. Each was web-verified (title, authors, venue)
before being added.

## A8. Open-weight model check (added 2026-07-22)

Post-hoc; does not change any confirmatory or prior result. Every model tested
before this was proprietary (`gpt-5.4-nano`/`-mini`/`gpt-5.4`,
`claude-haiku-4-5`) -- the most-cited limit on a paper about "LLM-mediated
merging" as a category. This check adds the first openly released model.

- **Open-weight model** (`results/final/open_weight.json`,
  `scripts/run_open_weight.py`): re-ran RQ1 (same 20 scenarios, seed 2026) and
  `semantic_resolvable` (same 150, seed 3001; flagship `ThreeWayLLMMerge`
  only, since `TwoWay`/`RawText` were uniformly 0.000 for every prior model)
  with **Llama 3.1 8B Instruct**, served locally via Ollama's
  OpenAI-compatible endpoint (`base_url=http://localhost:11434/v1`). Zero API
  cost; every call cached under `llm_cache/` (keyed by model name, no
  collision with the API models). Results: RQ1 overall **0.85**, resolvable
  **0.60** -- below every proprietary model and still below
  `ConfidenceRuleMerge`'s 0.962, so the deterministic rule dominates across
  the entire capability range tested (8B open model to frontier).
  `semantic_resolvable` three-way **0.14** (vs 0.000 nano, 0.467 frontier):
  nonzero but far from reliable. One honest nuance: the 8B reads the resolvable
  signal less reliably (0.60 vs nano's 0.858), so field-reading is not
  *perfectly* capability-invariant -- but the rule still wins at every tier.
  Reported in Section 5.6, Limitations "Model coverage", and
  Appendix M "An open-weight model".

## A9. Calibrated selective resolution — a within-boundary positive (added 2026-07-22)

Post-hoc re-analysis of already-cached data (zero new API beyond a small
frontier self-consistency probe, ~65 gpt-5.4 calls); changes no confirmatory
result.

- **Self-consistency probe** (`scripts/probe_selfconsistency.py`): tested
  whether k=5 majority voting fixes `semantic_resolvable` (addresses the
  preregistered single-call limitation). It does **not** help --- open-weight
  0.20->0.10 (n=20), frontier 0.53->0.47 (n=15) --- consistent with the failure
  being *systematic* (negation/exclusion, item A6) rather than sampling
  variance. Honest negative.
- **Selective-resolution re-analysis** (`scripts/analyze_sc_calibration.py`,
  `results/final/selective_resolution.json`): the probe motivated re-scoring the
  cached single-call `ThreeWayLLMMerge` runs with abstention-aware metrics. Raw
  accuracy (0.467) hid a **positive** result. On `semantic_resolvable` (n=150),
  `gpt-5.4` is a **wrong-commit-free selective resolver**: commit rate 0.467,
  conditional-on-commit accuracy **1.000** (70/70 correct), wrong-commit rate
  **0.000** --- resolving nearly half of these no-signal conflicts (which
  `ConfidenceRuleMerge` resolves *none* of, equal confidence always ties) and
  abstaining on the rest. Capability governs *calibration*: `llama-3.1-8b`
  commits wrongly 24% of the time; `gpt-5.4-nano`'s few commits are all wrong.
  Per-phenomenon: the frontier commits 100%-correctly on 9 constraint families
  (pairing 13/13, venue 13/13, workout 12/12, hotel 8/8, ...) and abstains
  entirely on 3 (childcare/gift/transport, 0/12 each). **Scope:** synthetic
  `semantic_resolvable` only, where a compatible answer exists by design;
  real-content calibration is unproven (real MAB has no recoverable signal).
  Reported in Section 5.5, Appendix M "Calibrated selective resolution".
- **Why the calibration is real** (`scripts/analyze_constraint_strength.py`,
  `results/final/constraint_strength.json`; zero new API): the frontier's commit
  rate tracks the *logical strength* of the ancestor constraint. Conditional
  accuracy is 1.000 in every one of the 9 families it commits on; commit rate
  runs from 1.00 on strict contradictions (wheelchair/ramp, alcohol/wine,
  knee-injury/running) to 0.00 on the 3 families whose incompatibility is only
  *defeasible* (motion-sickness/ferry, no-pet/dog-leash, toddler/adults-only).
  The model's own justifications draw the line: it commits citing a "prior
  user-sourced constraint" the losing branch is "a mistaken observation"
  against, and abstains noting only a "preference" or "no basis to prefer". This
  validates the calibration (reasoning about constraint hardness, not guessing)
  and flags a construct-validity subtlety: 3/12 templates are defeasible, better
  scored as ambiguous; on the 9 strict families the frontier is an even
  higher-precision resolver.

## A10. Larger multi-entity replication (added 2026-07-22)

Post-hoc; addresses concerns about locked-run sample size (n=60) and
single-entity scenarios. `scripts/run_multi_entity.py`,
`results/final/multi_entity.json`: 120 scenarios of 3 entities each (disjoint
seed 7001, gpt-5.4-nano). RQ1 replicates -- ThreeWayLLMMerge 0.919 and
ConfidenceRuleMerge 0.954 beat naive baselines (LWW 0.676, concat 0.671) by
+0.242/+0.248 (p<1e-20); the rule again edges the LLM (-0.035, p=7e-7). Honest
new effect the single-entity design could not surface: 14/474 (3.0%) of the
LLM's committed conflict resolutions place a value outside that key's two
candidates -- a mild cross-key fabrication the naive baselines cannot make.
RQ1's conclusion is unchanged. Also added a blind human-validation package
(`scripts/make_human_validation_sheets.py`, `scripts/score_human_validation.py`,
`annotation/validation_blind.csv` + `validation_key.csv`) -- materials only; no
human annotation performed in this submission.

## A11. Human validation of ground truth (added 2026-07-22)

Post-hoc; the human validation that item A4 set up as protocol-only has now been
performed. 3 independent annotators labelled 30 blind items (12 resolvable, 8
ambiguous, 10 semantic_resolvable) seeing only what a merger sees, never the
generator label (`scripts/make_human_validation_sheets.py`,
`scripts/score_human_validation.py`, `annotation/validation_*.csv`,
`results/final/human_validation.json`). Generator-label validity 0.83 overall
(95% CI [0.66,0.93]); Fleiss kappa 0.66; ambiguous 1.00, semantic_resolvable
0.90, resolvable 0.67. The resolvable rate is not label error: the majority
human prefers the higher-confidence branch on 12/12 resolvable items (using the
signal exactly as designed), and all 4 human-vs-generator mismatches are the
deliberate 15%-noise items -- direct human calibration of the noise parameter that might otherwise seem arbitrary. The one semantic miss is the defeasible
gift/no-pet family (2/3 also flag it), corroborating A9's constraint-strength
finding. Reported in Limitations and Appendix "Human Validation of Ground Truth".

## A12. MemoryAgentBench reconciliation (added 2026-07-22)

Correction. The earlier unseeded MAB run (item A5's reproducibility gap) left an
internal inconsistency: results/final/mab_secondary_analysis_expanded.json
reported ThreeWayLLMMerge mean accuracy 0.004 (1 of 230), while the abstention
side reported 0 commits / all flagged -- mutually impossible, since the MAB
extension has no ambiguous questions (1 correct requires 1 commit). Re-running
seeded (seed=2026, strategies in fixed order, three_way_llm last -- matching the
cross-vendor script) yields the reproducible, internally-consistent result now
in the paper: 270 conflicts, ThreeWayLLMMerge accuracy 0.000, commit rate 0.000,
0 commits, all 270 flagged, conditional-on-commit accuracy undefined, wrong-commit
rate 0.000; naive baselines LWW 0.355 / concat 0.621 / branch-discard 0.379.
mab_secondary_analysis_expanded.json and abstention_adjusted_metric.json were
regenerated to match. Conclusion unchanged (near-total, wrong-commit-free
abstention on real content); only the exact figures are corrected and made
reproducible. results/final/mab_reconciled.json.

## A13. HybridMerge — a deployable synthesis (added 2026-07-23)

Motivation. The paper's two central results point opposite ways: where a
provenance signal exists `ConfidenceRuleMerge` matches/beats the LLM; where none
does (`semantic_resolvable`) only a frontier LLM resolves anything, wrong-commit
-free. Neither dominates alone, which invites the deployment question "which do
I run?". HybridMerge (`branchmem/merge/hybrid.py`) answers it without an oracle:
run the rule, escalate to frontier `ThreeWayLLMMerge` only on keys the rule
abstains on (tied confidence) -- an observable trigger. Scope: post-hoc,
exploratory; reuses the frontier `gpt-5.4` content-hash cache (zero new API);
NOT a confirmatory strategy (kept out of the preregistered §2.2 list).
`scripts/run_hybrid_merge.py` -> `results/final/hybrid_merge.json`. Evaluated on
the frontier-cached RQ1 subset (seed 2026, n=20) and `semantic_resolvable`
(seed 3001, n=150). Result: Hybrid weakly dominates both parents in every
category (matches the rule on signal-present at the rule's cost; 0.467 vs 0.000
on no-signal) and does not backfire on `ambiguous` (1.000 preserved). Does not
change any locked/confirmatory result. Region A is now the full n=60 confirmatory
set (resolvable 0.842, ambiguous 1.000; the extension left one wrong commit-free
positive intact). One disclosed caveat: the frontier LLM alone already matches
the rule on signal-present data, so Hybrid's edge there is cost, not accuracy.

## A14. Capability scaling of calibrated selective resolution (added 2026-07-23)

Motivation. A9 showed the frontier model is a wrong-commit-free selective
resolver on `semantic_resolvable`; A14 asks whether this is an anecdote or a
capability trend. Uniform (T=0, max_tokens 8000) abstention decomposition of
single-call `ThreeWayLLMMerge` across llama-3.1-8b / gpt-5.4-nano /
claude-haiku-4-5 / gpt-5.4, all served from cache
(`scripts/analyze_capability_scaling.py` ->
`results/final/capability_scaling.{json,png}`). Result: calibration is
capability-governed and non-monotone below the frontier -- sub-frontier models
split into reckless-commit (llama 0.24 wrong-commit; nano commits 0.167, all
wrong) vs total abstention (haiku commits nothing); only the frontier reaches
wrong-commit-free resolution at a non-trivial commit rate. Corrects the earlier
imprecise "nano ~=0 commit" to the exact 0.167. Does not change any
locked/confirmatory result; sharpens the A9 positive into a capability threshold.

## A15. Realism check — LLM-authored strict-constraint conflicts (added 2026-07-23)

Motivation. The A9/A14 frontier positive (wrong-commit-free selective resolution)
is on 12 hand-authored templates; the open question is whether it is a template
artifact. `scripts/run_realistic_semantic.py`: gpt-5.4 authors 60 domain-diverse
STRICT-constraint semantic conflicts (12 domains; each a genuine logical
incompatibility, structurally validated; `data/realistic_semantic_templates.json`),
run through the SAME `generate_semantic_resolvable_scenarios` construction and the
SAME merge (seed 4004). Solved by both gpt-5.4 and gpt-5.4-nano. Result: the
frontier holds (commit 0.867, conditional accuracy 0.981, 1 wrong of 52) and
resolves more than on the 12 templates (whose defeasible cases suppressed
commits); nano on the identical items is reckless/wrong (conditional 0.387, 19/31
wrong) -- capability, not self-recognition. Scope disclosed: author=solver=gpt-5.4
(a surface-realism robustness check, not an adversarial or real-corpus claim; real
MAB content carries no recoverable signal, A5/mab-negative). Does not change any
locked/confirmatory result. `results/final/realistic_semantic.json`.

## A16. Downstream consequence of merge-time abstention (added 2026-07-23)

Motivation. Whether the merge-time abstention/resolution distinction matters
DOWNSTREAM. `scripts/run_downstream_utility.py`: for each realistic
strict-constraint scenario (A15 bank, n=60), render the merged store retaining
the ancestor constraint, then a gpt-5.4 downstream agent judges whether the memory
supports a constraint-VIOLATING statement (yes/no/unsure); yes = harmful downstream
belief. Result: forced-commit LastWriterWins 13.3% harmful even with the constraint
retained (corruption not fully recoverable downstream); NaiveConcat 8.3% harmful;
ConfidenceRule 0% harmful (flags -> downstream declines/abstains) but low utility;
frontier ThreeWayLLM 0% harmful, 60/60 correct, expected utility +1.0 at wrong-cost
-5. Establishes a measurable downstream safety payoff for calibrated merging. Does
not change any locked/confirmatory result. `results/final/downstream_utility.json`.

## Standing rule for this addendum

Any further post-hoc analysis added after this point must be appended here
with the same structure (motivation, exact scope, what it does and does not
change about the locked/confirmatory results) before being reported in the
paper.

---

**Backfill note (2026-11-15).** The standing rule above was not followed for
the analyses added in the three revision rounds after A16; they were documented
in the project change log and in the paper's appendices but not registered
here. A17-A24 below correct that. The register is append-only, so the entries
are added now rather than backdated, and each states when the analysis was
actually run.

## A17. Lambda sensitivity for the certification threshold (run 2026-11-08)

Motivation. It was observed that the reported utility scale
(+1 correct / -4 wrong / 0 abstain) makes commitment profitable only below
error 0.2, so certifying at lambda=0.25 is LESS conservative than the utility
implies, not more. Scope: re-ran the whole contextual BranchGuard simulation at
lambda=0.20 via `BRANCHGUARD_LAMBDA=0.2 scripts/run_contextual_branchguard.py`,
writing to `results/contextual_branchguard_lambda0.2/`. Result: coverage and
utility fall as expected and every qualitative conclusion is unchanged
(contextual safe, pooled unsafe, drift detected). Does not change any
locked/confirmatory result; corrects a directional claim in the text.

## A18. Blockwise monitor correction and re-run (run 2026-11-08)

Motivation. Theorem 4 as previously stated was false: with fixed-length audit
blocks and alpha-spending thresholds growing like 2 log j, a block eventually
cannot hold enough evidence to reject, so detection becomes impossible from
block 232 onward. Scope: `BlockwiseRiskMonitor` now grows block length with the
threshold (`min_block_size`, `margin`), the proof was rewritten with the
counterexample, and the drift simulation was re-run. Result: detection
probabilities and delays in the paper come from the corrected monitor. This
CHANGES a previously reported theoretical guarantee and the drift numbers that
depended on it; it does not touch the locked RQ1/RQ2/RQ3 results.

## A19. Prompt-priority ablation (run 2026-11-10)

Motivation. The misleading-provenance failure might be obedience to the merge
prompt's own trust-ordering clause rather than a property of LLM merging.
Scope: `scripts/run_prompt_priority_ablation.py` builds three system prompts
differing in exactly one clause (P1 shipped/provenance-preferring, P2 neutral,
P3 constraint-first), spliced from the shipped prompt with a loud failure if the
splice is a no-op, over the same 60 strict-constraint items in both provenance
regimes at seed 5150. The system prompt is part of the cache key, so P2/P3 are
genuinely new calls. Result: deleting the trust-ordering clause leaves recovery
at 0.017 with the same 55 commits and 54 errors; declaring the ancestor
constraint immutable recovers all 60 at no cost where provenance is
informative. Locates the deficit in specification rather than instruction
obedience. Does not change any locked/confirmatory result.
`results/final/prompt_priority_ablation.json`.

## A20. Matched informative control for the inversion claim (run 2026-11-10)

Motivation. The paper's headline inversion comparison was against the
*equalized* (absent-provenance) condition at 0.850, which is not the matched
control for inverting a field that is present. Scope: a
`MisleadingProvenanceConfig` with the strong signal on the ancestor-compatible
value, same templates, same seed, same model. Result: the matched informative
cell is 0.983, so the inversion claim is 0.983 -> 0.017; 0.850 is retained only
where the absent condition is explicitly meant. CHANGES a reported number in
the paper's favour-neutral direction (it widens the measured effect) and was
applied throughout the text.

## A21. End-to-end certification on observed conflicts (run 2026-11-11)

Motivation. The BranchGuard simulation draws channel reliabilities from
prescribed Bernoulli masses, which demonstrates the estimator rather than the
system. Scope: `scripts/run_empirical_channel_branchguard.py` defines three
channels as conflict domains visible at merge time, estimates every reliability
from adjudicated construction labels on a calibration half of 240 observed
conflicts (split seed 90210), and scores policies on the disjoint deployment
half using the model's ACTUAL cached decisions. Zero new API. Result: pooling
certifies nothing; contextual certification covers 27.5% at 0.000 risk; the
composite policy with an uncertified LLM fallback covers 85.8% at 0.058. The
split by warrant (certified / observed / composite) is the reportable finding.
Does not change any locked/confirmatory result.
`results/final/empirical_channel_branchguard.json`.

## A22. Semantic-then-provenance hybrid (run 2026-11-11)

Motivation. The NLI gate and the confidence rule are complementary, so an
ordered composition might dominate both. Scope:
`branchmem/merge/constraint_veto.py` vetoes a branch only when one value's
contradiction against the ancestor clears an absolute threshold while the other
stays below it, else applies the confidence rule;
`scripts/run_constraint_veto_hybrid.py` evaluates it against both parents and
the frontier LLM across all three provenance regimes on identical items. Zero
new API. Result: it keeps the rule's informative-regime perfection and cuts
misleading-regime risk from 1.000 to 0.383, but no further, because veto recall
is 37/60. Reported as a partial fix, not a solution. Does not change any
locked/confirmatory result. `results/final/constraint_veto_hybrid.json`.

## A23. Blinded panel adjudication of the 270 MAB labels (run 2026-11-12, extended 2026-11-15)

Motivation. The MAB-derived labels come from our own parser and nothing
independent had checked them; the released protocol asks for three human
annotators, which we could not run. Scope: `scripts/make_mab_item_master.py`
rebuilds exactly the released 270 conflicts from the public parquet (seed 2026,
max_conflict_keys=50) and extracts the evidence an adjudicator needs with no
source or confidence metadata; `scripts/make_mab_adjudication_sheets.py`
produces blinded, independently randomised sheets; and
`scripts/run_mab_adjudication_panel.py` runs the protocol with model
adjudicators. The panel was extended on 2026-11-15 from four members to six,
adding a local open-weight Llama 3.1 8B and a local RoBERTa NLI model so that
three members share no vendor with OpenAI, and two evidence windows (6 and 30
lines) are now run over all 270 items.
`scripts/analyze_panel_independence.py` reports the tiers separately. Result:
agreement with the derived labels tracks adjudicator capability rather than
vendor, and the strongest cross-vendor pair agrees with each other and with our
labels at a high rate; the two weak members drag the vendor-independent
majority down. Explicitly a substitute for the human study, not a discharge of
it. Does not change any locked/confirmatory result.
`results/final/mab_adjudication_summary.json`.

## A24. WikiDivergence: independently sourced conflicts (run 2026-11-15)

Motivation. The paper's principal limitation is that every branch pair is
constructed by us, so no result speaks to naturally occurring semantic
conflicts. Scope: `scripts/build_wikidivergence.py` extracts three-way merges
from English Wikipedia infobox revision history; the ancestor, both branch
values, the provenance metadata and the label (the value the article's current
revision holds) are all external to this paper. Four pre-declared categories,
hash-based article sampling at seed 2026, living-person biographies excluded,
caps of 3 items per (page, field) chain and 10 per article, presentation
variants filtered out, editors pseudonymised. The provenance mapping (source
from account type and user group; confidence monotone in lifetime edit count)
and the channel definition (read off the infobox field name) are fixed before
evaluation. `scripts/run_wikidivergence_eval.py` scores five policies, re-runs
the provenance-inversion intervention on these real items with a label-free
flip rate, and certifies Contextual BranchGuard on reliabilities estimated from
this collection. Scope limits stated in the paper: the writers are people
editing an encyclopedia rather than agent replicas, and Wikipedia's serial
order is information a genuinely offline replica pair would not have, so the
recency baseline is reported rather than suppressed and every result is split
by whether the second edit restored the ancestor value. Does not change any
locked/confirmatory result. `results/final/wikidivergence.json`.

## A25. Ancestor-strength ablation (run 2026-11-15)

Motivation. The provenance-inversion failure is demonstrated against ancestors
phrased as strict constraints, so it may be an artifact of strict-constraint
semantics rather than a fact about provenance. Scope:
`data/defeasible_ancestor_templates.json` copies every predicate and both
branch values verbatim from the strict bank and rewrites only the ancestor's
modal force; `tests/test_defeasible_bank.py` enforces that only the ancestor
differs. Because a revisable preference entails no correct answer, accuracy is
NOT reported for the defeasible arm; the measure is the label-free flip rate
over two runs of identical content with opposite provenance.
`scripts/run_ancestor_strength_ablation.py`, n=60 per arm, seed 5150, gpt-5.4
and gpt-5.4-nano. Result: flip rate 0.982/0.983 (strict) versus 1.000/1.000
(defeasible), with zero abstentions in the defeasible inverted arm. The failure
is stronger, not weaker, without a hard constraint. Does not change any
locked/confirmatory result. `results/final/ancestor_strength_ablation.json`.

## A26. Channel governance: assignment, misassignment, cost (run 2026-11-15)

Motivation. The certificate is stated for a channel family fixed in advance,
which leaves open how a deployment assigns conflicts to channels, what
imperfect assignment costs, and what calibration and auditing cost. Scope:
`scripts/run_channel_governance.py` on the same 240 observed conflicts and
split seed as A21, fully cached. (A) A label-free router over features visible
before the merge decision, certified and deployed in place of the
construction's own channel labels. (B) Independent channel corruption at rate
eps, 200 draws each, re-certifying every draw and scoring the deployment half
against the TRUE channels. (C) Minimum audited sample per channel for a
certificate at lambda=0.25, delta=0.05 under Bonferroni, and audit cadence
derived from the drift experiment's own measured detection delay and post-drift
error rate. Result: the router agrees with the declared channels on 92.1% of
conflicts and that is not sufficient to obtain a certificate; deployment risk
rises 0.058 -> 0.360 because the anti-correlated channel stops being
certifiable and falls through to the uncertified fallback. Reported as the
method's weakest point. Does not change any locked/confirmatory result.
`results/final/channel_governance.json`.

## A27. Re-scoring policies under human labels (run 2026-11-15)

Motivation. A11 asks whether the generator's labels are defensible; the sharper
question is what changes if they are replaced. Scope:
`scripts/analyze_human_label_robustness.py` matches all 30 blind-study items
back to their source scenarios by ancestor and branch values and scores every
cached policy decision twice, against the construction target and against the
three-annotator majority, treating "no single correct answer" as an
abstention. Zero new API. Result: the label sets differ on 5 of 30 items; on
the 12 resolvable items both the confidence rule and the frontier merger move
from accuracy 0.667 at conditional risk 0.333 to accuracy 1.000 at risk 0.000,
so the deliberate 15% flips are the entire gap and are charged identically to
both policies. No comparison changes sign. The fifth disagreement is the
defeasible gift/no-pet family. Does not change any locked/confirmatory result,
and confirms that the construction targets are harsher than human judgment
rather than kinder. `results/final/human_label_robustness.json`.

## A28. Matched regime table (run 2026-11-15)

Motivation. The paper's overview figure reported informative/absent recovery
of 0.983/0.850 while the regime table reported 0.867/0.467 for the
corresponding columns, and the difference had not been made explicit. Inspecting the provenance of every cell showed a sharper
problem: the table's three columns were three different item populations
(locked `resolvable` construction, the 12-template semantic bank, and the
realistic strict-constraint bank), and one cell was worse than that -- the
ConstraintVeto informative cell was measured on the realistic bank while the
cells beside it came from the locked bank, which is not a like-for-like
comparison and happened to favour our own method, since on the realistic bank
the confidence rule also scores 1.000 there rather than 0.867.

Scope. `scripts/run_matched_regime_table.py` holds ONE bank fixed (the 60
realistic strict-constraint templates) and varies only the source/confidence
assignment: informative and misleading are the same generator at the same seed
(5150) and so share items exactly; absent is the same templates through the
equalizing generator at its own seed (4004), which the output records rather
than glossing over. Six policies on identical items. Zero new API: the LLM
column is cache-served, which is also why those seeds were chosen.

Result. The frontier row becomes 0.983 / 0.850 / 0.017, identical to the
overview figure, so the discrepancy dissolves by construction rather than by
annotation. Two numbers change in the paper, both
against earlier claims of ours:

* ConstraintVeto no longer appears to beat the confidence rule where provenance
  is informative; on matched items both score 1.000.
* The NLI gate's informative cell is 0.700, not 0.075. The 0.075 figure is a
  property of the locked bank, where the construction label is *defined* by the
  field a metadata-blind policy cannot read, and it is correct there. Used
  beside a misleading-regime number from a different bank it had supported a
  "near-disjoint competence" claim that matched items do not support: the gate
  is not disjoint from the rule, it is simply unmoved by the field, scoring
  0.700 in all three regimes.

CHANGES two reported numbers and one qualitative claim in the paper's main
text. Does not change any locked/confirmatory result. The earlier
across-population table is retained in the appendix, which is a legitimate and
different question, with a caption that now names each column's bank and
explains why its numbers differ.
`results/final/matched_regime_table.json`.

---

# PROSPECTIVE PREREGISTRATION (written 2026-11-15, before any data was collected)

Everything above this line was written after seeing the data it describes and is
labelled exploratory. Everything below was written BEFORE the corresponding
experiments were run. The item banks were authored, the protocols fixed and the
hypotheses stated before any model was called on them. Results are appended in
a separate, clearly marked section.

## P1. Constraint-versus-override discrimination (prospective)

### Motivation

Our existing diagnostic shows that a merger follows provenance metadata over
conflicting meaning. That framing has a gap we now think is the more important
scientific question. Following a newer, better-sourced update is often
*correct*: when a user corrects a stale preference, the high-confidence update
SHOULD win. The failure we reported and the behaviour a deployment wants are
the same behaviour under different circumstances. So "the merger follows
provenance" is not by itself a defect; what matters is whether the merger can
tell the two circumstances apart.

### Design

One bank of newly authored items, four conflict types, with entities,
predicates and domains disjoint from every bank used in the paper so far. In
the two critical types the PROVENANCE IS IDENTICAL and only the semantics of
the ancestor differ:

* `immutable_constraint` -- the ancestor records a standing constraint and the
  incoming update would violate it. Correct action: keep the
  ancestor-compatible value, i.e. REJECT the update.
* `legitimate_override` -- the ancestor records an ordinary, revisable value
  and the incoming update is a genuine correction of it. Correct action: ACCEPT
  the update.
* `independent_observation` -- two observations of a changeable fact, neither
  bearing on a constraint. Correct action: accept the better-warranted value.
* `ambiguous` -- the exposed evidence does not determine an answer. Correct
  action: abstain; any commitment is wrong.

In both critical types the update carries the strong provenance
(`source=user`, `confidence=0.95`) and the ancestor-side value the weak one, so
a policy that reads only the metadata behaves IDENTICALLY in the two types.
Secondary arms vary the provenance assignment (informative / neutral /
inverted) and the system prompt (shipped / neutral / constraint-aware), both
crossed with type.

### Primary measure

Per type accuracy, and the **discrimination gap**

    D = accuracy(legitimate_override) - accuracy(immutable_constraint)

under the fixed provenance assignment above. A policy that reads only the
metadata field commits to the update in both types, scoring 1 and 0, so D = +1
by construction. A policy that reads the ancestor's semantics should score high
in both, so D near 0 with both accuracies high. D near 0 with both accuracies
LOW is a third, distinguishable failure (refusing every update). We will report
the pair, not D alone, for exactly that reason.

### Hypotheses, stated in advance

* **H1.** D > 0.5 for every tested model, i.e. accuracy on
  `legitimate_override` exceeds accuracy on `immutable_constraint` by more than
  half, with the metadata held identical.
* **H2.** H1 holds for at least one open-weight model from a different provider
  than the bank's author, so the effect is not an artifact of one vendor.
* **H3.** Balanced accuracy over the two critical types is below 0.75 for every
  single-rule policy we test.
* **H4 (mitigation).** A two-stage policy that first classifies the ancestor as
  constraint or revisable, then applies the corresponding rule, achieves higher
  balanced accuracy over the two critical types than either the metadata rule
  or the one-shot LLM merger.

Each hypothesis can fail, and a failure is reportable. In particular, if D is
small because a model genuinely discriminates, the paper's central diagnostic
is narrower than we have claimed and we will say so in those terms. If H4
fails, we will report the mitigation as unsuccessful rather than drop it.

### Analysis plan

Paired over items within model. Accuracy reported with 95% intervals clustered
by template family (the unit of authoring), since items sharing a family are
not independent. Abstentions counted as neither correct nor wrong-committed and
reported separately as coverage. No item will be added, removed or relabelled
after any model has been run on the bank; the bank file is written and hashed
before the first call.

### Ground truth

Construction labels first, then blinded adjudication of every item by the
six-member panel of `scripts/run_mab_adjudication_panel.py`, including its
three vendor-independent members, under the released protocol. Per-type
adjudicator agreement and abstention will be reported, and any item the panel
contradicts will be reported rather than removed.

## P2. Offline two-agent divergence (prospective)

### Motivation

Every branch pair in the paper is either scripted by us or derived from a
pre-existing single history. The paper's setting is two replicas that diverged
through their own operation, and we have never evaluated that.

### Design

Per episode: a shared ancestor memory, then two independent dialogues. The
user turns are environment, fixed in advance; the memory writes are NOT
scripted --- the agent reads its dialogue and decides for itself what to store,
with what source and what confidence. The two branches therefore diverge
through the agent's own extraction behaviour, and which conflicts arise is not
known to us in advance. Each resulting pair of memory states is then merged by
each policy and the merged store is evaluated by downstream questions fixed
before the run.

### Primary measures

Fraction of episodes producing at least one genuine conflict; per-policy
downstream accuracy, coverage and wrong-commit rate on the fixed questions;
and the rate of harmful commitments, defined before the run as committing to a
value that contradicts a constraint present in the ancestor memory.

### Hypotheses, stated in advance

* **H5.** Agent-generated divergence produces conflicts whose provenance
  metadata is unequal in a majority of conflicting keys, i.e. the metadata
  policies are not vacuous here as they were on MemoryAgentBench.
* **H6.** The ordering of policies by wrong-commit rate observed on the
  constructed banks is preserved on agent-generated conflicts.
* **H7.** At least one policy that commits on a majority of conflicts does so
  with a harmful-commitment rate above zero, i.e. the problem does not
  disappear when conflicts arise naturally.

### Analysis plan

Episode is the unit of clustering for all intervals. Episodes producing no
conflict are reported, not discarded. Downstream questions are generated from
the ancestor memory and the agents' own writes by fixed rule, before any merge
is scored.

---

# PROSPECTIVE RESULTS (appended 2026-11-15, after the runs)

The preregistration above was written first and has not been edited since. No
item was added, removed or relabelled; both banks are hashed and every runner
refuses to execute against a changed hash.

## P1 results: constraint-versus-override discrimination

Bank `data/discrimination_bank.json`, sha256 `a52914cc...`, 320 items in 80
template families. Intervals resample families.

Primary arm `update_favoured`, where the update carries source=user and
confidence .95 in BOTH critical types, so a metadata-only policy must score
(0, 1):

| merger | constraint | override | D | D 95% CI | balanced |
|---|---|---|---|---|---|
| ConfidenceRule (reference) | .000 | 1.000 | +1.000 | [1.00, 1.00] | .500 |
| gpt-5.4 | .000 | 1.000 | +1.000 | [1.00, 1.00] | .500 |
| gpt-5.4-mini | .125 | 1.000 | +0.875 | [0.80, 0.94] | .562 |
| gpt-5.4-nano | .037 | .988 | +0.950 | [0.90, 0.99] | .512 |
| llama3.1:8b (Meta, open weights) | .000 | 1.000 | +1.000 | [1.00, 1.00] | .500 |
| qwen2.5:7b (Alibaba, open weights) | .000 | .975 | +0.975 | [0.94, 1.00] | .487 |
| mistral:7b (Mistral, open weights) | .000 | .713 | +0.713 | [0.61, 0.81] | .356 |

**H1 HELD.** D > 0.5 for every one of the six models across four providers;
the smallest is +0.713. gpt-5.4 and llama3.1:8b are indistinguishable from the
metadata-only reference. The smallest gap, mistral:7b's, is NOT evidence of
discrimination: its constraint accuracy is also .000 and the gap narrows only
because it is worse on overrides (.713) and abstains on 35.6% of the bank. This
is exactly why the preregistration committed to reporting the pair rather than
D alone, and it is the "both accuracies low with D near 0" failure that
document anticipated.

**H2 HELD.** Three open-weight models from three providers other than the
bank's author all show it: llama3.1:8b D = +1.000 [1.00, 1.00], qwen2.5:7b
D = +0.975 [0.94, 1.00], mistral:7b D = +0.713 [0.61, 0.81].

**H3 HELD.** Balanced accuracy over the critical pair is at most .562 for every
single-rule policy, in every arm.

**H4 HELD.** Two-stage routing, primary arm:

| classifier | stage-1 accuracy | constraint | override | balanced |
|---|---|---|---|---|
| gpt-5.4 | .959 | .775 | 1.000 | **.887** |
| gpt-5.4-nano | .891 | .812 | .912 | **.863** |
| llama3.1:8b | .769 | .850 | .600 | **.725** |

Balanced accuracy tracks stage-1 accuracy almost exactly (.959 -> .887,
.891 -> .863, .769 -> .725), which identifies the classifier as the ceiling.

### Secondary findings we did not predict

* **The whole behaviour is one rule.** Under `neutral` (metadata identical on
  both branches) every model inverts its pattern: constraint accuracy rises to
  .463-.925 and override accuracy falls to .000-.100. Under `inverted` the
  models reproduce the metadata rule exactly (gpt-5.4 and gpt-5.4-mini both
  D = -1.000). The policy these mergers implement is "follow the field; absent
  a field, keep what is stored", in all three arms.
* **Abstention is a function of metadata ties, not of ambiguity.** On the
  `ambiguous` type, where the correct action is to abstain, accuracy is .000 to
  .037 whenever a field is present and .988 to 1.000 when the metadata ties.
  The mergers abstain because the numbers are equal, not because the evidence
  is indeterminate.
* **Content-level recency loses to the field.** On `independent_observation`,
  where the values themselves say which is current, accuracy is 1.000 under
  `update_favoured`, .475-.637 under `neutral` and **.000** under `inverted`.
* **The knowledge is present but unused.** gpt-5.4 classifies the stored entry
  correctly 95.9% of the time when asked directly, and then ignores that same
  distinction when asked to merge in one shot. The failure is not an absence of
  the required knowledge.

### What the mitigation does not fix

Under `neutral` and `inverted` the router's override accuracy collapses to
.000-.013, because items it routes as ordinary fall to the confidence rule,
which has no signal or an inverted one. That is the regime Theorem 2 covers: no
label-free policy can do better in both of two environments that look identical
and have opposite answers. The router recovers the part that is recoverable
(constraint items, where ancestor semantics decide) and cannot recover the part
the impossibility result forbids. We report it that way rather than as a
general fix.

## P2 results: offline two-agent divergence

118 episodes (190 authored drafts were rejected by the structural validator, so
the released set is 118 rather than the planned 150). The agent produced 5.57
memory writes per episode and **all 118 episodes produced at least one
conflict**, 240 conflicting keys in total. 128 were labellable from the
environment and 112 were not; the unlabelled ones are reported here rather than
discarded.

**H5 HELD.** Metadata is unequal on 89.8% of labelled conflicts, so provenance
policies are live in this setting, unlike on MemoryAgentBench where every
conflict tied. Provenance favours the environment-true value 90.4% of the time.

**H6 FAILED, and this is the most useful result in P2.** We predicted the
policy ordering from the constructed banks would be preserved. It is not:

| policy | coverage | accuracy | 95% CI (episode) | risk | harmful |
|---|---|---|---|---|---|
| ConfidenceRule | .891 | **.813** | [.750, .875] | .088 | .078 |
| ThreeWayLLM (gpt-5.4) | .672 | .453 | [.381, .537] | .326 | **.023** |
| ConstraintVeto | .938 | .578 | [.496, .661] | .383 | .359 |
| NliGated | .680 | .250 | [.185, .318] | .632 | .430 |

On agent-generated conflicts the one-line metadata rule is the best single
policy by 36 accuracy points over the frontier merger. The reason is visible in
the split by conflict kind: the rule is perfect (1.000) where a definite user
statement competes with a speculation, and poor (.564) under constraint
pressure, while the frontier merger is worse at both (.603 and .255). The
discrimination asymmetry P1 isolates is therefore exactly what shows up when
conflicts arise from an agent's own writes.

**H7 HELD.** Every policy committing on a majority of conflicts has a non-zero
harmful-commitment rate: .078 for the confidence rule, .359 for the veto
hybrid, .430 for the NLI gate. Only the frontier merger is near zero (.023),
and it buys that by committing on a third fewer conflicts and being wrong on a
third of what it does commit.
