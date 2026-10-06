# Engineering contract

The rules for any change to this repository, by a person or an agent.
[`AGENTS.md`](../AGENTS.md) is the short normative summary.

MUST / MUST NOT are mandatory; SHOULD / SHOULD NOT hold unless there is a
stated reason; MAY is optional. A request that conflicts with this contract
is reported before any work.

The contract grows with the project. When a recurring decision or an error
that could recur comes up, ask whether it is a rule of one experiment or of
the project; if of the project, add it here.

## 1. Scope discipline

- Change only what the current task requires. No unrelated refactors or
  opportunistic clean-ups.
- Preserve unrelated uncommitted changes. Do not modify user-owned
  notebooks, experiments or configs unless asked.
- Prefer small, reviewable changes over broad rewrites.

## 2. Git discipline

- Never commit or push without explicit approval.
- Show the exact staged diff before committing.
- Stage selectively; separate logically independent changes into separate
  commits; never include unrelated modifications.
- Do not rewrite history unless explicitly requested.

## 3. Scientific purpose

Every metric, figure, analysis function, experiment or diagnostic MUST
answer a stated question:

    Question
        ↓
    Diagnostic / experiment
        ↓
    Possible outcomes
        ↓
    Interpretation
        ↓
    Decision it can influence

If something does not fit this chain, it should normally not be added. No
metric proliferation, no decorative figures.

## 4. Evaluation integrity

- The test split stays closed until explicitly authorized. Validation is for
  analysis and iteration.
- Do not tune on validation results after evaluation has begun unless the
  protocol allows it.
- Paired comparisons use identical anchors/rows; prefer paired statistics
  for two conditions on the same samples.
- Report unsupported settings; never silently drop classes, rows, datasets
  or conditions to make an experiment succeed.

## 5. Reproducibility and provenance

Artifacts SHOULD record enough to recover how they were made: run ID,
checkpoint and its hash, config hash, dataset and feature-cache revisions,
Git commit, split, sampling method, seed, sample count, device/precision
when relevant, and the evaluation settings.

Every randomized operation MUST use an explicit seed. Nothing may depend on
notebook or implicit global state.

## 6. Artifact-first design

    compute once
        ↓
    persist artifacts
        ↓
    inspect repeatedly

Downstream stages SHOULD read versioned artifacts rather than recompute
upstream work. An analysis typically writes `summary.json`,
`scores.parquet` (or another table) when tabular results help, `report.md`
and `figures/`. Avoid redundant copies of the same information.

## 7. Offline command UX

A long-running command MUST NOT appear frozen when meaningful progress or
stage information can reasonably be exposed.

- Progress bars over meaningful units (batches, samples, recordings,
  bootstrap iterations, files). No noisy nested bars.
- Stage logs, when applicable: run/checkpoint, input artifact, split,
  device, sample count/limit, seed, output path, expensive stage
  transitions, skipped/unsupported cases, integrity checks, elapsed time.
- No per-sample logging.
- A successful command prints its final output path.

In this repository, bars and logs go through `turn_wm.progress` (`progress`,
`log`): tqdm on stderr, so stdout stays usable; `TURN_WM_PROGRESS=0` turns
them off. They never change a result.

## 8. show_<analysis> contract

A persisted analysis block whose results are meant to be reread provides
`show_<analysis>(path)`.

It MUST NOT reload the model, rerun inference or recompute an expensive
analysis.

It SHOULD load existing artifacts only, show the important figures, expose
the main results and the report, surface missing or unsupported results,
and work in notebook/Colab contexts.

No analysis code currently lives in this repository (removed 2026-10-06,
recoverable from commit `bf9e24e`); a new analysis block follows this
contract.

## 9. Failure policy

Fail early, with an error that says what is wrong, on anything that can
invalidate an experiment: wrong checkpoint, incompatible dataset revision,
train/validation leakage, mismatched anchors, incompatible representation
hashes, missing canonical classes, impossible action-sequence assumptions,
unexpected horizon semantics.

A fallback is allowed only when it preserves the scientific meaning of the
experiment and is documented.

## 10. Testing philosophy

Tests protect meaningful contracts, not implementation trivia: scientific
invariants, leakage prevention, deterministic sampling, artifact schemas,
critical boundaries, CLI behavior, correctness of comparisons. Avoid
redundant tests that mirror the implementation.

A new evaluation block SHOULD have at least one end-to-end synthetic smoke
test of its artifact generation.

## 11. Code simplicity / no overengineering

- The simplest implementation that works. No speculative abstractions or
  frameworks before several concrete uses justify them.
- Reuse existing project abstractions; do not duplicate rollout, data or
  model logic.
- Remove dead code created by the change. Every function has a reason to
  exist.

## 12. Data/action semantics

Do not silently reinterpret dataset fields. Keep observed events,
observational action proxies, controllable agent actions, labels, masks and
padding distinct.

The vocal action space is NO_EVENT / ONSET / OFFSET: observational
vocal-action/event proxies, not controllable actions. MASKED and PAD are
not semantic actions. No causal claim follows from these tokens without
interventional evidence.

## 13. Claims vs evidence

A claim MUST NOT be stronger than the evaluation supporting it.

- A linear probe that succeeds supports "the information is linearly
  accessible", not "the predictor uses it".
- An action ablation may support "the predictor uses the conditioning
  channel", not "the action is a causal intervention".
- Good rollout prediction does not establish planning utility; that needs a
  planning-grounded evaluation.

## 14. Scientific reports

A report SHOULD separate observation, evidence, interpretation, alternative
explanations, limitations, and the resulting decision or open question.
Hypotheses are not conclusions. Negative or null results of a predefined
experiment are reported.

## 15. New-stage checklist

Before implementing a new analysis stage, define:

1. scientific question
2. inputs
3. outputs
4. invariants
5. metrics and their roles
6. possible outcomes
7. failure modes
8. artifact schema
9. `show_*` viewer, if applicable
10. tests
11. CLI entry point, if appropriate

Do not implement the stage first and invent its purpose afterwards.

## 16. turn-wm-specific rules

- `features` is the upstream Mimi representation; `latent` is the learned
  WM projector output. Do not call z_t temporally contextualized unless the
  architecture makes it so.
- Autoregressive rollout feeds predicted latents back into later steps; it
  never silently consumes future ground-truth latents.
- Never invent an unavailable checkpoint: an unsaved step is not a
  recoverable checkpoint.
- Mimi-vs-WM comparisons use matched samples, labels, splits, readout
  family and metric.
- Dataset-specific behavior (EgoCom, Ego4D) stays out of core abstractions
  when avoidable, in adapters or clear boundaries.
- Do not add an evaluation because it is common in the literature; it must
  address an observed result, a design hypothesis, a benchmark requirement
  or an explicit research question.
