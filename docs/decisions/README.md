# Decision log

This directory records project-level decisions that must remain stable across
experiments. It separates accepted decisions from hypotheses and diagnostics so
that a downstream benchmark cannot silently redefine the research objective.

| ADR | Status | Decision |
|---|---|---|
| [0001](0001-v1-freeze.md) | Frozen | V1 scientific baseline, checkpoint, evidence and limitations |
| [0002](0002-v2-hierarchical-planning.md) | Accepted | V2 hierarchical fast/slow world model and receding-horizon planning |
| [0003](0003-v2-causal-actions.md) | Accepted | Separate observed conversation events from causal agent actions |
| [0004](0004-v2-evaluation-contract.md) | Accepted | Evaluation contract for representation, dynamics, action use and planning |

## Version boundary

V1 is the audio baseline trained in run `20260926-010750-4216838a`.
Its model definition and selected checkpoint are frozen. New architectural,
temporal or planning changes belong to V2.

The repository may still receive correctness fixes to evaluation tooling, but
such fixes must not silently redefine the V1 model or training recipe. Every
reported result must retain the model checkpoint identity and evaluation-code
revision.

V2 work starts from this decision boundary and is planning-first: TurnBench and
other downstream tasks are diagnostics of representation quality, not the
project objective.
