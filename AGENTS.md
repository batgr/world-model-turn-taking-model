# Agent contract

Normative rules for any coding agent working on this repository. MUST / MUST
NOT are mandatory; SHOULD / SHOULD NOT hold unless there is a stated reason;
MAY is optional. If a request conflicts with these rules, report the
conflict before proceeding.

You MUST read [`docs/engineering_contract.md`](docs/engineering_contract.md)
before substantial work. It expands every rule below.

## Rules

- You MUST change only what the current task requires.
- You MUST preserve unrelated user changes (notebooks, configs, experiments,
  uncommitted work) and MUST NOT include them in a commit.
- You MUST NOT commit or push without explicit approval.
- You MUST show the staged diff before committing, and SHOULD stage
  selectively, one logical change per commit.
- A scientific addition MUST have an explicit purpose: a question it answers
  and a decision it can influence.
- You MUST NOT add metrics, plots or functions without a reason.
- A long-running offline command MUST expose progress and stage information.
- A persisted analysis block MUST produce inspectable artifacts and SHOULD
  provide a `show_<analysis>()` viewer when its results are meant to be
  reread.
- Claims MUST NOT exceed the evidence that supports them.
- You MUST fail loudly on scientific-integrity violations (leakage,
  mismatched anchors or checkpoints, wrong split, missing classes).
- The test split MUST stay closed unless explicitly authorized.
- You SHOULD prefer simple, existing abstractions over speculative
  frameworks.

## Task protocol

```
read the contract → inspect the existing implementation → state what will
change → implement the minimal change → test → report (changed, not
changed, tests, unresolved issues, unrelated working-tree changes) → show
the staged diff → wait for commit approval
```
