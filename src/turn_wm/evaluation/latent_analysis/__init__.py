"""Representations of trained world models, extracted for offline analysis.

Snapshots:

- `extract` / `run`: write a run's representations to a snapshot
  (`turn-wm extract-latents`); `rollout` and `action_ablation` write
  rollout snapshots (`extract-rollouts`, `extract-action-ablation`);
- `snapshot`: read a snapshot back.

Analyses, each a computation module and its `*_report` module (files,
figures, report):

- `spectrum`, `pca`, `label_structure` (+ `label_source`), orchestrated by
  `analyze` (`turn-wm analyze-latents`);
- `rollout_dynamics` (`analyze-rollouts`);
- `action_ablation_analysis` (`analyze-action-ablation`);
- `probes` (+ `linear_probe`, the estimator) (`probe-latents`);
- `concepts` (+ `concept_labels`, the concept rules) (`probe-concepts`).

Shared: `artifacts` (provenance, output checks), `rendering` (figure
style), `seeding`. `show` displays written reports and figures in a notebook.

Nothing is imported here, so `show` can be imported from a notebook kernel
without the training dependencies.
"""
