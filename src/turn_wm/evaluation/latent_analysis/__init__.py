"""Representations of trained world models, extracted for offline analysis.

- `extract` / `run`: write a run's representations to a snapshot
  (`turn-wm extract-latents`);
- `spectrum`, `pca` and `analyze`: analyze a snapshot
  (`turn-wm analyze-latents`);
- `show`: display written results in a notebook.

Nothing is imported here, so `show` can be imported from a notebook kernel
without the training dependencies.
"""
