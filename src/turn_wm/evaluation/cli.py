"""Offline representation and rollout analysis commands."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

from turn_wm.evaluation.latent_analysis.action_ablation import (
    DEFAULT_ABLATION_SAMPLES,
    extract_action_ablation_run,
)
from turn_wm.evaluation.latent_analysis.action_ablation_report import (
    write_action_ablation,
)
from turn_wm.evaluation.latent_analysis.analyze import (
    ANALYSES,
    DEFAULT_ANALYSES,
    LABELS,
    PCA,
    SPECTRUM,
    analyze_snapshot,
)
from turn_wm.evaluation.latent_analysis.concepts_report import write_concepts
from turn_wm.evaluation.latent_analysis.label_structure import DEFAULT_BALANCED_CAP
from turn_wm.evaluation.latent_analysis.pca import (
    DEFAULT_MAX_PLOT_SAMPLES,
    DEFAULT_SILHOUETTE_SAMPLES,
)
from turn_wm.evaluation.latent_analysis.probes_report import write_probes
from turn_wm.evaluation.latent_analysis.rollout import (
    DEFAULT_ROLLOUT_SAMPLES,
    extract_rollout_run,
)
from turn_wm.evaluation.latent_analysis.rollout_dynamics import (
    DEFAULT_BOOTSTRAP,
    DEFAULT_TRAJECTORIES,
)
from turn_wm.evaluation.latent_analysis.rollout_dynamics_report import (
    write_rollout_dynamics,
)
from turn_wm.evaluation.latent_analysis.run import (
    DEFAULT_CHECKPOINT,
    DEFAULT_SPLIT,
    extract_run,
)
from turn_wm.evaluation.latent_analysis.show import (
    show_action_ablation,
    show_concepts,
    show_labels,
    show_pca,
    show_probes,
    show_rollouts,
    show_spectrum,
)

SPLITS = ("train", "validation", "test")


def add_analysis_commands(commands: argparse._SubParsersAction) -> None:
    _add_extract_latents(commands)
    _add_extract_rollouts(commands)
    _add_analyze_rollouts(commands)
    _add_extract_action_ablation(commands)
    _add_analyze_action_ablation(commands)
    _add_probe_latents(commands)
    _add_probe_concepts(commands)
    _add_analyze_latents(commands)


def _add_run_options(
    command: argparse.ArgumentParser,
    *,
    max_samples: int | None,
    output_default: str,
    split: bool = False,
) -> None:
    """Arguments shared by commands that rerun a trained checkpoint."""

    command.add_argument(
        "run_dir",
        type=Path,
        help="Run directory holding config.yaml, metadata.json and checkpoints/.",
    )
    command.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
        help="File under checkpoints/, or a path (default: %(default)s).",
    )
    if split:
        command.add_argument(
            "--split",
            choices=SPLITS,
            default=DEFAULT_SPLIT,
            help="Split to extract (default: %(default)s).",
        )
    command.add_argument(
        "--max-samples",
        type=_positive_int,
        default=max_samples,
        help=(
            "Samples to keep from the seeded order (default: all)."
            if max_samples is None
            else "Anchors to keep from the seeded order (default: %(default)s)."
        ),
    )
    command.add_argument(
        "--seed", type=int, help="Seed of the sample order (default: the run's seed)."
    )
    command.add_argument(
        "--batch-size",
        type=_positive_int,
        help="Batch size; does not change the samples (default: the run's).",
    )
    command.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Data loader workers (default: %(default)s).",
    )
    command.add_argument(
        "--device",
        default="cpu",
        help="Torch device, e.g. cpu, cuda, mps (default: %(default)s).",
    )
    command.add_argument(
        "--mimi-cache-root",
        type=Path,
        help="Where the run's Mimi cache now lives, if it moved.",
    )
    command.add_argument(
        "--output",
        type=Path,
        help=(
            "Directory to create; must not exist or be empty (default: "
            f"{output_default})."
        ),
    )


def _add_report_options(
    command: argparse.ArgumentParser,
    *,
    output_default: str,
    labels: bool = False,
) -> None:
    command.add_argument(
        "--output",
        type=Path,
        help=f"Directory to create (default: {output_default}).",
    )
    if labels:
        command.add_argument(
            "--labels-revision",
            help=(
                "Read label sidecars at this dataset revision instead of the "
                "snapshot's; its action grid must be byte-identical."
            ),
        )
    command.add_argument(
        "--bootstrap",
        type=_positive_int,
        default=DEFAULT_BOOTSTRAP,
        help="Bootstrap resamples per interval (default: %(default)s).",
    )
    command.add_argument(
        "--show",
        action="store_true",
        help="Then show the persisted report and figures. Results are unchanged.",
    )


def _add_probe_options(
    command: argparse.ArgumentParser, *, output_default: str
) -> None:
    command.add_argument(
        "train_snapshot",
        type=Path,
        help="Train-split snapshot (extract-latents --split train).",
    )
    command.add_argument(
        "validation_snapshot",
        type=Path,
        help="Validation-split snapshot of the same checkpoint.",
    )
    _add_report_options(
        command,
        output_default=output_default,
        labels=True,
    )


def _add_extract_latents(commands: argparse._SubParsersAction) -> None:
    extract = commands.add_parser(
        "extract-latents",
        help="Extract a trained run's representations for analysis.",
        description=(
            "Rebuild a training run's data (config, dataset revision, "
            "observation source, window) and write the encoder features and "
            "projected latents at each anchor of a seeded sample of one split."
        ),
    )
    _add_run_options(
        extract,
        max_samples=None,
        output_default="RUN_DIR/latent_analysis/<checkpoint>-<split>",
        split=True,
    )
    extract.set_defaults(handler=_extract_latents)


def _extract_latents(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> int:
    try:
        output = extract_run(
            args.run_dir,
            output_dir=args.output,
            checkpoint=args.checkpoint,
            split=args.split,
            max_samples=args.max_samples,
            seed=args.seed,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=args.device,
            mimi_cache_root=args.mimi_cache_root,
        )
    except (ValueError, FileNotFoundError) as error:
        # Not a run directory, edited config, other data revision or cache,
        # missing checkpoint or media, or a non-empty output directory.
        raise SystemExit(f"turn-wm: error: {error}") from error

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))

    print(f"Representations written to {output}")
    print(f"samples: {manifest['samples']}")

    for name, spec in manifest["representations"].items():
        print(f"{name}: {spec['shape']}")

    return 0


def _add_analyze_latents(commands: argparse._SubParsersAction) -> None:
    analyze = commands.add_parser(
        "analyze-latents",
        help="Analyze an extracted representation snapshot.",
        description=(
            "Analyze every representation of a snapshot written by "
            "extract-latents, globally and per corpus. Reads the snapshot "
            "only: no checkpoint, dataset or cache."
        ),
    )
    analyze.add_argument(
        "snapshot",
        type=Path,
        help="Snapshot directory holding representations.safetensors.",
    )
    analyze.add_argument(
        "--analysis",
        action="append",
        choices=ANALYSES,
        help=(
            "Analysis to run, repeatable (default: spectrum and pca; labels "
            "reads the Hub and runs only when named)."
        ),
    )
    analyze.add_argument(
        "--output",
        type=Path,
        help=(
            "Root of the results, one directory per analysis "
            "(default: SNAPSHOT/analysis)."
        ),
    )
    analyze.add_argument(
        "--silhouette-samples",
        type=_positive_int,
        default=DEFAULT_SILHOUETTE_SAMPLES,
        help="Rows per silhouette, seeded sample (default: %(default)s).",
    )
    analyze.add_argument(
        "--max-plot-samples",
        type=_positive_int,
        default=DEFAULT_MAX_PLOT_SAMPLES,
        help="Rows drawn in the PCA figures (default: %(default)s).",
    )
    analyze.add_argument(
        "--balanced-cap",
        type=_positive_int,
        default=DEFAULT_BALANCED_CAP,
        help="Rows per class of the balanced silhouette (default: %(default)s).",
    )
    analyze.add_argument(
        "--labels-revision",
        help=(
            "Read the label sidecars at this dataset revision instead of the "
            "snapshot's; its action grid must be byte-identical."
        ),
    )
    analyze.add_argument(
        "--show",
        action="store_true",
        help=(
            "Then show the spectrum, PCA and label results: inline in a notebook "
            "kernel, else a text table and the figure paths. Results are "
            "unchanged."
        ),
    )
    analyze.set_defaults(handler=_analyze_latents)


def _analyze_latents(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> int:
    try:
        outputs = analyze_snapshot(
            args.snapshot,
            analyses=args.analysis or DEFAULT_ANALYSES,
            output_root=args.output,
            silhouette_samples=args.silhouette_samples,
            max_plot_samples=args.max_plot_samples,
            balanced_cap=args.balanced_cap,
            labels_revision=args.labels_revision,
        )
    except (GatedRepoError, RepositoryNotFoundError) as error:
        # The label analysis reads the (private) dataset release.
        raise SystemExit(
            "turn-wm: error: the dataset release is not accessible; private "
            "datasets require a Hugging Face login (`hf auth login`, or "
            f"HF_TOKEN): {error}"
        ) from error
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        # Not a snapshot, a non-empty output directory or no matplotlib.
        raise SystemExit(f"turn-wm: error: {error}") from error

    for name, output in outputs.items():
        print(f"{name}: {output}")

    if args.show and SPECTRUM in outputs:
        show_spectrum(outputs[SPECTRUM])

    if args.show and PCA in outputs:
        show_pca(outputs[PCA])

    if args.show and LABELS in outputs:
        show_labels(outputs[LABELS])

    return 0


def _add_extract_rollouts(commands: argparse._SubParsersAction) -> None:
    rollouts = commands.add_parser(
        "extract-rollouts",
        help="Extract a trained run's validation rollout for analysis.",
        description=(
            "Run the validation rollout of a training run (the function "
            "validation uses) on a seeded sample of the validation split and "
            "write the anchor, true future and predicted future latents and "
            "the actions the rollout read. The test split is never read."
        ),
    )
    _add_run_options(
        rollouts,
        max_samples=DEFAULT_ROLLOUT_SAMPLES,
        output_default="RUN_DIR/latent_analysis/<checkpoint>-validation-rollout",
    )
    rollouts.set_defaults(handler=_extract_rollouts)


def _extract_rollouts(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> int:
    try:
        output = extract_rollout_run(
            args.run_dir,
            output_dir=args.output,
            checkpoint=args.checkpoint,
            max_samples=args.max_samples,
            seed=args.seed,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=args.device,
            mimi_cache_root=args.mimi_cache_root,
        )
    except (ValueError, FileNotFoundError) as error:
        raise SystemExit(f"turn-wm: error: {error}") from error

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))

    print(f"Rollout written to {output}")
    print(f"samples: {manifest['samples']}")

    for name, spec in manifest["representations"].items():
        print(f"{name}: {spec['shape']}")

    return 0


def _add_analyze_rollouts(commands: argparse._SubParsersAction) -> None:
    dynamics = commands.add_parser(
        "analyze-rollouts",
        help="Analyze the dynamics of an extracted validation rollout.",
        description=(
            "Skill, displacement alignment and movement ratio per horizon, on "
            "all anchors and on stable and changing joint-speech states (labels "
            "read from the snapshot's dataset release)."
        ),
    )
    dynamics.add_argument(
        "snapshot",
        type=Path,
        help="Rollout snapshot directory written by extract-rollouts.",
    )
    _add_report_options(
        dynamics,
        output_default="SNAPSHOT/analysis/rollout_dynamics",
        labels=True,
    )
    dynamics.add_argument(
        "--trajectories",
        type=int,
        default=DEFAULT_TRAJECTORIES,
        help="Transitions drawn as PCA trajectories; 0 skips (default: %(default)s).",
    )
    dynamics.set_defaults(handler=_analyze_rollouts)


def _analyze_rollouts(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> int:
    try:
        output = write_rollout_dynamics(
            args.snapshot,
            output_dir=args.output,
            labels_revision=args.labels_revision,
            bootstrap=args.bootstrap,
            trajectories=args.trajectories,
        )
    except (GatedRepoError, RepositoryNotFoundError) as error:
        raise SystemExit(
            "turn-wm: error: the dataset release is not accessible; private "
            "datasets require a Hugging Face login (`hf auth login`, or "
            f"HF_TOKEN): {error}"
        ) from error
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        raise SystemExit(f"turn-wm: error: {error}") from error

    print(f"rollout_dynamics: {output}")
    print(f"report: {output / 'report.md'}")

    if args.show:
        show_rollouts(output)

    return 0


def _add_extract_action_ablation(commands: argparse._SubParsersAction) -> None:
    ablation = commands.add_parser(
        "extract-action-ablation",
        help="Extract a run's validation rollout under action ablations.",
        description=(
            "Run the validation rollout of a training run with observed, "
            "state-preserving and shuffled future ego actions, and measure the "
            "one-step effect of forcing the anchor action. Only state-valid "
            "WAIT/START or HOLD/STOP comparisons are reported. The test split "
            "is never read."
        ),
    )
    _add_run_options(
        ablation,
        max_samples=DEFAULT_ABLATION_SAMPLES,
        output_default=(
            "RUN_DIR/latent_analysis/<checkpoint>-validation-action-ablation"
        ),
    )
    ablation.set_defaults(handler=_extract_action_ablation)


def _extract_action_ablation(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> int:
    try:
        output = extract_action_ablation_run(
            args.run_dir,
            output_dir=args.output,
            checkpoint=args.checkpoint,
            max_samples=args.max_samples,
            seed=args.seed,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=args.device,
            mimi_cache_root=args.mimi_cache_root,
        )
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        raise SystemExit(f"turn-wm: error: {error}") from error

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))

    print(f"Action ablation written to {output}")
    print(f"samples: {manifest['samples']}")

    return 0


def _add_analyze_action_ablation(commands: argparse._SubParsersAction) -> None:
    ablation_analysis = commands.add_parser(
        "analyze-action-ablation",
        help="Analyze an extracted action ablation.",
        description=(
            "Rollout skill, displacement alignment and movement ratio under "
            "observed, state-preserving and shuffled future ego actions, with "
            "paired differences, and the forced one-step action effect per focal "
            "state (valid WAIT/START or HOLD/STOP pairs only). Reads the snapshot "
            "only."
        ),
    )
    ablation_analysis.add_argument(
        "snapshot",
        type=Path,
        help="Snapshot directory written by extract-action-ablation.",
    )
    _add_report_options(
        ablation_analysis,
        output_default="SNAPSHOT/analysis/action_ablation",
    )
    ablation_analysis.set_defaults(handler=_analyze_action_ablation)


def _analyze_action_ablation(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> int:
    try:
        output = write_action_ablation(
            args.snapshot, output_dir=args.output, bootstrap=args.bootstrap
        )
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        raise SystemExit(f"turn-wm: error: {error}") from error

    print(f"action_ablation: {output}")
    print(f"report: {output / 'report.md'}")

    if args.show:
        show_action_ablation(output)

    return 0


def _add_probe_latents(commands: argparse._SubParsersAction) -> None:
    probes = commands.add_parser(
        "probe-latents",
        help="Linear probes of a run's features and latent.",
        description=(
            "Fit linear probes (logistic for categorical labels, ridge for "
            "continuous ones) on a train-split snapshot and evaluate them on a "
            "validation-split snapshot of the same checkpoint, for the Mimi "
            "features and the WM latent: pooled, within and across corpora. "
            "Refuses shared recordings and the test split."
        ),
    )
    _add_probe_options(
        probes,
        output_default="VALIDATION_SNAPSHOT/analysis/probes",
    )
    probes.set_defaults(handler=_probe_latents)


def _probe_latents(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> int:
    try:
        output = write_probes(
            args.train_snapshot,
            args.validation_snapshot,
            output_dir=args.output,
            labels_revision=args.labels_revision,
            bootstrap=args.bootstrap,
        )
    except (GatedRepoError, RepositoryNotFoundError) as error:
        raise SystemExit(
            "turn-wm: error: the dataset release is not accessible; private "
            "datasets require a Hugging Face login (`hf auth login`, or "
            f"HF_TOKEN): {error}"
        ) from error
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        raise SystemExit(f"turn-wm: error: {error}") from error

    print(f"probes: {output}")
    print(f"report: {output / 'report.md'}")

    if args.show:
        show_probes(output)

    return 0


def _add_probe_concepts(commands: argparse._SubParsersAction) -> None:
    concepts = commands.add_parser(
        "probe-concepts",
        help="Linear probes of vocal, multi-party, social and unrelated concepts.",
        description=(
            "Probe the Mimi features and the WM latent for concepts beyond the "
            "current conversational state: vocal activity, multi-party "
            "structure, social signals and information unrelated to the "
            "conversation. Concepts that vary within a recording are fitted on "
            "the train-split snapshot and evaluated on the validation-split "
            "snapshot; recording-level concepts use conversation-grouped CV "
            "over both. Refuses shared recordings and the test split."
        ),
    )
    _add_probe_options(
        concepts,
        output_default="VALIDATION_SNAPSHOT/analysis/concepts",
    )
    concepts.set_defaults(handler=_probe_concepts)


def _probe_concepts(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> int:
    try:
        output = write_concepts(
            args.train_snapshot,
            args.validation_snapshot,
            output_dir=args.output,
            labels_revision=args.labels_revision,
            bootstrap=args.bootstrap,
        )
    except (GatedRepoError, RepositoryNotFoundError) as error:
        raise SystemExit(
            "turn-wm: error: the dataset release is not accessible; private "
            "datasets require a Hugging Face login (`hf auth login`, or "
            f"HF_TOKEN): {error}"
        ) from error
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        raise SystemExit(f"turn-wm: error: {error}") from error

    print(f"concepts: {output}")
    print(f"report: {output / 'report.md'}")

    if args.show:
        show_concepts(output)

    return 0


def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"expected an integer, got {value!r}"
        ) from None

    if number <= 0:
        raise argparse.ArgumentTypeError(f"must be positive, got {number}")

    return number
