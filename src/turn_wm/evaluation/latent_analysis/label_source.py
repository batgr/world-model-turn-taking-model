"""
Label sidecars for a representation snapshot: audit, inventory and exact join.

The labels are the data release's (`<corpus>/labels/`, see
`turn_wm.data.labels`); nothing here derives a label. For each corpus:

1. audit: read `registry.json` and every extractor's `manifest.json` at the
   label revision. A label counts as materialized only when its extractor's
   manifest lists it; the registry alone proves nothing. Each manifest
   records the SHA-256 of the action grid its labels were built from, which
   must be the SHA-256 of the action grid the snapshot's run trained on;
2. join: grid labels are read for the snapshot's recordings only and joined
   on (corpus, recording_id, decision_index = anchor_idx), exactly. Every
   joined row must also agree on time (`decision_time_s` == `anchor_time`).
   A missing key is reported, never approximated.

By default the labels come from the snapshot's own dataset revision. Another
revision is used only when asked for, and only if its action grid file is
byte-identical (same LFS SHA-256) to the snapshot's.

Labels become analysis variables by their registry dtype and shape:
booleans are two classes; floats are continuous; per-horizon lists become
one variable per horizon (horizons read from the extractor's manifest); the
joint-state occupancy becomes its dominant joint state. Integer indices
without a shared vocabulary (e.g. a participant index) are excluded.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import pyarrow.parquet as pq

from turn_wm.data.labels import MANIFEST_FILE, REGISTRY_FILE, Fetch, hub_store

CONVERSATIONAL_STATE = "conversational_state"
TEMPORAL_STATE = "temporal_state"
FUTURE = "future"
NUISANCE = "nuisance"
SECTIONS = (CONVERSATIONAL_STATE, TEMPORAL_STATE, FUTURE, NUISANCE)

CATEGORICAL = "categorical"
CONTINUOUS = "continuous"

# The first pass: labels tied to the turn-taking question, and controls.
SELECTION: tuple[tuple[str, str], ...] = (
    (CONVERSATIONAL_STATE, "instantaneous.ego_speaking"),
    (CONVERSATIONAL_STATE, "instantaneous.others_active"),
    (CONVERSATIONAL_STATE, "instantaneous.joint_speech_state_occupancy"),
    (TEMPORAL_STATE, "timing.time_to_next_floor_change"),
    (TEMPORAL_STATE, "timing.time_to_next_speaker_onset"),
    (TEMPORAL_STATE, "timing.silence_duration"),
    (TEMPORAL_STATE, "timing.time_since_floor_change"),
    (FUTURE, "next_speaker.current_speaker_continues"),
    (FUTURE, "next_speaker.next_unique_speaker"),
    (FUTURE, "future.future_ego_onset"),
    (FUTURE, "future.future_other_onset"),
    (FUTURE, "future.future_overlap"),
    (FUTURE, "future.future_joint_speech_state"),
    (NUISANCE, "nuisance.global_audio_rms"),
    (NUISANCE, "prosody.ego_rms_db"),
    (NUISANCE, "prosody.ego_voiced_fraction"),
)

# The joint wearer/others states, in the registry's order ("0 silence, 1 ego
# only, 2 others only, 3 ego and others"; occupancy lists them alike).
JOINT_STATES = ("silence", "ego_only", "others_only", "both")
BOOLEAN_CLASSES = ("false", "true")

# Snapshot anchor_time is float32: seconds agree to well under a millisecond.
TIME_TOLERANCE_S = 1e-3


@dataclass(frozen=True)
class CorpusLabelSource:
    """Where one corpus's labels are read, and the grid they must match."""

    corpus: str
    fetch: Fetch
    grid_sha256: str  # the action grid the snapshot's run trained on
    provenance: dict[str, Any]


@dataclass(frozen=True)
class CorpusAudit:
    """What one corpus's label release actually provides."""

    corpus: str
    provenance: dict[str, Any]
    registry: dict[str, Any] | None
    manifests: dict[str, dict[str, Any] | None] = field(default_factory=dict)
    # Extractor -> why its labels cannot be used (e.g. another grid).
    rejected: dict[str, str] = field(default_factory=dict)
    missing_reason: str | None = None

    def entry(self, label: str) -> dict[str, Any] | None:
        if self.registry is None:
            return None

        return next((e for e in self.registry["labels"] if e["name"] == label), None)

    def unavailable_reason(self, label: str) -> str | None:
        """None when `label` can be read; else why not."""

        if self.registry is None:
            return self.missing_reason
        entry = self.entry(label)

        if entry is None:
            return "not in the registry"
        if entry["availability"] != "available":
            return f"unsupported: {entry.get('unsupported_reason')}"
        extractor = entry["extractor"]
        if extractor in self.rejected:
            return self.rejected[extractor]
        manifest = self.manifests.get(extractor)
        if manifest is None:
            return f"extractor {extractor} not published"
        if label not in manifest["materialized_labels"]:
            return manifest.get("unavailable_labels", {}).get(label, "not built")

        return None


@dataclass(frozen=True)
class LabelVariable:
    """One analysed variable, aligned with the snapshot rows."""

    name: str  # the label, or "<label>@<h>s" for one horizon
    label: str
    section: str
    kind: str  # CATEGORICAL or CONTINUOUS
    classes: tuple[str, ...] | None
    horizon_s: float | None
    values: list[Any]  # per snapshot row; None when missing or invalid


@dataclass(frozen=True)
class JoinedLabels:
    variables: list[LabelVariable]
    excluded: dict[str, dict[str, str]]  # label -> corpus -> reason
    coverage: list[dict[str, Any]]
    alignment: dict[str, Any]


# ---------------------------------------------------------------------------
# Sources and audit
# ---------------------------------------------------------------------------


def hub_label_sources(
    provenance: Mapping[str, Any],
    *,
    labels_revision: str | None = None,
) -> dict[str, CorpusLabelSource]:
    """The label stores of the snapshot's dataset, on the Hugging Face Hub."""

    from huggingface_hub import HfApi

    from turn_wm.data.source import DATASETS

    data = provenance.get("data") or {}
    name = data.get("dataset")
    snapshot_revision = data.get("dataset_revision")

    if name not in DATASETS or snapshot_revision is None:
        raise ValueError(
            "The snapshot does not record its dataset and dataset revision; "
            "its labels cannot be matched"
        )

    source = DATASETS[name]
    revision = labels_revision or snapshot_revision
    api = HfApi()

    def grid_shas(at: str) -> dict[str, str | None]:
        files = {
            entry.path: entry.lfs.sha256
            for entry in api.list_repo_tree(
                source.repo_id, repo_type="dataset", revision=at, recursive=True
            )
            if getattr(entry, "lfs", None) is not None
        }
        return {c.name: files.get(c.action_grid_file) for c in source.corpora}

    snapshot_grids = grid_shas(snapshot_revision)
    label_grids = (
        snapshot_grids if revision == snapshot_revision else grid_shas(revision)
    )
    sources = {}

    for corpus in source.corpora:
        grid = snapshot_grids[corpus.name]

        if grid is None:
            raise ValueError(
                f"No {corpus.action_grid_file} at revision {snapshot_revision}"
            )

        if label_grids[corpus.name] != grid:
            raise ValueError(
                f"{corpus.name}: the action grid at label revision {revision} "
                f"({label_grids[corpus.name]}) is not the snapshot's "
                f"({grid} at {snapshot_revision}); its labels cannot be joined"
            )

        sources[corpus.name] = CorpusLabelSource(
            corpus=corpus.name,
            fetch=hub_store(source.repo_id, corpus.labels_dir, revision=revision),
            grid_sha256=grid,
            provenance={
                "repo_id": source.repo_id,
                "snapshot_revision": snapshot_revision,
                "labels_revision": revision,
                "labels_dir": corpus.labels_dir,
                "action_grid_file": corpus.action_grid_file,
                "action_grid_sha256": grid,
            },
        )

    return sources


def audit_corpus(source: CorpusLabelSource) -> CorpusAudit:
    """Read the registry and manifests; reject extractors built on another grid."""

    try:
        registry = _json(source.fetch(REGISTRY_FILE))
    except FileNotFoundError:
        return CorpusAudit(
            corpus=source.corpus,
            provenance=source.provenance,
            registry=None,
            missing_reason="no labels/registry.json at this revision",
        )

    manifests: dict[str, dict[str, Any] | None] = {}
    rejected: dict[str, str] = {}

    for extractor in sorted(
        {e["extractor"] for e in registry["labels"] if e["extractor"]}
    ):
        try:
            manifest = _json(source.fetch(f"{extractor}/{MANIFEST_FILE}"))
        except FileNotFoundError:
            manifest = None

        manifests[extractor] = manifest

        if manifest is None:
            continue

        built_from = ((manifest.get("inputs") or {}).get("action_grid") or {}).get(
            "sha256"
        )

        if built_from is None:
            rejected[extractor] = "grid provenance not recorded in its manifest"
        elif built_from != source.grid_sha256:
            rejected[extractor] = (
                f"built from action grid {built_from}, not the snapshot's "
                f"{source.grid_sha256}"
            )

    return CorpusAudit(
        corpus=source.corpus,
        provenance=source.provenance,
        registry=registry,
        manifests=manifests,
        rejected=rejected,
    )


def label_inventory(
    audits: Mapping[str, CorpusAudit],
    coverage: Sequence[Mapping[str, Any]] = (),
) -> list[dict[str, Any]]:
    """One row per (registry label, corpus); coverage for the analysed ones."""

    measured: dict[tuple[str, str], dict[str, Any]] = {}

    for row in coverage:
        key = (row["label"], row["dataset"])
        # Per-horizon variables share their label's coverage.
        measured.setdefault(key, dict(row))

    rows = []

    for corpus, audit in audits.items():
        labels = audit.registry["labels"] if audit.registry else []

        for entry in labels:
            name = entry["name"]
            reason = audit.unavailable_reason(name)
            manifest = audit.manifests.get(entry["extractor"]) or {}
            seen = measured.get((name, corpus), {})
            rows.append(
                {
                    "label": name,
                    "family": entry["family"],
                    "role": entry.get("role"),
                    "level": entry.get("level"),
                    "source_kind": entry.get("source_kind"),
                    "modalities": list(entry.get("modalities") or []),
                    "dataset": corpus,
                    "available": entry["availability"] == "available",
                    "materialized": reason is None,
                    "unavailable_reason": reason,
                    "coverage": seen.get("coverage_fraction"),
                    "null_fraction": seen.get("null_fraction"),
                    "extractor": entry.get("extractor"),
                    "table": entry.get("table"),
                    "extractor_version": manifest.get("extractor_version"),
                    "code_revision": (manifest.get("code_revision") or {}).get(
                        "git_commit"
                    ),
                    "revision": audit.provenance.get("labels_revision"),
                }
            )

    return rows


# ---------------------------------------------------------------------------
# Exact join
# ---------------------------------------------------------------------------


def join_labels(
    metadata: Mapping[str, Sequence[Any]],
    audits: Mapping[str, CorpusAudit],
    sources: Mapping[str, CorpusLabelSource],
    *,
    selection: Sequence[tuple[str, str]] = SELECTION,
) -> JoinedLabels:
    """The selected grid labels at every snapshot row, joined exactly."""

    datasets = [str(d) for d in metadata["dataset"]]
    recordings = [str(r) for r in metadata["recording_id"]]
    anchors = [int(a) for a in metadata["anchor_idx"]]
    times = [float(t) for t in metadata["anchor_time"]]
    rows = len(datasets)

    unknown = sorted(set(datasets) - set(audits))

    if unknown:
        raise ValueError(f"No label source for corpora {unknown}")

    for corpus, audit in audits.items():
        if audit.registry is None and corpus in datasets:
            raise ValueError(
                f"{corpus}: {audit.missing_reason}; pass a later labels "
                "revision explicitly if one is published"
            )

    variables: dict[str, LabelVariable] = {}
    excluded: dict[str, dict[str, str]] = {}
    coverage = []
    checked = 0
    max_difference = 0.0

    for corpus in sorted(set(datasets)):
        audit, source = audits[corpus], sources[corpus]
        members = [i for i, d in enumerate(datasets) if d == corpus]
        usable = []

        for section, label in selection:
            reason = audit.unavailable_reason(label)
            entry = audit.entry(label)

            if entry is not None and entry.get("extractor") in audit.rejected:
                raise ValueError(
                    f"{corpus}: {label} comes from extractor {entry['extractor']}, "
                    f"{audit.rejected[entry['extractor']]}"
                )

            if reason is None and entry is not None:
                reason = _unsupported_shape(entry)

            if reason is None:
                usable.append((section, entry))
            else:
                excluded.setdefault(label, {})[corpus] = reason

        by_file: dict[str, list[dict[str, Any]]] = {}

        for _, entry in usable:
            manifest = audit.manifests[entry["extractor"]]
            path = f"{entry['extractor']}/{manifest['tables']['grid']['file']}"
            by_file.setdefault(path, []).append(entry)

        for path, entries in by_file.items():
            columns = ["recording_id", "decision_index", "decision_time_s"]
            columns += [c for e in entries for c in e["columns"] if c not in columns]
            table = pq.read_table(
                source.fetch(path),
                columns=columns,
                filters=[
                    ("recording_id", "in", sorted({recordings[i] for i in members}))
                ],
            ).to_pydict()
            index = {
                key: j
                for j, key in enumerate(
                    zip(table["recording_id"], table["decision_index"], strict=True)
                )
            }
            found = [index.get((recordings[i], anchors[i])) for i in members]

            for i, j in zip(members, found, strict=True):
                if j is None:
                    continue

                difference = abs(table["decision_time_s"][j] - times[i])
                max_difference = max(max_difference, difference)

                if difference > TIME_TOLERANCE_S:
                    raise ValueError(
                        f"{corpus}/{recordings[i]} decision_index {anchors[i]}: "
                        f"grid time {table['decision_time_s'][j]} differs from the "
                        f"snapshot's anchor_time {times[i]}; not the same timeline"
                    )

            checked += sum(j is not None for j in found)

            for entry in entries:
                section = next(s for s, e in usable if e is entry)
                manifest = audit.manifests[entry["extractor"]]

                for variable, values in _variables(entry, table, found, manifest):
                    target = variables.get(variable.name)

                    if target is None:
                        target = replace(
                            variable, section=section, values=[None] * rows
                        )
                        variables[variable.name] = target

                    for i, value in zip(members, values, strict=True):
                        target.values[i] = value

                    valid = sum(value is not None for value in values)
                    joined = sum(j is not None for j in found)
                    coverage.append(
                        {
                            "variable": variable.name,
                            "label": entry["name"],
                            "dataset": corpus,
                            "n_snapshot": len(members),
                            "n_joined": joined,
                            "n_valid": valid,
                            "n_missing": len(members) - joined,
                            "coverage_fraction": valid / len(members),
                            "null_fraction": (
                                (joined - valid) / joined if joined else None
                            ),
                        }
                    )

    order = {label: k for k, (_, label) in enumerate(selection)}

    return JoinedLabels(
        variables=sorted(
            variables.values(), key=lambda v: (order[v.label], v.horizon_s or 0.0)
        ),
        excluded=excluded,
        coverage=coverage,
        alignment={
            "key": "(dataset, recording_id, decision_index = anchor_idx)",
            "time_check": "decision_time_s == anchor_time",
            "time_tolerance_s": TIME_TOLERANCE_S,
            "joined_rows_checked": checked,
            "max_abs_time_difference_s": max_difference,
        },
    )


def _unsupported_shape(entry: Mapping[str, Any]) -> str | None:
    if entry.get("table") != "grid" or entry.get("level") != "grid":
        return f"not a grid-level label ({entry.get('level')}, {entry.get('table')})"

    dtype, shape = entry["dtype"], entry["shape"]

    if shape == "[]" and dtype.startswith(("int", "uint")):
        return "integer index without a shared vocabulary across recordings"

    if shape == "[]" and dtype.startswith(("bool", "float")):
        return None

    if shape == "[joint_state]" or (
        shape == "[horizon]" and dtype.startswith("fixed_size_list<")
    ):
        return None

    return f"shape {shape} / dtype {dtype} not analysed in this pass"


def _variables(
    entry: Mapping[str, Any],
    table: Mapping[str, list[Any]],
    found: Sequence[int | None],
    manifest: Mapping[str, Any],
):
    """(variable, values per joined snapshot row) for one registry label."""

    name, dtype, shape = entry["name"], entry["dtype"], entry["shape"]
    column = entry["columns"][0]
    valid_column = next((c for c in entry["columns"][1:] if c.endswith("_valid")), None)

    def cell(j):
        if j is None:
            return None
        if valid_column is not None and not table[valid_column][j]:
            return None
        return table[column][j]

    raw = [cell(j) for j in found]

    if shape == "[]":
        if dtype.startswith("bool"):
            values = [None if v is None else BOOLEAN_CLASSES[int(bool(v))] for v in raw]
            yield _variable(name, CATEGORICAL, BOOLEAN_CLASSES), values
        else:
            values = [None if v is None else float(v) for v in raw]
            yield _variable(name, CONTINUOUS, None), values
        return

    if shape == "[joint_state]":
        # The dominant joint state of the cell (the first on ties).
        values = [
            None
            if v is None or any(x is None for x in v)
            else JOINT_STATES[max(range(len(v)), key=v.__getitem__)]
            for v in raw
        ]
        yield (
            _variable(f"{name}:dominant", CATEGORICAL, JOINT_STATES, label=name),
            values,
        )
        return

    horizons = list((manifest.get("config") or {}).get("future_horizons_s") or [])

    for v in raw:
        if v is not None and len(v) != len(horizons):
            raise ValueError(
                f"{name} has {len(v)} horizons; its manifest configures {horizons}"
            )

    classes = BOOLEAN_CLASSES if "bool" in dtype else JOINT_STATES

    for h, horizon in enumerate(horizons):
        values = []

        for v in raw:
            element = None if v is None else v[h]
            if element is None:
                values.append(None)
            elif classes is BOOLEAN_CLASSES:
                values.append(BOOLEAN_CLASSES[int(bool(element))])
            else:
                values.append(JOINT_STATES[int(element)])

        yield (
            _variable(
                f"{name}@{horizon:g}s",
                CATEGORICAL,
                classes,
                label=name,
                horizon=horizon,
            ),
            values,
        )


def _variable(name, kind, classes, *, label=None, horizon=None) -> LabelVariable:
    return LabelVariable(
        name=name,
        label=label or name,
        section="",
        kind=kind,
        classes=None if classes is None else tuple(classes),
        horizon_s=horizon,
        values=[],
    )


def _json(path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
