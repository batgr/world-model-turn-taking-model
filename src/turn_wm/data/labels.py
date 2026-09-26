"""Optional label sidecars published next to a corpus: select and read, nothing else.

Every label is defined, derived and versioned by the data repository
(``world-model-turn-taking-data``, ``conv-wm build labels``). This module never
derives a label; it reads the ``registry.json`` shipped with the labels, picks
the requested ones and reads only their columns.

```yaml
labels:
  enabled: false          # default: nothing is fetched or read
  include: [all]          # exact names, "family.*" or "all"
  modalities: []          # empty = no filter; else keep labels whose modalities
                          # are all listed ([audio, video] keeps both and audio+video)
```

A label named exactly that the corpus does not provide raises; one reached
through ``all`` or a wildcard is reported in ``CorpusLabels.skipped``.

Grid labels have exactly the rows of the corpus's action grid, in the same
order, so ``anchor_row`` indexes them as well (:func:`check_grid_alignment`).
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

REGISTRY_FILE = "registry.json"
MANIFEST_FILE = "manifest.json"
SUPPORTED_REGISTRY_VERSIONS = frozenset({1})

TABLE_KEYS: Mapping[str, tuple[str, ...]] = {
    "grid": ("recording_id", "decision_index", "decision_time_s"),
    "events": ("recording_id", "event_type", "time_s"),
    "segments": ("recording_id", "segment_type", "segment_id", "start_s", "end_s"),
    "participants": ("recording_id", "participant_index", "participant_id"),
    "recordings": ("recording_id",),
}
PARTICIPANT_AXIS_COLUMN = "participant_ids"

Fetch = Callable[[str], Path]
"""Store-relative POSIX path (e.g. ``speech/grid.parquet``) -> local file."""


class LabelSelectionError(ValueError):
    """The selection names a label, family or modality the registry does not know."""


class LabelUnavailableError(LookupError):
    """An explicitly requested label is not provided by this corpus."""


@dataclass(frozen=True)
class LabelsConfig:
    """Which labels to load. The default loads none."""

    enabled: bool = False
    include: tuple[str, ...] = ()
    modalities: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> LabelsConfig:
        """Build from a (Hydra) mapping; ``None`` or ``{}`` means disabled."""
        if not value:
            return cls()
        unknown = set(value) - {"enabled", "include", "modalities"}
        if unknown:
            raise LabelSelectionError(f"unknown labels keys {sorted(unknown)}")
        return cls(
            enabled=bool(value.get("enabled", False)),
            include=tuple(value.get("include") or ()),
            modalities=tuple(value.get("modalities") or ()),
        )


@dataclass(frozen=True)
class CorpusLabels:
    """The selected labels of one corpus, one Arrow table per table kind."""

    tables: Mapping[str, pa.Table] = field(default_factory=dict)
    labels: tuple[str, ...] = ()
    skipped: Mapping[str, str] = field(default_factory=dict)
    registry_version: int | None = None

    def __bool__(self) -> bool:
        return bool(self.tables)


def local_store(root: Path) -> Fetch:
    """A label store on disk (``<model_ready or release>/<dataset>/labels``)."""
    return lambda relative: root / PurePosixPath(relative)


def hub_store(repo_id: str, base: str, *, revision: str | None) -> Fetch:
    """A label store inside a Hub dataset repository, pinned to ``revision``."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError

    def fetch(relative: str) -> Path:
        filename = str(PurePosixPath(base) / relative)
        try:
            return Path(
                hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    repo_type="dataset",
                    revision=revision,
                )
            )
        except EntryNotFoundError as error:
            raise FileNotFoundError(f"{repo_id}:{filename}") from error

    return fetch


def resolve(
    config: LabelsConfig, registry: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """The registry entries a selection resolves to, and the ones it skips."""
    labels: list[dict[str, Any]] = list(registry["labels"])
    names = {entry["name"] for entry in labels}
    families = {entry["family"] for entry in labels} | set(registry.get("families", {}))
    known_modalities = {m for entry in labels for m in entry["modalities"]}
    unknown = set(config.modalities) - known_modalities
    if unknown:
        raise LabelSelectionError(f"unknown modalities {sorted(unknown)}")
    if not config.include:
        raise LabelSelectionError("labels.enabled is true but labels.include is empty")
    wanted: set[str] = set()
    explicit: set[str] = set()
    for item in config.include:
        if item == "all":
            wanted |= names
        elif item.endswith(".*"):
            family = item[:-2]
            if family not in families:
                raise LabelSelectionError(f"unknown label family {family!r}")
            wanted |= {e["name"] for e in labels if e["family"] == family}
        elif item in names:
            wanted.add(item)
            explicit.add(item)
        else:
            raise LabelSelectionError(f"unknown label {item!r}")
    chosen: list[dict[str, Any]] = []
    skipped: dict[str, str] = {}
    for entry in labels:
        if entry["name"] not in wanted:
            continue
        if config.modalities and not set(entry["modalities"]) <= set(config.modalities):
            if entry["name"] in explicit:
                raise LabelSelectionError(
                    f"{entry['name']} needs modalities {entry['modalities']}"
                )
            continue
        if entry["availability"] != "available":
            reason = f"unsupported: {entry.get('unsupported_reason')}"
            if entry["name"] in explicit:
                raise LabelUnavailableError(f"{entry['name']} is {reason}")
            skipped[entry["name"]] = reason
            continue
        chosen.append(entry)
    return chosen, skipped


def load_labels(
    fetch: Fetch | None,
    config: LabelsConfig,
    *,
    recording_ids: Sequence[str] | None = None,
) -> CorpusLabels:
    """Read the labels ``config`` asks for; touch nothing when disabled."""
    if not config.enabled:
        return CorpusLabels()
    if fetch is None:
        raise LabelUnavailableError("this corpus publishes no label sidecars")
    registry = _json(fetch(REGISTRY_FILE))
    version = registry.get("registry_version")
    if version not in SUPPORTED_REGISTRY_VERSIONS:
        raise LabelSelectionError(f"unsupported label registry version {version}")
    entries, skipped = resolve(config, registry)
    explicit = {item for item in config.include if item != "all" and "*" not in item}
    manifests: dict[str, dict[str, Any] | None] = {}
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    chosen: list[str] = []
    for entry in entries:
        extractor = entry["extractor"]
        if extractor not in manifests:
            manifests[extractor] = _optional_json(fetch, f"{extractor}/{MANIFEST_FILE}")
        manifest = manifests[extractor]
        if manifest is None or entry["name"] not in manifest["materialized_labels"]:
            reason = (
                f"extractor {extractor} not published"
                if manifest is None
                else manifest.get("unavailable_labels", {}).get(
                    entry["name"], "not built"
                )
            )
            if entry["name"] in explicit:
                raise LabelUnavailableError(f"{entry['name']}: {reason}")
            skipped[entry["name"]] = reason
            continue
        chosen.append(entry["name"])
        grouped.setdefault((extractor, entry["table"]), []).append(entry)
    tables: dict[str, list[pa.Table]] = {}
    for (extractor, table), group in sorted(grouped.items()):
        manifest = manifests[extractor]
        assert manifest is not None
        path = fetch(f"{extractor}/{manifest['tables'][table]['file']}")
        tables.setdefault(table, []).append(_read(path, table, group, recording_ids))
    return CorpusLabels(
        tables={name: _combine(name, parts) for name, parts in tables.items()},
        labels=tuple(chosen),
        skipped=skipped,
        registry_version=version,
    )


def check_grid_alignment(labels: pa.Table, action_grid: pa.Table) -> None:
    """Require label grid rows to be the action grid's, so ``anchor_row`` indexes both."""
    for column in ("recording_id", "decision_index"):
        left, right = labels.column(column), action_grid.column(column)
        if left.type != right.type:  # e.g. string vs large_string: compare values
            left = left.cast(right.type)
        if not left.equals(right):
            raise ValueError(
                f"label grid is not aligned with the action grid on {column!r}; "
                "the labels were built from another action grid"
            )


def _read(
    path: Path,
    table: str,
    entries: Sequence[Mapping[str, Any]],
    recording_ids: Sequence[str] | None,
) -> pa.Table:
    columns = list(TABLE_KEYS[table])
    if table == "grid" and any("participant" in e["shape"] for e in entries):
        columns.append(PARTICIPANT_AXIS_COLUMN)
    for entry in entries:
        columns += [c for c in entry["columns"] if c not in columns]
    filters: list[tuple[str, str, Any]] = []
    if recording_ids is not None:
        filters.append(("recording_id", "in", list(recording_ids)))
    row_filters = [entry.get("row_filter") for entry in entries]
    if table in ("events", "segments") and all(row_filters):
        column = row_filters[0][0]  # type: ignore[index]
        filters.append((column, "in", sorted({value for _, value in row_filters})))  # type: ignore[misc]
    return pq.read_table(path, columns=columns, filters=filters or None)


def _combine(table: str, parts: list[pa.Table]) -> pa.Table:
    if len(parts) == 1:
        return parts[0]
    if table == "grid":
        combined = parts[0]
        for part in parts[1:]:
            check_grid_alignment(part, combined)
            for name in part.column_names:
                if name not in combined.column_names:
                    combined = combined.append_column(name, part.column(name))
        return combined
    return pa.concat_tables(parts, promote_options="default")


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _optional_json(fetch: Fetch, relative: str) -> dict[str, Any] | None:
    try:
        path = fetch(relative)
    except FileNotFoundError:
        return None
    return _json(path) if path.exists() else None


__all__ = [
    "CorpusLabels",
    "LabelSelectionError",
    "LabelUnavailableError",
    "LabelsConfig",
    "check_grid_alignment",
    "hub_store",
    "load_labels",
    "local_store",
    "resolve",
]
