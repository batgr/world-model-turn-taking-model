"""The decision grid rate is measured on the loaded data and checked by a run."""

from dataclasses import replace

import pytest
from corpora import make_corpus

from turn_wm.config import load_config
from turn_wm.data.source import LoadedData
from turn_wm.training.observations import prepare_observations


def at_rate(corpus, rate_hz):
    grid = corpus.action_grid.map(
        lambda row: {"decision_time_s": row["decision_index"] / rate_hz}
    )
    return replace(corpus, action_grid=grid)


def corpus(name="a", rate_hz=10.0, **metadata):
    base = make_corpus(name, state="SILENT", splits={"train": 1})
    return replace(at_rate(base, rate_hz), metadata=metadata)


@pytest.mark.parametrize("rate_hz", [10.0, 12.5])
def test_the_grid_rate_is_measured_on_the_action_grid(rate_hz):
    assert corpus(rate_hz=rate_hz).grid_rate_hz == rate_hz
    assert LoadedData(corpora=(corpus(rate_hz=rate_hz),)).grid_rate_hz == rate_hz


def test_metadata_disagreeing_with_the_grid_is_refused():
    with pytest.raises(ValueError, match="declares a 10.0 Hz grid"):
        _ = corpus(rate_hz=12.5, grid={"frequency_hz": 10.0}).grid_rate_hz


def test_corpora_on_different_grids_are_refused():
    loaded = LoadedData(corpora=(corpus("a", 10.0), corpus("b", 12.5)))

    with pytest.raises(ValueError, match="different decision grid rates"):
        _ = loaded.grid_rate_hz


def test_a_run_refuses_data_on_another_grid_than_its_config():
    cfg = load_config(["data.observation_source=raw_audio"])  # data.grid_rate_hz=10
    loaded = LoadedData(corpora=(corpus(rate_hz=12.5),))

    with pytest.raises(ValueError, match="set data.grid_rate_hz=12.5"):
        prepare_observations(cfg, loaded, media_roots={})
