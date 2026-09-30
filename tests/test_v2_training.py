"""A two-epoch V2 fit exercises the curriculum, metrics and callbacks."""

import lightning as L
import torch
from torch.utils.data import DataLoader

from turn_wm.config import load_config
from turn_wm.training.lewm import LeWMModule
from turn_wm.training.train import _build_callbacks


def test_v2_fit_reaches_a_full_h10_epoch_and_scores_transitions(tmp_path):
    cfg = load_config(
        [
            "model=lewm_bn",
            "train=lewm_v2",
            "embed_dim=32",
            "data.context_steps=4",
            "prediction.rollout_context_size=4",
            "model.predictor.depth=1",
            "model.predictor.heads=2",
            "model.predictor.dim_head=16",
            "model.predictor.mlp_dim=64",
            "model.projector.hidden_dim=64",
            "model.pred_proj.hidden_dim=64",
            "loss.sigreg.kwargs.num_proj=16",
            "trainer.max_epochs=2",
            "loader.batch_size=2",
        ]
    )
    torch.manual_seed(2)
    batch = {
        "sample_id": ["a", "b"],
        "dataset": ["egocom", "egocom"],
        "context_lengths": torch.tensor([4, 4]),
        "context_action": torch.tensor(
            [[0, 0, 0, 1], [0, 0, 0, 0]], dtype=torch.long
        ),
        "context_valid": torch.ones(2, 4, dtype=torch.bool),
        "future_action": torch.zeros(2, 10, dtype=torch.long),
        "future_valid": torch.ones(2, 10, dtype=torch.bool),
        "context_state": torch.zeros(2, 4, dtype=torch.long),
        "future_state": torch.ones(2, 10, dtype=torch.long),
        "context_features": torch.randn(2, 4, 512),
        "future_features": torch.randn(2, 10, 512),
    }
    trainer = L.Trainer(
        max_epochs=2,
        max_steps=-1,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        callbacks=_build_callbacks(cfg, run_dir=tmp_path),
    )
    trainer.fit(
        LeWMModule(cfg),
        train_dataloaders=DataLoader([batch, batch], batch_size=None),
        val_dataloaders=DataLoader([batch], batch_size=None),
    )

    assert trainer.global_step == 4
    assert "val/rollout_10_mse" in trainer.callback_metrics
    assert (tmp_path / "checkpoints" / "last.ckpt").is_file()
