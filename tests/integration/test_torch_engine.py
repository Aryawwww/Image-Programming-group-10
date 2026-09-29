from __future__ import annotations

import torch

from pv_pipeline.determinism import configure
from pv_pipeline.models.small_cnn import SmallCNN
from pv_pipeline.smoke import select_smoke_rows
from pv_pipeline.training.torch_engine import train_epochs


def test_small_cnn_training_engine_one_epoch(context, tmp_path):
    configure(42177)
    selected = select_smoke_rows(context)
    model = SmallCNN(context.config["small_cnn"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=1, eta_min=1e-6
    )
    result = train_epochs(
        context=context,
        model=model,
        train_rows=selected["train"],
        val_rows=selected["val"],
        device=torch.device("cpu"),
        stage="integration",
        epochs=1,
        batch_size=8,
        optimizer=optimizer,
        scheduler=scheduler,
        gradient_clip_norm=5.0,
        history_path=tmp_path / "history.csv",
    )
    assert result.best_record["epoch"] == 0
    assert len(result.history) == 1
    assert (tmp_path / "history.csv").is_file()
    replay = train_epochs(
        context=context,
        model=model,
        train_rows=selected["train"],
        val_rows=selected["val"],
        device=torch.device("cpu"),
        stage="integration",
        epochs=1,
        batch_size=8,
        optimizer=optimizer,
        scheduler=scheduler,
        gradient_clip_norm=5.0,
        history_path=tmp_path / "history.csv",
    )
    assert replay.best_record == result.best_record
    assert len(replay.history) == 1
