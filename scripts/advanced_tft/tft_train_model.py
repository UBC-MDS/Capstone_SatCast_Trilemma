"""
tft_train_model.py

Handles model initialization, learning rate tuning using Tuner(trainer),
optimizer configuration, training, and checkpointing for the TFT model.
"""

import lightning.pytorch as pl
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.tuner import Tuner
from pytorch_forecasting import TemporalFusionTransformer
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pathlib import Path


def tft_configure_custom_optimizers(self, lr):
    """
    Custom optimizer and scheduler setup for TFT.

    Parameters:
    -----------
    self : TemporalFusionTransformer
    lr : float
        Learning rate to use in the optimizer.

    Returns:
    --------
    dict : Optimizer and LR scheduler config for PyTorch Lightning
    """
    optimizer = AdamW(
        self.parameters(),
        lr=lr,
        weight_decay=1e-4,
        eps=1e-6,
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.7,
        patience=8,
        min_lr=1e-6,
    )
    return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler,
            "monitor": "val_loss",
            "interval": "epoch",
            "frequency": 1,
        },
    }


def tft_train_model(tft_ds, train_dl, val_dl, loss_fn):
    """
    Train the TFT model using a custom loss and dynamic learning rate tuning.

    Parameters:
    -----------
    tft_ds : TimeSeriesDataSet
        Dataset for TFT initialization.
    train_dl : DataLoader
        Dataloader for training set.
    val_dl : DataLoader
        Dataloader for validation set.
    loss_fn : nn.Module
        Loss function to use during training.

    Returns:
    --------
    tft : TemporalFusionTransformer
        Trained model instance.
    trainer : pl.Trainer
        PyTorch Lightning trainer used for training.
    """
    # Set global seed for reproducibility
    pl.seed_everything(42)

    # Step 1: Build provisional model for LR tuning
    tft = TemporalFusionTransformer.from_dataset(
        tft_ds,
        learning_rate=1e-4,  # Placeholder
        hidden_size=32,
        hidden_continuous_size=8,
        lstm_layers=2,
        dropout=0.2,
        loss=loss_fn,
        output_size=1,
        weight_decay=1e-4,
        reduce_on_plateau_patience=0,
    )

    print(f"Model has {tft.size() / 1e3:.1f}k parameters")

    # Step 2: Minimal trainer for LR tuning
    tuner_trainer = Trainer(
        accelerator="gpu",
        devices=1,
        gradient_clip_val=0.1,
        enable_progress_bar=True,
    )

    # Step 3: Tune learning rate
    res = Tuner(tuner_trainer).lr_find(
        model=tft,
        train_dataloaders=train_dl,
        val_dataloaders=val_dl,
        min_lr=1e-6,
        max_lr=10.0,
        num_training=300
    )
    suggested_lr = res.suggestion()
    print(f"Suggested learning rate: {suggested_lr:.2e}")
    res.plot(suggest=True, show=True)

    # Step 4: Rebuild model using tuned LR
    tft = TemporalFusionTransformer.from_dataset(
        tft_ds,
        learning_rate=suggested_lr,
        hidden_size=32,
        hidden_continuous_size=8,
        lstm_layers=2,
        dropout=0.2,
        loss=loss_fn,
        output_size=1,
        weight_decay=1e-4,
        reduce_on_plateau_patience=0,
    )


    tft.configure_optimizers = lambda: tft_configure_custom_optimizers(tft, suggested_lr)

    # Step 5: Set up callbacks and final trainer
    project_root = Path(__file__).resolve().parents[2]
    model_save_dir = project_root / "results" / "models"


    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=15,
            min_delta=0.003,
            mode="min",
            verbose=True
        ),
        LearningRateMonitor(logging_interval='step'),
        ModelCheckpoint(
            dirpath=str(model_save_dir),
            filename="best-model-{epoch}-{val_loss:.4f}",
            monitor="val_loss",
            save_top_k=1,
            save_last=True,
            mode="min",
            verbose=True
        ),
    ]

    trainer = Trainer(
        max_epochs=100,
        accelerator="gpu",
        devices=1,
        precision="bf16-mixed",
        gradient_clip_val=0.5,
        callbacks=callbacks,
        val_check_interval=0.5,
        accumulate_grad_batches=1,
        deterministic=False,
        enable_progress_bar=True,
    )

    # Step 6: Train the model
    trainer.fit(tft, train_dl, val_dl)

    return tft, trainer
