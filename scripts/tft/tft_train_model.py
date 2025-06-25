# tft_train_model.py
# author: Ximin Xu
# date: 2025-06-18

"""
Script to train a Temporal Fusion Transformer (TFT) model for Bitcoin fee forecasting 
using PyTorch Lightning and custom volatility-aware loss.

This script performs the following steps:
1. Loads prepared dataloaders for training and validation.
2. Initializes a provisional TFT model and uses PyTorch Lightning’s `lr_find`
   to determine the optimal learning rate.
3. Rebuilds the TFT model with the suggested learning rate and a custom optimizer.
4. Sets up training callbacks: early stopping, learning rate monitoring, and checkpointing.
5. Trains the model using mixed precision and saves the best-performing checkpoint to disk.
6. Loads and returns the best model for downstream inference.

Usage:
    Called from a training pipeline after dataloader construction.

Dependencies:
    - PyTorch Lightning, PyTorch Forecasting, torch, lightning.pytorch
    - Custom loss function passed in during training
"""

import lightning.pytorch as pl
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.tuner import Tuner
from pytorch_forecasting import TemporalFusionTransformer
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pathlib import Path
import torch

if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


def tft_configure_custom_optimizers(self, lr):
    """
    Custom optimizer and scheduler setup for TFT.

    Parameters:
    -----------
    self : TemporalFusionTransformer
        The model instance.
    lr : float
        Learning rate to use in the optimizer.

    Returns:
    --------
    dict : Optimizer and LR scheduler config for PyTorch Lightning
    """
    optimizer = AdamW(  # Use AdamW with weight decay and epsilon
        self.parameters(),
        lr=lr,
        weight_decay=1e-4,
        eps=1e-6,
    )
    scheduler = ReduceLROnPlateau(  # Learning rate scheduler that reduces LR on plateau
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
            "monitor": "val_loss",  # Watch val_loss to trigger scheduler
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

    # Step 1: Build provisional model for LR tuning (placeholder LR)
    tft = TemporalFusionTransformer.from_dataset(
        tft_ds,
        learning_rate=1e-4,  # Placeholder, will be tuned later
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

    # Step 2: Create a minimal trainer for LR tuning
    tuner_trainer = Trainer(
        accelerator=device,  # Use GPU if available
        devices=1,
        gradient_clip_val=0.1,  # Prevent exploding gradients
        enable_progress_bar=True,
    )

    # Step 3: Run learning rate finder to get best LR suggestion
    res = Tuner(tuner_trainer).lr_find(
        model=tft,
        train_dataloaders=train_dl,
        val_dataloaders=val_dl,
        min_lr=1e-6,
        max_lr=10.0,
        num_training=300,
    )
    suggested_lr = res.suggestion()
    print(f"Suggested learning rate: {suggested_lr:.2e}")
    res.plot(suggest=True, show=True)

    # Step 4: Rebuild model with tuned learning rate
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

    # Inject custom optimizer logic into the model
    tft.configure_optimizers = lambda: tft_configure_custom_optimizers(
        tft, suggested_lr
    )

    # Step 5: Set up callbacks for early stopping, LR monitoring, and checkpointing
    project_root = Path(__file__).resolve().parents[2]
    model_save_dir = project_root / "results" / "models" 

    callbacks = [
        EarlyStopping(  # Stop training early if val_loss doesn't improve
            monitor="val_loss", patience=15, min_delta=0.003, mode="min", verbose=True
        ),
        LearningRateMonitor(logging_interval="step"),  # Log LR every step
        ModelCheckpoint(  # Save the best and latest model checkpoints
            dirpath=str(model_save_dir),
            filename="best-model-{epoch}-{val_loss:.4f}",
            monitor="val_loss",
            save_top_k=1,
            mode="min",
            verbose=True,
        ),
    ]
    precision = "bf16-mixed" if device == "cuda" else 32
    # Step 6: Initialize final trainer for model training
    trainer = Trainer(
        max_epochs=1,  # You can change this
        accelerator=device,  # Use GPU
        devices=1,
        precision=precision,  # Use bfloat16 mixed precision (AMP) or float32
        gradient_clip_val=0.5,  # Clip gradients
        callbacks=callbacks,  # Attach defined callbacks
        val_check_interval=0.5,  # Validate halfway through each epoch
        accumulate_grad_batches=1,
        deterministic=False,
        enable_progress_bar=True,
    )

    # Step 7: Start training
    trainer.fit(tft, train_dl, val_dl)

    # Step 8: Retrieve and load the best checkpoint for evaluation or deployment
    best_model_path = trainer.checkpoint_callback.best_model_path
    print(f"Best model checkpoint saved at: {best_model_path}")

    best_tft = TemporalFusionTransformer.load_from_checkpoint(
        best_model_path, map_location="cpu"
    )

    return best_tft, trainer
