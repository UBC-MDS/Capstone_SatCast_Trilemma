"""
train_tft_model.py

Script to train a Temporal Fusion Transformer (TFT) model using PyTorch Lightning
with a custom loss function and optimizer for Bitcoin fee forecasting.

This script assumes the following:
- `train_ds`, `train_dl`, and `val_dl` are pre-defined.
- `MAEWithDynamicStdAndDeviationPenalty` is implemented and imported.
- `res.suggestion()` is obtained from an Optuna study or similar hyperparameter tuning result.

Author: Your Name
Date: 2025-06-06
"""

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_forecasting import TemporalFusionTransformer
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Define model
def instantiate_tft(train_ds, learning_rate):
    """
    Instantiate the Temporal Fusion Transformer with the given dataset and learning rate.
    
    Parameters:
    -----------
    train_ds : TimeSeriesDataSet
        The dataset used to define input/output sizes and feature configurations.
    learning_rate : float
        Initial learning rate from hyperparameter optimization.

    Returns:
    --------
    TemporalFusionTransformer
        Configured TFT model.
    """
    return TemporalFusionTransformer.from_dataset(
        train_ds,
        learning_rate=learning_rate,
        hidden_size=32,
        hidden_continuous_size=8,
        lstm_layers=2,
        dropout=0.2,
        loss=MAEWithDynamicStdAndDeviationPenalty(beta=1.0, clip_weight=10.0),
        output_size=1,
        weight_decay=1e-4,
        reduce_on_plateau_patience=0,
    )


# Define optimizer and scheduler logic
def configure_optimizers_fix(self):
    """
    Custom optimizer configuration for the TFT model, using AdamW and ReduceLROnPlateau.
    """
    optimizer = AdamW(
        self.parameters(),
        lr=res.suggestion(),
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


# Define callbacks
def get_callbacks():
    """
    Get callbacks for training monitoring and model checkpointing.

    Returns:
    --------
    list
        List of PyTorch Lightning callbacks.
    """
    return [
        EarlyStopping(
            monitor="val_loss",
            patience=15,
            min_delta=0.003,
            mode="min",
            verbose=True,
        ),
        LearningRateMonitor(),
        ModelCheckpoint(
            dirpath="./saved_models",
            filename="best-model-{epoch}-{val_loss:.4f}",
            monitor="val_loss",
            save_top_k=1,
            mode="min",
            save_last=True,
            verbose=True,
        ),
    ]


# Training script
def train_model(train_ds, train_dl, val_dl, learning_rate):
    """
    Train the TFT model using given dataloaders and learning rate.
    
    Parameters:
    -----------
    train_ds : TimeSeriesDataSet
    train_dl : DataLoader
    val_dl   : DataLoader
    learning_rate : float
    
    Returns:
    --------
    model : TemporalFusionTransformer
        The trained TFT model loaded from best checkpoint.
    """
    # Instantiate model and override optimizer
    tft = instantiate_tft(train_ds, learning_rate)
    tft.configure_optimizers = configure_optimizers_fix.__get__(tft, type(tft))

    # Set up trainer
    trainer = pl.Trainer(
        max_epochs=100,
        accelerator="gpu",
        devices=1,
        precision="bf16-mixed",
        gradient_clip_val=0.5,
        callbacks=get_callbacks(),
        accumulate_grad_batches=1,
        deterministic=False,
        enable_progress_bar=True,
        val_check_interval=0.5,
    )

    # Train the model
    trainer.fit(tft, train_dl, val_dl)

    # Load best model from checkpoint
    best_model_path = trainer.checkpoint_callback.best_model_path
    if not best_model_path:
        best_model_path = "./saved_models/best-model-epoch=16-val_loss=0.7079.ckpt"

    model = TemporalFusionTransformer.load_from_checkpoint(
        checkpoint_path=best_model_path,
        map_location="cuda"
    )

    return model


# Example entry point (only run if this file is executed directly)
if __name__ == "__main__":
    # Replace these with actual objects
    # from your_pipeline import train_ds, train_dl, val_dl, res
    raise NotImplementedError("Replace the placeholders with actual training components before running.")
