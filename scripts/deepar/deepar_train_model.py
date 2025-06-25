# deepar_train_model.py
# author: Ximin Xu
# date: 2025-06-18
"""
Trains a DeepAR model using PyTorch Forecasting for probabilistic Bitcoin transaction fee forecasting.

This script performs the following steps:
1. Initializes a DeepAR model with default parameters and placeholder learning rate.
2. Uses PyTorch Lightning's `Tuner.lr_find()` to find an optimal learning rate.
3. Rebuilds the model with the tuned learning rate.
4. Applies early stopping and model checkpointing callbacks.
5. Trains the model on CPU using the final Trainer configuration.
6. Saves the best-performing checkpoint to disk.

Usage:
------
Used after data preprocessing and dataloader construction as part of the DeepAR training pipeline.
The best model is saved to:

    results/models/best-deepar-full.ckpt
"""



import torch
from lightning.pytorch import Trainer
from lightning.pytorch.tuner import Tuner
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_forecasting.models.deepar import DeepAR
from pytorch_forecasting.metrics import MultivariateNormalDistributionLoss
from pathlib import Path

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu" # mps has a bug in deepAR, use cpu
def train_deepar_model(training, train_dl, val_dl):
    """
    Trains a DeepAR model with tuning and early stopping.

    Parameters
    ----------
    training : TimeSeriesDataSet
        Training dataset used to initialize the DeepAR model.
    train_dl : DataLoader
        DataLoader for training.
    val_dl : DataLoader
        DataLoader for validation.
    """
    project_root = Path(__file__).resolve().parents[2]
    model_save_dir = project_root / "results" / "models"

    # Step 1: Set up initial trainer for tuning
    tuner_trainer = Trainer(accelerator="cpu", devices=1, gradient_clip_val=0.1)

    # Step 2: Create DeepAR model for learning rate tuning
    net = DeepAR.from_dataset(
        training,
        learning_rate=3e-2,
        hidden_size=30,
        rnn_layers=2,
        loss=MultivariateNormalDistributionLoss(rank=30),
        optimizer="Adam",
    )

    # Step 3: Use learning rate finder
    res = Tuner(tuner_trainer).lr_find(
        net,
        train_dataloaders=train_dl,
        val_dataloaders=val_dl,
        min_lr=1e-5,
        max_lr=1e0,
        early_stop_threshold=100,
    )
    print(f"Suggested learning rate: {res.suggestion()}")
    net.hparams.learning_rate = res.suggestion()

    # Step 4: Setup early stopping & checkpoint
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=1e-4,
        patience=10,
        mode="min",
        verbose=False,
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=model_save_dir,
        filename="best-deepar-full",
        save_top_k=1,
        mode="min",
    )

    # Step 5: Trainer for real training
    trainer = Trainer(
        max_epochs=1,
        accelerator="cpu", 
        enable_model_summary=True,
        gradient_clip_val=0.1,
        callbacks=[early_stop_callback, checkpoint_callback],
        enable_checkpointing=True,
    )

    # Step 6: Recreate model with tuned learning rate
    net = DeepAR.from_dataset(
        training,
        learning_rate=net.hparams.learning_rate,
        hidden_size=30,
        rnn_layers=2,
        loss=MultivariateNormalDistributionLoss(rank=30),
        optimizer="Adam",
    )

    # Step 7: Train the model
    trainer.fit(net, train_dataloaders=train_dl, val_dataloaders=val_dl)

    # Step 8: Retrieve and load the best checkpoint
    best_model_path = checkpoint_callback.best_model_path
    print(f"Best model checkpoint saved at: {best_model_path}")
