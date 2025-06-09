"""
tft_load_best_model.py

Load best-performing TFT model from checkpoint.
"""

from pytorch_forecasting import TemporalFusionTransformer

def tft_load_best_model(trainer, fallback_path):
    """
    Load best model checkpoint or fallback.

    Parameters:
    -----------
    trainer : pl.Trainer
    fallback_path : str

    Returns:
    --------
    model : TemporalFusionTransformer
    """
    best_path = trainer.checkpoint_callback.best_model_path or fallback_path
    return TemporalFusionTransformer.load_from_checkpoint(best_path, map_location="cuda")
