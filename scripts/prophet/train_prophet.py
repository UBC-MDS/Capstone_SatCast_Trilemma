import sys, logging
from pathlib import Path
from prophet.serialize import model_to_json

logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet.plot").disabled = True         # plotly spam

root = Path(__file__).resolve().parents[2]
sys.path.append(str(root / "src"))

from prophet_utils   import create_model_new
from data_preprocess import data_preprocess

def train_prophet(raw_df):
    df = data_preprocess(raw_df)

    m = create_model_new()   # your helper; must NOT call .fit()

    # ONE, plain fit → 1 optimiser pass in Prophet 1.1.6
    m.fit(df)                # no kwargs → no 10-restart path

    out = root / "results" / "models" / "prophet_model.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(model_to_json(m))      # official serialisation API:contentReference[oaicite:1]{index=1}
    print("Prophet model saved ➜", out)
