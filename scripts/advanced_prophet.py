import sys
from pathlib import Path
current_file = Path(__file__).resolve()
project_root = current_file.parents[1]
src_path = project_root / "scripts" / "prophet"
sys.path.insert(0, str(src_path))
from model_optimization import model_optimization
from data_preprocess import data_preprocess
from train_prophet import train_prophet
import click

@click.command()
@click.option('--df', type=str, help="Path to training data")
@click.option('--result', type=str, help="Path to save model")
def main(df,result):
    df = data_preprocess(df)
    print(df.tail())
    model_optimization(df,result)
    train_prophet(df)


if __name__ == '__main__':
    main()

