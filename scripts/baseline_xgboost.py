import sys
from pathlib import Path
current_file = Path(__file__).resolve()
project_root = current_file.parents[1]
src_path = project_root / "scripts" / "xgboost"
sys.path.insert(0, str(src_path))
from model_optimization import optimization
import click

@click.command()
@click.option('--data-path', type=str, help="Path to training data")
@click.option('--result', type=str, help="Path to save model")

def main(data_path,result):
    optimization(data_path,result)

if __name__ == '__main__':
    main()
