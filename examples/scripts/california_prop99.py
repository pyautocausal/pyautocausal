"""Run a complete California Proposition 99 panel without sentinel imputation."""
import argparse
from pathlib import Path
from pyautocausal import create_panel_graph, export_outputs
from pyautocausal.datasets import california_prop99


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('output/california_prop99'))
    args = parser.parse_args()
    data = california_prop99.rename(columns={
        'state': 'id_unit', 'year': 't', 'treated': 'treat', 'cigsale': 'y',
    })[['id_unit', 't', 'treat', 'y']].copy()
    graph = create_panel_graph(args.output).fit(df=data)
    export_outputs(graph, df=data, output_path=args.output)


if __name__ == '__main__':
    main()
