"""Run the packaged staggered-adoption example."""
import argparse
from pathlib import Path
from pyautocausal import create_panel_graph, export_outputs
from pyautocausal.datasets import minimum_wage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('output/minimum_wage'))
    args = parser.parse_args()
    data = minimum_wage.copy()
    graph = create_panel_graph(args.output).fit(df=data)
    export_outputs(graph, df=data, output_path=args.output)


if __name__ == '__main__':
    main()
