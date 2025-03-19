import pandas as pd
from ProMap import ProductsDatasets
import argparse
import NeuroEvolution
import NeatConfigParser

def generate_configs(config_path : str,  generate : bool = True, add_defaul : bool = True):
    if not generate:
        return
    parser = NeatConfigParser.NeatConfigParser(config_path)
    parser.createConfig(add_default_values=add_defaul)

def main(args: argparse.Namespace):

    a = ProductsDatasets.Load_extended_promap_cz()
    evolution = NeuroEvolution.Evolution("Config/ProMapCz", a, scaling=args.scale, dimension_reduction=args.dimension_reduction)
    
    evolution.run(args.iterations, args.parallel)

    print(f"validation of: {evolution.validation()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Neat arguments
    parser.add_argument('--parallel', '--par',  action='store_true', default=False, help="Runs Neat in parallel.",)
    parser.add_argument('--iterations', '--iter', default=50, type=int, help='Number of generations in neat')

    # dataset prepro
    parser.add_argument('--dimension_reduction', '--dims',default='raw', choices=['raw', 'lda', 'pca'],type=str.lower, help="Specify the dimension reduction technique: 'raw', 'lda', or 'pca'")
    parser.add_argument('--scale', '--s',default=False, action='store_true', help="Standardize data")

    # Config generation
    parser.add_argument('--config_path', default='Conig/', type=str, help='Path to configs')
    parser.add_argument('--config_generation', '-g', default=True, action='store_false',help='Disables config generation')

    main(parser.parse_args())

