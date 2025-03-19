import pandas as pd
from ProMap import ProductsDatasets
import argparse
import NeuroEvolution
import NeatConfigParser

def generate_configs(config_directory : str, input_path: str,  generate : bool = True, add_defaul : bool = True):
    if not generate:
        return

    parser = NeatConfigParser.NeatConfigParser(config_directory)
    parser.createConfig(input=input_path,add_default_values=add_defaul)

def main(args: argparse.Namespace):
    generate_configs(config_directory=args.config_directory, input_path=args.input, generate=args.config_generation, add_defaul=args.default)
    
    a = ProductsDatasets.Load_extended_promap_cz()
    evolution = NeuroEvolution.Evolution("Config/ProMapCz", a, scaling=args.scale, dimension_reduction=args.dimension_reduction)
    evolution.run(args.iterations, args.parallel)

    print(f"validation of: {evolution.validation()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Neat arguments
    parser.add_argument('--parallel', '--par',  action='store_true', default=False, help="Runs Neat in parallel.",)
    parser.add_argument('--iterations', '--iter', default=50, type=int, help='Number of generations in neat')

    # dataset preprocessing
    parser.add_argument('--dimension_reduction', '--dims',default='raw', choices=['raw', 'lda', 'pca'],type=str.lower, help="Specify the dimension reduction technique: 'raw', 'lda', or 'pca'")
    parser.add_argument('--scale', '--s',default=False, action='store_true', help="Standardize data")

    # Config generation
    parser.add_argument('--config_directory', default='ConfigGeneration', type=str, help='Directory name in which all generated configs are saved')
    parser.add_argument('--config_generation', '-g', default=True, action='store_false',help='Disables config generation')
    parser.add_argument('--input', '--i', type=str, default='input/input.json', help='Path to config generation input.')
    parser.add_argument('--default', '--d', action='store_false', default=True, help='Disables default value generations in config.' )
    main(parser.parse_args())

