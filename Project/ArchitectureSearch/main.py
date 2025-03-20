import pandas as pd
from ProMap import ProductsDatasets
import argparse
import NeuroEvolution
import NeatConfigParser
import os 

def generate_configs(config_directory : str, input_path: str,  generate : bool = True, add_defaul : bool = True) : 
    if not generate:
        return

    parser = NeatConfigParser.NeatConfigParser(config_directory)
    return parser.createConfig(input=input_path,add_default_values=add_defaul)


def main(args: argparse.Namespace):
    generated_configs = generate_configs(config_directory=args.config_directory, input_path=args.input, generate=args.config_generation, add_defaul=args.default)
    
    if args.all_files:
        configs = [x.name for x in os.scandir(args.config_directory) if x.name.endswith(NeatConfigParser.SUFFIX)]
    else:
        configs = generated_configs

    for config in configs:
        print(config)
    a = ProductsDatasets.LoadByName(args.dataset)

    config = "Config/ProMapCz"
    evolution = NeuroEvolution.Evolution(config, a, scaling=args.scale, dimension_reduction=args.dimension_reduction)
    evolution.run(args.iterations, args.parallel)
    print(evolution.Name)
    print(f"validation of: {evolution.validation()}")
    # TODO create a directory visualization and store there the png file, confusion matrix, f1 scores and other values
    # also 1. Fitness Over Generations:
    # 3. Distribution of Fitness Scores:
    # 4. Species Diversity:
    
    # TODO check if output path exists
    evolution.visualize('vizual/a')

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Neat arguments
    parser.add_argument('--parallel', '--par',  action='store_true', default=False, help="Runs Neat in parallel.",)
    parser.add_argument('--iterations', '--iter', default=50, type=int, help='Number of generations in neat')

    # dataset preprocessing arguments
    parser.add_argument('--dimension_reduction', '--dims',default='raw', choices=['raw', 'lda', 'pca'],type=str.lower, help="Specify the dimension reduction technique: 'raw', 'lda', or 'pca'")
    parser.add_argument('--scale', '--s',default=False, action='store_true', help="Standardize data")
    
    # dataset arguments
    parser.add_argument('--dataset', '--d',default='promapcz', type=str.lower, choices=ProductsDatasets.NAME_MAP.keys(), help='name of promap dataset or path')

    # output arguments
    parser.add_argument('--output', '-o', type=str.lower, default='output', help='Output path')

    # Config generation
    parser.add_argument('--config_directory', default='ConfigGeneration', type=str, help='Directory name in which all generated configs are saved')
    parser.add_argument('--config_generation', '-g', default=True, action='store_false',help='Disables config generation')
    parser.add_argument('--input', '--i', type=str, default='input/input.json', help='Path to config generation input.')
    parser.add_argument('--default', action='store_false', default=True, help='Disables default value generations in config.' )
    parser.add_argument('--all_files','--n', action='store_true', default=False, help='Generates configs from all ini files in config directory set by config_directory argument. ')
    
    main(parser.parse_args())

