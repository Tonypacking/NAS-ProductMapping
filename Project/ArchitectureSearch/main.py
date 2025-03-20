import pandas as pd
from ProMap import ProductsDatasets
import argparse
import NeuroEvolution
import NeatConfigParser
import os 
import pickle
import numpy as np
def generate_configs(config_directory : str, input_path: str,  generate : bool = True, add_defaul : bool = True) : 
    if not generate:
        return

    parser = NeatConfigParser.NeatConfigParser(config_directory)
    return parser.createConfig(input=input_path,add_default_values=add_defaul)


def main(args: argparse.Namespace):
    generated_configs = generate_configs(config_directory=args.config_directory, input_path=args.input, generate=args.config_generation, add_defaul=args.default)
    
    if args.all_files:
        configs = [x.name for x in os.scandir(args.config_directory) if x.name.endswith(NeatConfigParser.NeatConfigParser.SUFFIX)]
    else:
        configs = generated_configs

    for config in configs:
        data = ProductsDatasets.LoadByName(args.dataset)
        # create output path directory -  args.output
        if not os.path.isdir(args.output):
            os.mkdir(args.output)
        
        evolution = NeuroEvolution.Evolution(config, data, scaling=args.scale, dimension_reduction=args.dimension_reduction)

        # extract folder name in which we will save our results.
        folder_name = config.split('/')[-1][:-len(NeatConfigParser.NeatConfigParser.SUFFIX)]

        output_path = os.path.join(args.output, evolution.dataset_name, folder_name)

        if not os.path.isdir(output_path):
            os.makedirs(output_path, exist_ok=True)

        evolution.run(args.iterations, args.parallel)
        output = evolution.validation()
        np.set_printoptions(linewidth=np.inf)
        print(output)
        with open(os.path.join(output_path, 'Validation results:'), mode='w') as f :
            for key, value in output.items():
                # special case confusion matrix
                if isinstance(value, np.ndarray):
                    value = [list(x) for x in value]
                f.write(f"{key}: {str(value)}\n\n")

                
        evolution.visualize(os.path.join(output_path,'BestNetwork'))
        with open(os.path.join(output_path,'best_network'), 'wb') as f:
            if evolution.Best_network is not None:
                pickle.dump(evolution.Best_network,f)
    
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
    parser.add_argument('--output', '-o', type=str.lower, default='output', help='Output directory name.')

    # Config generation
    parser.add_argument('--config_directory', default='ConfigGeneration', type=str, help='Directory name in which all generated configs are saved')
    parser.add_argument('--config_generation', '-g', default=True, action='store_false',help='Disables config generation')
    parser.add_argument('--input', '--i', type=str, default='input/input.json', help='Path to config generation input.')
    parser.add_argument('--default', action='store_false', default=True, help='Disables default value generations in config.' )
    parser.add_argument('--all_files','--n', action='store_true', default=False, help='Generates configs from all ini files in config directory set by config_directory argument. ')
    
    main(parser.parse_args())

