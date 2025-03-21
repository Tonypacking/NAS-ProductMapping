import pandas as pd
import sklearn.metrics
from ProMap import ProductsDatasets
import argparse
import NeuroEvolution
import NeatConfigParser
import os 
import pickle
import numpy as np
import sklearn 
import matplotlib.pyplot as plt

def generate_configs(config_directory : str, input_path: str,  generate : bool = True, add_defaul : bool = True) : 
    """Helper function to generate config files.

    Args:
        config_directory (str): Path to config directory in which all generated configs will be saved.
        input_path (str): Path to input json file
        generate (bool, optional): If new configs are to be generated. Defaults to True.
        add_defaul (bool, optional): Explicitly adds default neat values to the config. Defaults to True.

    Returns:
        None
    """
    if not generate:
        return

    parser = NeatConfigParser.NeatConfigParser(config_directory)
    return parser.createConfig(input=input_path,add_default_values=add_defaul)


def main(args: argparse.Namespace):
    """Main function in which neuron architecture search runs.

    Args:
        args (argparse.Namespace): User's arguments.
    """
    generated_configs = generate_configs(config_directory=args.config_directory, input_path=args.input, generate=args.config_generation, add_defaul=args.default)
    
    if args.all_files:
        configs = [x.name for x in os.scandir(args.config_directory) if x.name.endswith(NeatConfigParser.NeatConfigParser.SUFFIX)]
    else:
        configs = generated_configs

    best_networks = []
    for config in configs:
        data = ProductsDatasets.Load_by_name(args.dataset)
        # create output path directory -  args.output
        if not os.path.isdir(args.output):
            os.mkdir(args.output)

        evolution = NeuroEvolution.Evolution(config, data, scaling=args.scale, dimension_reduction=args.dimension_reduction)

        # extract folder name in which we will save our results.
        folder_name = config.split('/')[-1][:-len(NeatConfigParser.NeatConfigParser.SUFFIX)]
        used_preprocessing = '_'

        if args.scale:
            used_preprocessing += 'Standard_Scaling_'

        used_preprocessing += args.dimension_reduction

        output_path = os.path.join(args.output, evolution.dataset_name+used_preprocessing, folder_name)

        if not os.path.isdir(output_path):
            os.makedirs(output_path, exist_ok=True)

        evolution.run(args.iterations, args.parallel)

        if args.validate_all:
            outputs = evolution.validate_all()
        else:
            outputs = [(evolution.dataset_name, evolution.validate())]
        

        for dataset_name, output in outputs:
            with open(os.path.join(output_path, f'{dataset_name}_validation_results.txt'), mode='w') as f :
                for key, value in output.items():
                    # special case : confusion matrix is printited as a png file
                    if isinstance(value, np.ndarray):
                        display = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix=value, display_labels= ['Non-Match','Match'])
                        display.plot(cmap='Blues', values_format='d',)
                        plt.tight_layout()
                        plt.savefig(os.path.join(output_path, f'{dataset_name}_confusion_matrix.png'))
                        plt.close()
                        continue

                    f.write(f"{key}: {str(value)}\n\n")

                    if key == 'f1_score':
                        best_networks.append((value, dataset_name+used_preprocessing+'_'+folder_name))
        
        evolution.plot_network(os.path.join(output_path,'BestNetwork'))
        evolution.plot_statistics(os.path.join(output_path,'Statistics'))

        with open(os.path.join(output_path,'best_network'), 'wb') as f:
            if evolution.Best_network is not None:
                pickle.dump(evolution.Best_network,f)

    best_networks.sort(key=lambda x: x[0], reverse=True)
    best_networks_path = os.path.join(args.output,'best_networks',evolution.dataset_name+used_preprocessing)

    if not os.path.isdir(best_networks_path):
        os.makedirs(best_networks_path, exist_ok=True)

    with open(best_networks_path+'best', mode='w') as f :
        for rank, (value, path) in enumerate(best_networks[: args.kbest], start=1):
            f.write(f"Rank: {rank} with f1_score: {value}.   Attributes: {path}\n")

    print(best_networks[:args.kbest])
    
    
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
    parser.add_argument('--output', '--o', type=str.lower, default='output', help='Output directory name.')
    parser.add_argument('--validate_all', '--v', action='store_true', default=False, help='Validates input against all possible datasets. If feature count is not same, it is ignored')
    parser.add_argument('--kbest', '--k', default=10,type=int, help='prints k best networks')
    # Config generation
    parser.add_argument('--config_directory', '--dir', default='ConfigGeneration', type=str, help='Directory name in which all generated configs are saved')
    parser.add_argument('--config_generation', '--g', default=True, action='store_false',help='Disables config generation')
    parser.add_argument('--input', '--i', type=str, default='input/input.json', help='Path to config generation input.')
    parser.add_argument('--default','--def', action='store_true', default=False, help='Disables default value generations in config.' )
    parser.add_argument('--all_files','--all', action='store_true', default=False, help='Generates configs from all ini files in config directory set by config_directory argument. ')
    
    main(parser.parse_args())

