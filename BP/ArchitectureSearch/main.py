import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..'))) # To load Utils module

import csv
import pandas as pd
import sklearn.metrics
from Utils.ProMap import ProductsDatasets
import argparse
import NeuroEvolution
import NeatConfigParser
import pickle
import numpy as np
import sklearn 
import matplotlib.pyplot as plt
import random
import logging
# TODO import hyperNEAT and ES hyperNEAT
from BP.ArchitectureSearch.CoDeepNEAT.NAS.CoDeapNEATProductMapping import CoDeepNEAT
NEAT_METHOD = 'BasicNEAT'
ALL = 'all'
HYPERNEAT_METHOD = 'HyperNEAT'
CODEAPNEAT_METHOD = 'CoDeepNEAT'
METHOD_CHOICES = [ALL,NEAT_METHOD, HYPERNEAT_METHOD, CODEAPNEAT_METHOD]

# NAS methods directory names
NEAT_DIRECTORY = 'NEAT'
HYPERNEAT_DIRECTORY = 'HyperNEAT'
CODEEPNEAT_DIRECTORY = "CoDeepNEAT"

CSV_HEADER = ['TRAINING_DATASET', 'TESTING_DATASET','SCALED','DIMENSION_REDUCTION','METHOD','PARAMETERS', 'F1_SCORE', 'ACCURACY', 'PRECISION', 'RECALL', 'BALANCED_ACCURACY']
GLOBAL_FILE = 'global_results.csv'

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def Generate_configs(config_directory : str, input_path: str,method:str,  generate : bool = True, add_defaul : bool = True) : 
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
    logger.info(f"Generating configs in {config_directory} from {input_path} with method {method}.")
    parser = NeatConfigParser.NeatConfigParser(config_directory)
    return parser.CreateNeatConfig(input=input_path,add_default_values=add_defaul, method=method)

def main(args: argparse.Namespace):
    """Main function in which neuron architecture search runs.

    Args:
        args (argparse.Namespace): User's arguments.
    """
    np.random.seed(seed=args.seed)
    random.seed(args.seed)
    logger.info(f"Running Neuron Architecture Search with arguments: {args}")

    if args.dataset == ALL:
        available_datasets = list(ProductsDatasets.NAME_MAP.keys())
    else:
        available_datasets = [args.dataset]
    
    for dataset in available_datasets:
        #HACK: Instead of passing dataset to all methods, change it globaly here.
        args.dataset = dataset

        logger.info(f"Running Neuron Architecture Search for dataset: {dataset}")
        # TODO add more NAS methods like random search, DARTS and RL based NAS
        if args.NAS_method == NEAT_METHOD or args.NAS_method == ALL:
            logger.info(f"Running NEAT for dataset: {dataset}")
            Neat_Nas(args=args)
            logger.info(f"Finished NEAT for dataset: {dataset}")

        if args.NAS_method == HYPERNEAT_METHOD or args.NAS_method == ALL:
            logger.info(f"Running HyperNEAT for dataset: {dataset}")
            EsHyperNeatNas(args=args)
            logger.info(f"Finished HyperNEAT for dataset: {dataset}")
        
        if args.NAS_method ==CODEAPNEAT_METHOD or args.NAS_method == ALL:
            logger.info("Running CoDeepNEAT for dataset: {dataset}")
            CoDeepNeat(args)
            logger.info(f"Finished CoDeepNEAT for dataset: {dataset}")

def Write_Global_Result(args: argparse.Namespace, row : list):
    """Writes global resutls to a args.output/global_results.csv file.
    If the file does not exist, it creates it with a header.
    If the row length does not match the header length, it logs a warning and does not write the row.
    If args.remove_global_results is set to True, it clears the global_results.csv.

    Args:
        args (argparse.Namespace): user's arguments.
        row (list): Row to be written to the global results file.
    """
    if len(row) != len(CSV_HEADER):
        logging.warning(f"Row length {len(row)} does not match header length {len(CSV_HEADER)}. Row will not be written.")
        logging.warning(f"Row: {row}")
        return

    path = os.path.join(args.output, GLOBAL_FILE)
    # check if file doesnt exist or remove global is set or there isnt any data
    if not os.path.exists(path) or args.remove_global_results or os.path.getsize(path) == 0:
        # result were cleared 
        args.remove_global_results = False
        with open(path, mode='w') as f:
            csw_writer = csv.writer(f)
            csw_writer.writerow(CSV_HEADER)

    with open(path, mode='a') as f:
        #write row
        csw_writer = csv.writer(f)
        csw_writer.writerow(row)

def Plot_Confusion_Matrix(confusion_matrix: np.ndarray, output_path: str, TrainingDatasetNAme: str, TestingDatasetName:str ):
    """Plots confusion matrix and saves it to the output path.

    Args:
        confusion_matrix (np.ndarray): Confusion matrix to be plotted.
        output_path (str): Path to the output directory where the confusion matrix will be saved.
        TrainingDatasetNAme (str): Training dataset which was used to train the model.
        TestingDatasetName (str): Testing dataset which was used to test the model and plots the confusion matrix with testing data.
    """
    display = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels= ['Non-Match','Match'])
    display.plot(cmap='Blues', values_format='d',)
    plt.title(f'Confusion Matrix for {TestingDatasetName} \ntrained on {TrainingDatasetNAme}')
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f'Train_{TrainingDatasetNAme}_{TestingDatasetName}_confusion_matrix.png'))
    plt.close()

def Neat_Nas(args: argparse.Namespace):
    """Runs neuron architecture search using NEAT.

    1. Generates configs if config_generation is set to True.
        2. Loads dataset specified by dataset argument.
        3. Runs NEAT for each config.
        4. Validates each config against dataset specified by validate_all argument.
        5. Saves validation results in output directory.
        6. Saves best networks in output directory.
    7. Plots best network and statistics.
    8. Saves best networks in a separate directory.
    9. Prints k best networks based on f1_score.
    Args:
        args (argparse.Namespace): User's arguments.
    """
    generated_configs = Generate_configs(config_directory=args.config_directory, input_path=args.input, generate=args.config_generation, add_defaul=args.default, method=NEAT_METHOD)
    # print(f"Generated configs: {generated_configs}")

    if args.all_files:
        configs = [x.name for x in os.scandir(args.config_directory) if x.name.endswith(NeatConfigParser.NeatConfigParser.NEAT_SUFFIX)]
    else:
        configs = generated_configs

    best_networks = []
    for config in configs:
        # load training dataset

        data = ProductsDatasets.Load_by_name(args.dataset)
        # create output path directory -  args.output/NEAT
        neat_dir = os.path.join(args.output, NEAT_DIRECTORY)
        if not os.path.isdir(neat_dir):
            os.makedirs(neat_dir,exist_ok=True)

        evolution = NeuroEvolution.NEATEvolution(config, data, scaling=args.scale, dimension_reduction=args.dimension_reduction)

        # extract used parameters and save it as a folder name in which we will save our results.
        folder_name = config.split('/')[-1][:-len(NeatConfigParser.NeatConfigParser.NEAT_SUFFIX)]
        used_preprocessing = '_'

        if args.scale:
            used_preprocessing += 'Standard_Scaling_'

        used_preprocessing += args.dimension_reduction

        output_path = os.path.join(neat_dir, evolution.dataset_name+used_preprocessing, folder_name)

        if not os.path.isdir(output_path):
            os.makedirs(output_path, exist_ok=True)

        evolution.RunNEAT(args.iterations, args.parallel)

        if args.validate_all:
            outputs = evolution.validate_all()
        else:
            outputs = [(evolution.dataset_name, evolution.validate())]

        with open(os.path.join(output_path, f'{folder_name}_local_scores.csv'), mode='w') as f:
            # Create csv writer and write header
            csw_writer = csv.writer(f)
            csw_writer.writerow(CSV_HEADER)   

            confusion_matrix_path = os.path.join(output_path, 'confusion_matrices')
            if not os.path.isdir(confusion_matrix_path):
                os.makedirs(confusion_matrix_path, exist_ok=True) 
            for dataset_name, output in outputs:
                # Save to local results

                row = [data.dataset_name, dataset_name,args.scale,args.dimension_reduction,NEAT_METHOD,folder_name, output['f1_score'], output['accuracy'], output['precision'], output['recall'], output['balanced_accuracy']]
                csw_writer.writerow(row)
                # Save results to global results too
                Write_Global_Result(args, row)

                Plot_Confusion_Matrix(output['confusion_matrix'], confusion_matrix_path, evolution.dataset_name, dataset_name)
                # save network to the best network
                best_networks.append((output['f1_score'], dataset_name+used_preprocessing+'_'+folder_name))
        # Plot NN statistics and network
        evolution.plot_network(os.path.join(output_path,'BestNetwork'))
        evolution.plot_statistics(os.path.join(output_path,'Statistics'))

        with open(os.path.join(output_path,'best_network'), 'wb') as f:
            if evolution.Best_network is not None:
                pickle.dump(evolution.Best_network,f)

    best_networks.sort(key=lambda x: x[0], reverse=True)
    best_networks_path = os.path.join(neat_dir,'best_networks',evolution.dataset_name+used_preprocessing)

    if not os.path.isdir(best_networks_path):
        os.makedirs(best_networks_path, exist_ok=True)

    with open(best_networks_path+'best', mode='w') as f :
        for rank, (value, path) in enumerate(best_networks[: args.kbest], start=1):
            f.write(f"Rank: {rank} with f1_score: {value}.   Attributes: {path}\n")

    # print(best_networks[:args.kbest])
    logger.info(f"Best networks for training dataset: {evolution.dataset_name}")

    for accuracy, dataset_name in best_networks[:args.kbest]:
        logger.info(f"Best network for {dataset_name} has : {accuracy}")

def EsHyperNeatNas(args: argparse.Namespace):
    """Runs neuron architecture search using HyperNEAT.
    Args:
        args (argparse.Namespace): User's arguments.
    """
    if args.all_files:
        configs = [x.name for x in os.scandir(args.config_directory) if x.name.endswith(NeatConfigParser.NeatConfigParser.HYPERNEAT_SUFFIX)]
    else:
        configs = Generate_configs(config_directory=args.config_directory, input_path=args.input, generate=args.config_generation, add_defaul=args.default, method=HYPERNEAT_METHOD)
    best_networks = []
    
    for config in configs:
        data = ProductsDatasets.Load_by_name(args.dataset)
        hyperNEAT_dir = os.path.join(args.output, HYPERNEAT_DIRECTORY)
        if not os.path.isdir(hyperNEAT_dir):
            os.makedirs(hyperNEAT_dir,exist_ok=True)
        
        evolution = NeuroEvolution.HyperNEATEvolution(config,version=args.hyper_size, dataset=data, scaling=args.scale, dimension_reduction=args.dimension_reduction)

        # extract used parameters and save it as a folder name in which we will save our results.
        folder_name = config.split('/')[-1][:-len(NeatConfigParser.NeatConfigParser.HYPERNEAT_SUFFIX)]
        used_preprocessing = '_'

        if args.scale:
            used_preprocessing += 'Standard_Scaling_'

        used_preprocessing += args.dimension_reduction

        output_path = os.path.join(hyperNEAT_dir, evolution.dataset_name+used_preprocessing, folder_name)
        evolution.RunHyperNEAT(args.iterations, args.parallel)

        if args.validate_all:
            outputs = evolution.validate_all()
        else:
            outputs = [(evolution.dataset_name, evolution.validate())]


        if not os.path.isdir(output_path):
            os.makedirs(output_path, exist_ok=True)

        with open(os.path.join(output_path, f'{folder_name}_local_scores.csv'), mode='w') as f:
            # Create csv writer and write header
            csw_writer = csv.writer(f)
            csw_writer.writerow(CSV_HEADER)   

            confusion_matrix_path = os.path.join(output_path, 'confusion_matrices')
            if not os.path.isdir(confusion_matrix_path):
                os.makedirs(confusion_matrix_path, exist_ok=True) 

            for dataset_name, output in outputs:
                # Save to local results

                row = [data.dataset_name, dataset_name,args.scale,args.dimension_reduction,HYPERNEAT_METHOD,folder_name, output['f1_score'], output['accuracy'], output['precision'], output['recall'], output['balanced_accuracy']]
                csw_writer.writerow(row)
                # Save results to global results too
                Write_Global_Result(args, row)

                Plot_Confusion_Matrix(output['confusion_matrix'], confusion_matrix_path, evolution.dataset_name, dataset_name)
                # save network to the best network
                best_networks.append((output['f1_score'], dataset_name+used_preprocessing+'_'+folder_name))

        # Plot NN statistics and network


        #evolution.plot_network(os.path.join(output_path,'BestNetwork'))
        evolution.plot_statistics(os.path.join(output_path,'Statistics'))
        evolution.plot_CPPN_network(os.path.join(output_path,'CPPN'))
        evolution.plot_best_network(os.path.join(output_path,'BestNetwork'))

def CoDeepNeat(args: argparse.Namespace):
    co_deep_neat = CoDeepNEAT()

    co_deep_neat.RunCoDeepNEAT(args)

    if args.validate_all:
        outputs = co_deep_neat.validate_all()
    else:
        outputs = [(co_deep_neat._train_dataset.dataset_name, co_deep_neat.validate())]

    output_path = os.path.join(args.output,CODEEPNEAT_DIRECTORY)
    if not os.path.isdir(output_path):
        os.makedirs(output_path, exist_ok=True)

    with open(os.path.join(output_path, f'{CODEEPNEAT_DIRECTORY}_local_scores.csv'), mode='w') as f:
        # Create csv writer and write header
        csw_writer = csv.writer(f)
        csw_writer.writerow(CSV_HEADER)  

        for dataset_name, output in outputs:
            # Save to local results
            row = [args.dataset, dataset_name,args.scale,args.dimension_reduction,CODEAPNEAT_METHOD,co_deep_neat.training_parameters, output['f1_score'], output['accuracy'], output['precision'], output['recall'], output['balanced_accuracy']]
            csw_writer.writerow(row)
            # Save results to global results too
            Write_Global_Result(args, row)
        # TODO write the output in output.csv and Global csv result

if __name__ == "__main__":

    logging.basicConfig(
        level=logging.WARNING,
        format='- %(name)s - %(levelname)s - %(message)s',
    )
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--save', type=str, default='Saves/', help='Path to a save directory')
    parser.add_argument('--seed', default=42, type=int, help='Sets a seed to random number generation.')


    # Neat arguments
    parser.add_argument('--parallel', '--par',  action='store_true', default=False, help="Runs Neat in parallel.",)
    parser.add_argument('--iterations', '--iter', default=50, type=int, help='Number of generations in neat')

    # HyperNEAT arguments
    parser.add_argument('--hyper_size', '--hs', default='S', choices=['S', 'M', 'L'], type=str.upper, help='Size of the hyperneat network. S - small, M - medium, L - large')


    # dataset preprocessing arguments
    parser.add_argument('--dimension_reduction', '--dims',default='raw', choices=['raw', 'lda', 'pca'],type=str.lower, help="Specify the dimension reduction technique: 'raw', 'lda', or 'pca'")
    parser.add_argument('--scale', '--s',default=False, action='store_true', help="Standardize data")
    
    # dataset arguments
    available_datasets = list(ProductsDatasets.NAME_MAP.keys()) + [ALL]
    parser.add_argument('--dataset', '--d',default='promapcz', type=str.lower, choices=available_datasets, help='name of promap dataset or path')
    # output arguments
    logger.debug(f"{available_datasets}")
    parser.add_argument('--output', '--o', type=str.lower, default='output', help='Output directory name.')
    parser.add_argument('--validate_all', '--v', action='store_false', default=True, help="Validates input against all possible datasets. If feature count is not same, the testing dataset is extened or reduces to match the training dataset's features")
    parser.add_argument('--kbest', '--k', default=10,type=int, help='prints k best networks')
    parser.add_argument('--remove_global_results','--rg',action='store_true', default=False, help='Removes global results from global_results.csv. If not set, appends to the file.')
    # Config generation
    parser.add_argument('--config_directory', '--dir', default='ConfigGeneration', type=str, help='Directory name in which all generated configs are saved')
    parser.add_argument('--config_generation', '--g', default=True, action='store_false',help='Disables config generation')
    parser.add_argument('--input', '--i', type=str, default='input/input.json', help='Path to config generation input.')
    parser.add_argument('--default','--def', action='store_false', default=True, help='Disables default value generations in config.' )
    parser.add_argument('--all_files','--all', action='store_true', default=False, help='Generates configs from all .neat (.ini) files in config directory set by config_directory argument. ')
    
    parser.add_argument('--NAS_method','--nas', type=str, default=ALL, choices=METHOD_CHOICES, help='Selects the method of NAS.')

    main(parser.parse_args())

