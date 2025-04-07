import sklearn.exceptions
from WeightSearch_Evolution import Evo_WeightSearch, EvolutionaryNeuronNetwork
from WeightSearch_Backpropagation import Backprop_Weight_Search

import re
from pprint import pprint
from Utils.ProMap import ProductsDatasets
import argparse
import numpy as np
import random
import csv
import os
import sys
import sklearn
import warnings
import csv
from analyze_predictions import Analyzer

def log_statistics(save_path: str,train_dataset:str, method: str, statistics: list[tuple[str, dict[str, float]]], append = False, dimension_reduction = None):
    if not append:
        header = ['Train dataset','Tested dataset', 'Method']
        if dimension_reduction:
            header.append('Dimension reduction')
            header = header + [x for x in statistics[0][1].keys() if x != 'confusion_matrix']
    mode = 'w'

    if append: mode = 'a'

    with open(save_path, mode=mode, newline='') as file:            
        writer = csv.writer(file)
        if not append: writer.writerow(header)
        for test_data_name, score_dict in statistics:
            x = [train_dataset,test_data_name, method]
            if dimension_reduction:
                x.append(dimension_reduction)
            writer.writerow( x + [v for k, v in score_dict.items() if k != 'confusion_matrix'])

def run_all(args: argparse, validation_path):
    if args.run_all == 'all':
        dims = ['raw', 'lda', 'pca']
        promap_data = ProductsDatasets.NAME_MAP.keys()

    elif args.run_all == 'dim':
        promap_data = [args.dataset]
        dims = ['raw', 'lda', 'pca']

    elif args.run_all == 'dataset':
        promap_data = ProductsDatasets.NAME_MAP.keys()
        dims = [args.dimension_reduction]
    else:
        raise ValueError('Invalid argumnet.')
    append = False
    for dim_reduction in dims:
        args.dimension_reduction = dim_reduction
        for dataset_name in promap_data:
            args.dataset = dataset_name
            
            evo_path = os.path.join(args.save_evo, dim_reduction, dataset_name)
            back_path = os.path.join(args.save_back, dim_reduction, dataset_name)
            
            os.makedirs(name=evo_path, exist_ok=True)
            os.makedirs(name=back_path, exist_ok=True)

            # Backpropagation weight search     
            bp_weight_search = Backprop_Weight_Search(args)
            bp_weight_search.run(iterations=args.iterations)
            bp_weight_search.plot_bestmodel_accuracy_progress(back_path, show=False)
            ws_output = bp_weight_search.validate_all()
            
            ws_weights =  bp_weight_search.best_model.coefs_
            ws_biases = bp_weight_search.best_model.intercepts_

            log_statistics(save_path=validation_path,train_dataset= dataset_name, method='backpropagation',append=append,statistics=ws_output, dimension_reduction=dim_reduction)
            append = True
            # Evolutionary weight search
            eva_search = Evo_WeightSearch()
            eva_search.run(args, evo_path )

            eva_weights = eva_search._neuron_network._nn.coefs_
            eva_biases = eva_search._neuron_network._nn.intercepts_

            eva_output = eva_search.validate_all()
            log_statistics(save_path=validation_path,train_dataset= dataset_name, method='evolutionary',append=append,statistics=eva_output, dimension_reduction=dim_reduction)

def main(args: argparse.Namespace):

    # set random seeds
    np.random.seed(seed=args.seed)
    random.seed(args.seed)
    # Ignore Convergence warnings.
    warnings.filterwarnings('ignore', category=sklearn.exceptions.ConvergenceWarning)

    validation_output = os.path.join(args.output, 'validation_predictions.csv')

    if args.run_all:
        run_all(args=args, validation_path= validation_output)
        return
    
    grad_search = Backprop_Weight_Search(args)
    grad_search.run(args.iterations, seed = args.seed, parallel=True)
    grad_statistics = grad_search.validate_all()
    grad_search.plot_bestmodel_accuracy_progress(args.save_back)
    log_statistics(save_path=validation_output,train_dataset=args.dataset,method='Backpropagation', statistics=grad_statistics, append=False, dimension_reduction=args.dimension_reduction)

    eva_search = Evo_WeightSearch()
    eva_search.run(args, args.save_evo)
    eva_statistics = eva_search._neuron_network.validate_all()
    log_statistics(save_path=validation_output,train_dataset=args.dataset,method='Evolutionary', statistics=eva_statistics, append=True, dimension_reduction=args.dimension_reduction)





def parse_tuple_list(values: str) -> list[tuple[int]]:
    """Parser which converts user's input to list of integer tuples

    Args:
        values (str): user's string input

    Returns:
        list[tuple[int]]: list of tuples
    """
    tuples = values.split(',')
    try:
        parsed_tuples = []
        for t in tuples:
            stripped = re.sub(r'\s+', ' ', t.strip())  # Remove extra whitespaces
            tuple_elements = tuple(map(int, stripped.split()))  # Convert to integer tuple

            if any(x <= 0 for x in tuple_elements):
                raise ValueError(f"Tuple {tuple_elements} contains non-positive integer(s).")
    
            if tuple_elements: # dont append empty tuples
                parsed_tuples.append(tuple_elements)
            else:
                print(f"Warning empty tuple detected -> Not added to hidden layer")

        return parsed_tuples
    
    except ValueError as e:
        print(f"{e}")
        exit(1)
    except:
        print(f" Hidden layer sizes :{values} are in incorrect format. Use it in format")
        exit(1) # Exit program if the values are in incorect format

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    default_save_path = os.path.join(os.path.dirname(__file__), 'Output')
    evo_path = os.path.join(default_save_path, 'Evolutionary_search')
    backprop_path = os.path.join(default_save_path, 'Backpropagation_search')

    # General options
    parser.add_argument('--seed', default=42, type=int, help='Sets a seed to random number generation.')
    parser.add_argument('--hidden_layers', '--h', default="8 4 2, 16 8", type=parse_tuple_list, help='A list of tuples specifying the sizes of hidden layers. Input is string coma separates tuples. Values must be integers.'
        ' Example format: "1 2 3 4, 5 6 , 7" is parsed to [(1,2,3,4), (5,6), (7)]')
    # parser.add_argument('--method', '--m')
    parser.add_argument('--run_all', '--all',choices=['all', 'dim', 'dataset'] ,default=None, help=
                        'Choices:\nAll- runs all promap datasets.\n'
                        'Dim:- runs all possible dimension reductions only for one dataset(chosen by --dataset argument)\n'
                        'Dataset- runs and trains on all promap datasets with only one dimension reduction (chosen by --dims argument) ')
    # Evolution weight search arguments
    parser.add_argument('--save_evo', '--se', type=str, default=evo_path, help='Directory for saving every generated model')
    # parser.add_argument('--load_evo', '--le', type=str, default='Saves/example.model', help='Path to a model to be loaded.')
    parser.add_argument('--generations', '--gen', type=int, default=50, help='Nmber of generations in weight search evolution.')
    parser.add_argument('--metrics',type=str.lower,default='f1_macro',choices=EvolutionaryNeuronNetwork.Get_Metrics().keys(), help='Fitness function for searching weights via evolution algorithms')
    
    # Backpropagation weight search arguments
    parser.add_argument('--iterations', '--iter', default=50, type=int, help='Number of generations in backpropagation weight search.')
    parser.add_argument('--save_back', '--sb', type=str, default=backprop_path, help='Directory for saving every generated model')
   # parser.add_argument('--load_back', '--lb', type=str, default='Saves/example.model', help='Path to a model to be loaded.')

    # dataset preprocessing arguments
    parser.add_argument('--dimension_reduction', '--dims',default='raw', choices=['raw', 'lda', 'pca'],type=str.lower, help="Specify the dimension reduction technique: 'raw', 'lda', or 'pca'")
    parser.add_argument('--scale', '--s',default=False, action='store_true', help="Standardize data")
    
    # dataset arguments
    parser.add_argument('--dataset', '--d',default='promapcz', type=str.lower, choices=ProductsDatasets.NAME_MAP.keys(), help='name of promap dataset or path')

    # output arguments
    parser.add_argument('--output', '--o', type=str, default=default_save_path, help='Output directory name in which models validaitons are stored.')
    parser.add_argument('--kbest', '--k', default=10,type=int, help='prints k best networks')
    parser.add_argument('--analyze', '--a', default=False, action='store_true', help='Analyze the results of the evolutionary search and backpropagation search without running the search again')

    args = parser.parse_args()
    
    if not args.analyze:
        os.makedirs(name=args.output, exist_ok=True)
        os.makedirs(name=args.save_evo,exist_ok=True)
        os.makedirs(name=args.save_back,exist_ok=True)  
        main(args=args)
    
    analyze_path = args.output+'/validation_predictions.csv'
    if not os.path.exists(analyze_path):
        raise FileNotFoundError(f"File {analyze_path} does not exist. Please run the script with --analyze argument.")
    
    analyzator = Analyzer(analyze_path, args.save_evo, args.save_back)
    analyzator.analyze(args.output)