from WeightSearch_Evolution import WeightSearch, EvolutionaryNeuronNetwork
import re
from Utils.ProMap import ProductsDatasets
import argparse
import numpy as np
import random
def main(args: argparse.Namespace):

    np.random.seed(seed=args.seed)
    random.seed(args.seed)
    eva_search = WeightSearch(args)
    eva_search.run(args.generations)

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

    # General options
    parser.add_argument('--seed', default=42, type=int, help='Sets a seed to random number generation.')
    parser.add_argument('--hidden_layers', '--h', default="8 4 2, 16 8", type=parse_tuple_list, help='A list of tuples specifying the sizes of hidden layers. Input is string coma separates tuples. Values must be integers.'
    ' Example format: "1 2 3 4, 5 6 , 7" is parsed to [(1,2,3,4), (5,6), (7)]')


    # Evolution weight search arguments
    parser.add_argument('--save', type=str, default='Saves/', help='Directory for saving every generated model')
    parser.add_argument('--load', type=str, default='Saves/example.model', help='Path to a model to be loaded.')
    parser.add_argument('--generations', '--gen', type=int, default=50, help='Nmber of generations in weight search evolution.')
    parser.add_argument('--metrics',type=str.lower,default='f1_macro',choices=EvolutionaryNeuronNetwork.Get_Metrics().keys(), help='Fitness function for searching weights via evolution algorithms')
    

    # dataset preprocessing arguments
    parser.add_argument('--dimension_reduction', '--dims',default='raw', choices=['raw', 'lda', 'pca'],type=str.lower, help="Specify the dimension reduction technique: 'raw', 'lda', or 'pca'")
    parser.add_argument('--scale', '--s',default=False, action='store_true', help="Standardize data")
    
    # dataset arguments
    parser.add_argument('--dataset', '--d',default='promapcz', type=str.lower, choices=ProductsDatasets.NAME_MAP.keys(), help='name of promap dataset or path')

    # output arguments
    parser.add_argument('--output', '--o', type=str.lower, default='output', help='Output directory name.')
    parser.add_argument('--validate_all', '--v', action='store_false', default=True, help='Validates input against all possible datasets. If feature count is not same, it is ignored')
    parser.add_argument('--kbest', '--k', default=10,type=int, help='prints k best networks')
    
    args = parser.parse_args()
    main(args=args)

