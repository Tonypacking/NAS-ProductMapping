from WeightSearch_Evolution import WeightSearch, EvolutionaryNeuronNetwork

from Utils.ProMap import ProductsDatasets
import argparse
import numpy as np
import random
def main(args: argparse.Namespace):

    np.random.seed(seed=args.seed)
    random.seed(args.seed)
    eva_search = WeightSearch(args)
    eva_search.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # General options
    parser.add_argument('--seed', default=42, type=int, help='Sets a seed to random number generation.')
    
    # Evolution weight search arguments
    parser.add_argument('--save', type=str, default='Saves/', help='Directory for saving every generated model')
    parser.add_argument('--load', type=str, default='Saves/example.model', help='Path to a model to be loaded.')
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

