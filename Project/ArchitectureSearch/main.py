import pandas as pd
from ProMap import ProductsDatasets
import argparse
import NeuroEvolution
import NeatConfigParser

def main(args: argparse.Namespace):

    a = ProductsDatasets.Load_extended_promap_cz()
    evolution = NeuroEvolution.Evolution("Config/ProMapCz", a)
    
    evolution.run(args.iterations, args.parallel)

    print(f"validation of: {evolution.validation()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--parallel', '--par',  action='store_true', default=False, help="Runs Neat in parallel.",)
    parser.add_argument('--iterations', '--iter', default=50, type=int, help='Number of generations in neat')
    main(parser.parse_args())

