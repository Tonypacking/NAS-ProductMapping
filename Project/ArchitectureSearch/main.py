import pandas as pd
from ProMap import ProductsDatasets
import argparse
import NeuroEvolution
parser = argparse.ArgumentParser()

parser.add_argument('--verbose', action='store_true', help='Enable verbouse output')

def main(args: argparse.Namespace):

    a = ProductsDatasets.Load_basic_promap_cz()
    
    input()
    evolution = NeuroEvolution.Evolution("Config/ProMapCz", a)
    evolution.run()

    print(f"validation of: {evolution.validation()}")


if __name__ == "__main__":
    main(parser.parse_args())

