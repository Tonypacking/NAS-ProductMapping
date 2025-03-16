import pandas as pd
from ProMap import ProductsDatasets
import argparse
import NeuroEvolution
parser = argparse.ArgumentParser()

def main(args: argparse.Namespace):

    

    a = ProductsDatasets.Load_extended_promap_cz()

    evolution = NeuroEvolution.Evolution("Config/ProMapCz", a)
    evolution.run(parralel=True)

    print(f"validation of: {evolution.validation()}")

if __name__ == "__main__":
    main(parser.parse_args())

