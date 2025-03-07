import pandas as pd
import os

class ProMapDataset:
    """_summary_
    Class for easily loading and manipulating ProMap datasets.
    """
    __default_path = "../Data/Product-Mapping-Datasets/Basic ProMap Datasets"  # Default dataset root path
    
    @staticmethod
    def __Split_data(path:str, dataset_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        """_summary_
        Helper function to load and split the data into train and test data.
        Args:
            path (str): prefix path to the dataset
            dataset_name (str): dataset name

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: train and test set
        """
        train_suffix = f"{dataset_name}-train_data_similarities.csv"
        test_suffix = f"{dataset_name}-test_data_similarities.csv"
        test_data = pd.read_csv(os.path.join(path, test_suffix))
        train_data = pd.read_csv(os.path.join(path, train_suffix))
        return train_data.iloc, test_data
    
    @staticmethod
    def Load_amazon_google(path: str = os.path.join(__default_path,"amazon-google")) -> tuple[pd.DataFrame, pd.DataFrame]:
        return ProMapDataset.__Split_data(path, "amazon_google")
    
    @staticmethod
    def Load_amazon_walmart(path:str = os.path.join(__default_path, "amazon-walmart") ) -> tuple[pd.DataFrame, pd.DataFrame]:
        return ProMapDataset.__Split_data(path, "amazon_walmart")

    @staticmethod
    def Load_promap_cz(path :str =  os.path.join(__default_path,"ProMapCz")  ) -> tuple[pd.DataFrame, pd.DataFrame]:
        return ProMapDataset.__Split_data(path, "promapcz")

    @staticmethod
    def Load_promap_en(path:str = os.path.join(__default_path, "ProMapEn") ) -> tuple[pd.DataFrame, pd.DataFrame]:
        return ProMapDataset.__Split_data(path, "promapen")
