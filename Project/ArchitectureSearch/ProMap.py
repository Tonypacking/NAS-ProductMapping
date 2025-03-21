import pandas as pd
import os
from Dataset import Dataset
class ProductsDatasets:
    """_summary_
    Class for easily loading and manipulating ProMap datasets.
    """
    _default_basic_path = "../Data/Product-Mapping-Datasets/Basic ProMap Datasets"  # Default basic dataset root path
    _default_extended_path = "../Data/Product-Mapping-Datasets/Extended ProMap Datasets/ProMapsExtended/similarities" # Default extended dataset root path
    _default_multi_path = "../Data/Product-Mapping-Datasets/Extended ProMap Datasets/ProMapsMulti/similarities"
    NAME_MAP = {
        'google' : 'amazon-google',
        'walmart' : "amazon-walmart",
        'promapcz' : 'promapcz',
        'promapen' : 'promapen',
        'promapczext' : 'promapczext',
        'promapenext' : 'promapenext',
        'amazonext' : 'promapmulti_amazon_ext'
    }

    @staticmethod
    def Load_by_name(name : str) -> Dataset:
        """Loads Product datasets based on shorter name.

        Args:
            name (str): args name

        Raises:
            ValueError: unknown dataset shorter name.

        Returns:
            Dataset: Product dataset
        """
        match name:
            case 'google':
                return ProductsDatasets.Load_basic_amazon_google()
            case 'walmart' :
                return ProductsDatasets.Load_basic_amazon_walmart()
            case 'promapcz' :
                return ProductsDatasets.Load_basic_promap_cz()
            case 'promapen' :
                return ProductsDatasets.Load_basic_promap_en()
            case 'promapczext' :
                return ProductsDatasets.Load_extended_promap_cz()
            case 'promapenext' :
                return ProductsDatasets.Load_extended_promap_en()
            case 'amazonext':
                return ProductsDatasets.Load_extended_amazon_walmart()
            case _:
                raise ValueError(f'Unknown dataset: {name}')
    
    @staticmethod
    def __Split_data(path:str, dataset_name: str) -> Dataset:
        """_summary_
        Helper function to load and split the data into train and test data.
        Args:
            path (str): prefix path to the dataset
            dataset_name (str): dataset name

        Returns:
            Dataset: train and test set
        """
        train_suffix = f"{dataset_name}-train_data_similarities.csv"
        test_suffix = f"{dataset_name}-test_data_similarities.csv"
        test_data = pd.read_csv(os.path.join(path, test_suffix))
        train_data = pd.read_csv(os.path.join(path, train_suffix))
        return Dataset(train_data, test_data, dataset_name)
    
    @staticmethod
    def Load_basic_amazon_google(path: str = os.path.join(_default_basic_path,"amazon-google")) -> Dataset:
        return ProductsDatasets.__Split_data(path, "amazon_google")
    @staticmethod
    def __Split_data(path:str, dataset_name: str) -> Dataset:
        """_summary_
        Helper function to load and split the data into train and test data.
        Args:
            path (str): prefix path to the dataset
            dataset_name (str): dataset name

        Returns:
            Dataset: train and test set
        """
        train_suffix = f"{dataset_name}-train_data_similarities.csv"
        test_suffix = f"{dataset_name}-test_data_similarities.csv"
        test_data = pd.read_csv(os.path.join(path, test_suffix))
        train_data = pd.read_csv(os.path.join(path, train_suffix))
        return Dataset(train_data, test_data, dataset_name)
    
    @staticmethod
    def Load_basic_amazon_google(path: str = os.path.join(_default_basic_path,"amazon-google")) -> Dataset:
        return ProductsDatasets.__Split_data(path, "amazon_google")
    
    @staticmethod
    def Load_basic_amazon_walmart(path:str = os.path.join(_default_basic_path, "amazon-walmart") ) -> Dataset:
        return ProductsDatasets.__Split_data(path, "amazon_walmart")

    @staticmethod
    def Load_basic_promap_cz(path :str =  os.path.join(_default_basic_path,"ProMapCz")  ) -> Dataset:
        return ProductsDatasets.__Split_data(path, "promapcz")

    @staticmethod
    def Load_basic_promap_en(path:str = os.path.join(_default_basic_path, "ProMapEn") ) -> Dataset:
        return ProductsDatasets.__Split_data(path, "promapen")

    @staticmethod
    def Load_extended_promap_cz(path: str = _default_extended_path) -> Dataset:
        return ProductsDatasets.__Split_data(path, "promapczext")

    @staticmethod
    def Load_extended_promap_en(path: str = _default_extended_path) -> Dataset:
        return ProductsDatasets.__Split_data(path, "promapenext")
    
    @staticmethod
    def Load_extended_amazon_walmart(path: str = _default_extended_path) -> Dataset:
        return ProductsDatasets.__Split_data(path, "promapmulti_amazon_ext")
    
    