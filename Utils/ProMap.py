import pandas as pd
import os
from Utils.Dataset import Dataset
from pathlib import Path

class ProductsDatasets:
    """_summary_
    Class for easily loading and manipulating ProMap datasets.
    """
    _default_basic_path = os.path.join(Path(os.path.join(__file__,"../../Data/Product-Mapping-Datasets/Basic ProMap Datasets" )).resolve())  # Default basic dataset root path
    _default_extended_path = os.path.join(Path(os.path.join(__file__,"../../Data/Product-Mapping-Datasets/Extended ProMap Datasets/ProMapsExtended/similarities" )).resolve())  # Default extended dataset root path
    _default_multi_path = os.path.join(Path(os.path.join(__file__,"../../Data/Product-Mapping-Datasets/Basic ProMap Datasets")).resolve()) 
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
    def __Split_data(path:str, dataset_name: str, extend_to : str|None) -> Dataset:
        """
        Helper function to load a dataset and split it into training and testing sets.

        Args:
            path (str): The base directory path where the dataset is stored.
            dataset_name (str): The name of the dataset to be loaded.
            extend_to (str | None): Optional. Path to an additional dataset (e.g., 'promap') used to fill missing columns in the main dataset (`dataset_name`). Missing columns will be filled with zeros based on the `extend_to` dataset. If not provided, no columns will be added.

        Returns: Dataset wrapper with proper train and test sets 
        """


        train_suffix = f"{dataset_name}-train_data_similarities.csv"
        test_suffix = f"{dataset_name}-test_data_similarities.csv"
        test_data = pd.read_csv(os.path.join(path, test_suffix))
        train_data = pd.read_csv(os.path.join(path, train_suffix))

        if extend_to:
            extended_train_data = pd.read_csv(f"{extend_to}-train_data_similarities.csv")

            missing_columns = set(extended_train_data.columns) - set(train_data.columns) # set label intersection. Find missins labels.
            # Extend missing features. Fill them by zero
            for col in missing_columns:
                test_data[col] = 0
                train_data[col] = 0

            # Reorder the features so the features for testing datasets align.
            train_data = train_data[extended_train_data.columns]
            test_data = test_data[extended_train_data.columns]
        return Dataset(train_data, test_data, dataset_name)
    
    # @staticmethod
    # def Load_basic_amazon_google(path: str = os.path.join(_default_basic_path,"amazon-google")) -> Dataset:
    #     """Loads amazon google dataset

    #     Args:
    #         path (str, optional): _description_. Path to the promap dataset

    #     Returns:
    #         Dataset: Dataset encapsulation.
    #     """
    #     return ProductsDatasets.__Split_data(path, "amazon_google", extend_to=)
    # @staticmethod
    # def __Split_data(path:str, dataset_name: str) -> Dataset:
    #     """_summary_
    #     Helper function to load and split the data into train and test data.
    #     Args:
    #         path (str): prefix path to the dataset
    #         dataset_name (str): dataset name

    #     Returns:
    #         Dataset: train and test set
    #     """
    #     train_suffix = f"{dataset_name}-train_data_similarities.csv"
    #     test_suffix = f"{dataset_name}-test_data_similarities.csv"
    #     test_data = pd.read_csv(os.path.join(path, test_suffix))
    #     train_data = pd.read_csv(os.path.join(path, train_suffix))
    #     return Dataset(train_data, test_data, dataset_name)
    
    @staticmethod
    def Load_basic_amazon_google(extend_to: str|None = os.path.join(_default_basic_path,"ProMapCz", 'promapcz'), path: str = os.path.join(_default_basic_path,"amazon-google")) -> Dataset:
        """Loads amazon google dataset

        Args:
            path (str, optional): _description_. Path to the promap dataset

        Returns:
            Dataset: Dataset encapsulation.
        """

        return ProductsDatasets.__Split_data(path, "amazon_google", extend_to=extend_to)
    
    @staticmethod
    def Load_basic_amazon_walmart(extend_to: str|None = os.path.join(_default_basic_path,"ProMapCz", 'promapcz'), path:str = os.path.join(_default_basic_path, "amazon-walmart") ) -> Dataset:
        """Loads amazon walmart dataset

        Args:
            path (str, optional): _description_. Path to the promap dataset

        Returns:
            Dataset: Dataset encapsulation.
        """
        return ProductsDatasets.__Split_data(path, "amazon_walmart", extend_to=extend_to)

    @staticmethod
    def Load_basic_promap_cz(extend_to: str|None = None, path :str =  os.path.join(_default_basic_path,"ProMapCz")  ) -> Dataset:
        """Loads promap cz dataset

        Args:
            path (str, optional): _description_. Path to the promap dataset

        Returns:
            Dataset: Dataset encapsulation.
        """
        return ProductsDatasets.__Split_data(path, "promapcz", extend_to=extend_to)

    @staticmethod
    def Load_basic_promap_en(extend_to: str|None = None, path:str = os.path.join(_default_basic_path, "ProMapEn") ) -> Dataset:
        """Loads promap en dataset

        Args:
            path (str, optional): _description_. Path to the promap dataset

        Returns:
            Dataset: Dataset encapsulation.
        """
        return ProductsDatasets.__Split_data(path, "promapen", extend_to=extend_to)

    @staticmethod
    def Load_extended_promap_cz( extend_to :str | None = None, path: str = _default_extended_path) -> Dataset:
        """Loads extended promap cz  dataset

        Args:
            path (str, optional): _description_. Path to the promap dataset

        Returns:
            Dataset: Dataset encapsulation.
        """
        return ProductsDatasets.__Split_data(path, "promapczext", extend_to= extend_to)

    @staticmethod
    def Load_extended_promap_en(extend_to: str|None = None, path: str = _default_extended_path) -> Dataset:
        """Loads extended promap en  dataset

        Args:
            path (str, optional): _description_. Path to the promap dataset

        Returns:
            Dataset: Dataset encapsulation.
        """
        return ProductsDatasets.__Split_data(path, "promapenext", extend_to=extend_to)
    
    @staticmethod
    def Load_extended_amazon_walmart(extend_to: str|None = os.path.join(_default_basic_path,"ProMapCz", 'promapcz'), path: str = _default_extended_path) -> Dataset:
        """Loads extended amazon walmart dataset

        Args:
            path (str, optional): _description_. Path to the promap dataset

        Returns:
            Dataset: Dataset encapsulation.
        """
        return ProductsDatasets.__Split_data(path, "promapmulti_amazon_ext", extend_to=extend_to)
    
    