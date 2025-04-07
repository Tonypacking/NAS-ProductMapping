import pandas as pd
import os
from Utils.Dataset import Dataset
from pathlib import Path

_default_basic_path = os.path.join(Path(os.path.join(__file__,"../../Data/Product-Mapping-Datasets/Basic ProMap Datasets" )).resolve())  # Default basic dataset root path
_default_extended_path = os.path.join(Path(os.path.join(__file__,"../../Data/Product-Mapping-Datasets/Extended ProMap Datasets/ProMapsExtended/similarities" )).resolve())  # Default extended dataset root path
_default_multi_path = os.path.join(Path(os.path.join(__file__,"../../Data/Product-Mapping-Datasets/Basic ProMap Datasets")).resolve()) 

class ProductsDatasets:
    """_summary_
    Class for easily loading and manipulating ProMap datasets.
    """
    NAME_MAP = {
        'google' : 'amazon-google',
        'walmart' : "amazon-walmart",
        'promapcz' : 'promapcz',
        'promapen' : 'promapen',
        'promapczext' : 'promapczext',
        'promapenext' : 'promapenext',
        'amazonext' : 'promapmulti_amazon_ext',
    }
    
    @staticmethod
    def Load_by_name(name : str, match_columns :Dataset|None = None, remove_columns : list[str]|None = None) -> Dataset:
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
                return ProductsDatasets.Load_basic_amazon_google(remove_columns = remove_columns, match_columns=match_columns)
            case 'walmart' :
                return ProductsDatasets.Load_basic_amazon_walmart(remove_columns = remove_columns, match_columns=match_columns)
            case 'promapcz' :
                return ProductsDatasets.Load_basic_promap_cz(remove_columns = remove_columns, match_columns=match_columns)
            case 'promapen' :
                return ProductsDatasets.Load_basic_promap_en(remove_columns = remove_columns, match_columns=match_columns)
            case 'promapczext' :
                return ProductsDatasets.Load_extended_promap_cz(remove_columns = remove_columns, match_columns=match_columns)
            case 'promapenext' :
                return ProductsDatasets.Load_extended_promap_en(remove_columns = remove_columns, match_columns=match_columns)
            case 'amazonext':
                return ProductsDatasets.Load_extended_amazon_walmart(remove_columns = remove_columns, match_columns=match_columns)
            case _:
                raise ValueError(f'Unknown dataset: {name}')
    
    @staticmethod
    def __Split_data(path:str, dataset_name: str, remove_columns : list[str] | None, match_columns_dataset : Dataset|None ) -> Dataset:
        """_summary_

        Args:
            path (str): path to a dataset
            dataset_name (str): name of the dataset
            remove_columns (list[str] | None): Column names to be removed from the dataset
            match_columns_dataset (Dataset| None): Columns to be present in train and test dataset. If the created dataset file doesn't include any column from match_columns dataset. new column of zeros is created. Order of columns are then order based on match_columns dataset

        Returns:
            Dataset: new dataset with splitted train and test data.
        """
        train_suffix = f"{dataset_name}-train_data_similarities.csv"
        test_suffix = f"{dataset_name}-test_data_similarities.csv"
        test_data = pd.read_csv(os.path.join(path, test_suffix))
        train_data = pd.read_csv(os.path.join(path, train_suffix))

        if match_columns_dataset:
            all_columns = match_columns_dataset.ids + list(match_columns_dataset.feature_labels) + [match_columns_dataset.target_labels]

            missing_features =  set(all_columns) -  set(train_data.columns) 

            if missing_features is not None : # add missing features to train and test data
                for col in missing_features:
                    train_data[col] = 0
                    test_data[col] = 0

                # reorder columns
                test_data = test_data[all_columns]
                train_data = train_data[all_columns]

        if remove_columns:
            test_data = test_data.drop(columns=remove_columns, errors='ignore')
            train_data = train_data.drop(columns=remove_columns, errors='ignore')

        return Dataset(train_data, test_data, dataset_name)
     
    @staticmethod
    def Load_basic_amazon_google(match_columns :Dataset | None = None, remove_columns : list[str] | None = None, path: str = os.path.join(_default_basic_path,"amazon-google")) -> Dataset:
        """Loads amazon google dataset

        Args:
            path (str, optional): _description_. Path to the promap dataset

        Returns:
            Dataset: Dataset encapsulation.
        """

        return ProductsDatasets.__Split_data(path, "amazon_google", remove_columns=remove_columns, match_columns_dataset=match_columns)
    
    @staticmethod
    def Load_basic_amazon_walmart(match_columns :Dataset | None = None, remove_columns : list[str] | None = None, path:str = os.path.join(_default_basic_path, "amazon-walmart") ) -> Dataset:
        """Loads amazon walmart dataset

        Args:
            path (str, optional): _description_. Path to the promap dataset

        Returns:
            Dataset: Dataset encapsulation.
        """

        return ProductsDatasets.__Split_data(path, "amazon_walmart", remove_columns=remove_columns, match_columns_dataset=match_columns)

    @staticmethod
    def Load_basic_promap_cz(match_columns :Dataset | None = None, remove_columns : list[str] | None = None, path :str =  os.path.join(_default_basic_path,"ProMapCz")  ) -> Dataset:
        """Loads promap cz dataset

        Args:
            path (str, optional): _description_. Path to the promap dataset

        Returns:
            Dataset: Dataset encapsulation.
        """

        return ProductsDatasets.__Split_data(path, "promapcz", remove_columns=remove_columns, match_columns_dataset=match_columns)

    @staticmethod
    def Load_basic_promap_en(match_columns :Dataset | None = None, remove_columns : list[str] | None = None,path:str = os.path.join(_default_basic_path, "ProMapEn") ) -> Dataset:
        """Loads promap en dataset

        Args:
            path (str, optional): _description_. Path to the promap dataset

        Returns:
            Dataset: Dataset encapsulation.
        """

        return ProductsDatasets.__Split_data(path, "promapen", remove_columns=remove_columns, match_columns_dataset=match_columns)

    @staticmethod
    def Load_extended_promap_cz(match_columns :Dataset | None = None, remove_columns : list[str] | None = None, path: str = _default_extended_path) -> Dataset:
        """Loads extended promap cz  dataset

        Args:
            path (str, optional): _description_. Path to the promap dataset

        Returns:
            Dataset: Dataset encapsulation.
        """

        return ProductsDatasets.__Split_data(path, "promapczext", remove_columns=remove_columns,match_columns_dataset=match_columns)

    @staticmethod
    def Load_extended_promap_en(match_columns :Dataset | None = None, remove_columns : list[str] | None = None, path: str = _default_extended_path) -> Dataset:
        """Loads extended promap en  dataset

        Args:
            path (str, optional): _description_. Path to the promap dataset

        Returns:
            Dataset: Dataset encapsulation.
        """

        return ProductsDatasets.__Split_data(path, "promapenext", remove_columns=remove_columns, match_columns_dataset=match_columns)
    
    @staticmethod
    def Load_extended_amazon_walmart(match_columns :Dataset | None = None, remove_columns : list[str] | None = None, path: str = _default_extended_path) -> Dataset:
        """Loads extended amazon walmart dataset

        Args:
            path (str, optional): _description_. Path to the promap dataset

        Returns:
            Dataset: Dataset encapsulation.
        """

        return ProductsDatasets.__Split_data(path, "promapmulti_amazon_ext", remove_columns=remove_columns, match_columns_dataset=match_columns)
    