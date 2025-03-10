import pandas as pd
import numpy as np

class Dataset:
    """_summary_
    Dataset wrapper. 
    Splits training and testing data to features and targets
    """
    def __init__(self, training_data: pd.DataFrame, testing_data: pd.DataFrame, name: None| str  = None):
        self.ids = ['id1', 'id2']
        self.dataset_name = name
        self.feature_labels = training_data.drop(self.ids, axis=1).iloc[:, :-1].columns
        self.target_labels = training_data.columns[-1]

        # Training data
        self.train_set = training_data.drop(self.ids, axis=1).iloc[:, :-1].values # drop id1 and id2, becase we don't need product ids (URLs) for classifier
        self.train_targets = training_data.iloc[:, -1].values
        self.training_product_ids = training_data[self.ids].values

        # Testing data
        self.test_set = testing_data.drop(self.ids, axis=1).iloc[:, :-1].values # drop id1 and id2, becase we don't need product ids (URLs) for classifier
        self.test_targets = testing_data.iloc[:, -1].values
        self.testing_product_ids = testing_data[self.ids].values
        
       