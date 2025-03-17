import pandas as pd
import numpy as np
import sklearn.decomposition
import sklearn.discriminant_analysis
from sklearn.preprocessing import StandardScaler
import sklearn
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
        
    def scale_features(self):
        """
        Standardize features via sklearn.preprocessing.StandardScaler
        """
        scaler = StandardScaler()
        self.train_set = scaler.fit_transform(self.train_set)
        self.test_set = scaler.transform(self.test_set)
    
    def reduce_dimensions(self, method: str = 'lda'):
        """
        Reduce dimensions of test and train data
        LDA will compute maximum number of components based on the number of classes in the dataset.
        PCA will select the number of components such that the amount of variance that needs to be explained is greater than 95%

        Args:
            method (str, optional): Reduction method selection. Available dim. reduction is via LDA or PCA. Defaults to 'lda'.

        Raises:
            ValueError: If method is unknown dimension reduction methon
        """
        if method.lower() == 'lda':
            
            lda = sklearn.discriminant_analysis.LinearDiscriminantAnalysis()
            self.train_set = lda.fit_transform(self.train_set, self.train_targets)
            self.test_set = lda.transform(self.test_set)
        elif method.lower() == 'pca':

            pca = sklearn.decomposition.PCA(n_components=0.95)
            self.train_set = pca.fit_transform(self.train_set)
            self.test_set = pca.transform(self.test_set)
        else:
            raise ValueError('Unknows reduction method. Valid reduction method is: lda or pca')