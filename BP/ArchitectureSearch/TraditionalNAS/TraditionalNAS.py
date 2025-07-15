import sklearn
import numpy as np
import sys, os

import sklearn.model_selection
import sklearn.preprocessing
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..','..'))) # To load Utils module
import Utils.ProMap as ProMap
from Utils.ProMap import Dataset
from Utils.ProMap import ProductsDatasets, Dataset
import pickle

class Gridsearch_NAS:
    """
    Traditional NAS strategy
    """
    def __init__(self, args):
        self._dataset = ProductsDatasets.Load_by_name(args.dataset)
        self.best_mlp_model = None
        self.best_params = None
        self._scaler = None
        self._transformer = None

        if args.scale:
            self._scaler = self._dataset.scale_features()

        if args.dimension_reduction:
            self._transformer = self._dataset.reduce_dimensions(args.dimension_reduction)
            

    def runNAS(self, save_model_dir):
        """
        Runs traditional NAS

        Args:
            save_model_dir (str): Save directory
        """
        mlp = sklearn.neural_network.MLPClassifier(max_iter=200, random_state=42)
        param_grid = {
            'hidden_layer_sizes': [
                (50,),          
                (100,),             
                (100, 50, 25),
                (1000, 256,128,64,32,16),
                (1000,2)    
            ],
            'activation': [
                'relu', 
                'tanh', 
                'logistic'
                ],
                "alpha" : [1,0.01,0.0001],
            "solver": ["adam", "sgd"]
        }
        grid_search = sklearn.model_selection.GridSearchCV(mlp, param_grid, cv=5, n_jobs=-1, verbose=2, refit=True)
        _encoder =  sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore', sparse_output=False)

        features = _encoder.fit_transform(self._dataset.train_targets.reshape(-1 , 1))
        grid_search.fit(self._dataset.train_set, features)

        self.best_mlp_model = grid_search.best_estimator_   
        self.best_params = grid_search.best_params_
        model_path = os.path.join(save_model_dir, "TraditionalModel.model")
        try:
            with open(model_path, 'wb') as file:
                pickle.dump( self.best_mlp_model , file)

        except Exception as e:            
            print("MODEL NOT SAVED")
            return None
        
    def validate(self, test_set = None , target_set = None) -> dict[str, float]:
        """Validates best network against unseen data.

        Args:
            test_set (Optional[Sequence], optional): testing set. Defaults to None.
            target_set (Optional[Sequence], optional): testing true outpiut. Defaults to None.

        Returns:
            dict[str, float]: dictionary of name of a metric and metric's value
        """
        if not self.best_mlp_model:
            return None
        
        if test_set is None and target_set is None:
            predicted = self.best_mlp_model.predict(self._dataset.test_set)
            target_set = self._dataset.test_targets
            

        elif (test_set is None and target_set is not None) or (test_set is not None and target_set is None):
            assert ValueError("Invalid test set or target set")
        else:
            predicted = self.best_mlp_model.predict(test_set)
        predicted = np.argmax(predicted, axis=1)  
        return {
                'f1_score' : sklearn.metrics.f1_score(y_pred=predicted, y_true=target_set),
                'accuracy' : sklearn.metrics.accuracy_score(y_pred=predicted, y_true=target_set),
                'precision' : sklearn.metrics.precision_score(y_pred=predicted, y_true=target_set),
                'recall' : sklearn.metrics.recall_score(y_pred=predicted, y_true=target_set),
                'confusion_matrix' : sklearn.metrics.confusion_matrix(y_pred=predicted, y_true=target_set),
                'balanced_accuracy': sklearn.metrics.balanced_accuracy_score(y_pred=predicted, y_true=target_set),
            } 
        
        
    def validate_all(self) -> list[tuple[str, dict[str, float]]]:
            """Validates against all promap datasets if feature count is the same.
            Resises the testing dataset to match the training dataset's feature columns.

            Returns:
                list[tuple[str, dict[str, float]]]: List of tuples with name of tested dataset and dictionary of metric and metrics value.
            """
            outputs = []
            for name in ProMap.ProductsDatasets.NAME_MAP:
                tested_dataset= ProMap.ProductsDatasets.Load_by_name(name)

                if tested_dataset.feature_labels.shape < self._dataset.feature_labels.shape:
                    tested_dataset.extend_dataset(self._dataset)

                elif tested_dataset.feature_labels.shape > self._dataset.feature_labels.shape:
                    tested_dataset.reduce_dataset(self._dataset)

                if self._scaler:
                    tested_dataset.test_set = self._scaler.transform(tested_dataset.test_set)
                    
                if self._transformer:
                    tested_dataset.test_set = self._transformer.transform(tested_dataset.test_set)
                
                outputs.append((tested_dataset.dataset_name, self.validate(tested_dataset.test_set, tested_dataset.test_targets)))
            return outputs
    