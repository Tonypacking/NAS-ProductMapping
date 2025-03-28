import sklearn
import argparse
import sys
import os
import sklearn.metrics
import pickle
import sklearn.model_selection
import sklearn.neural_network
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..'))) # To load Utils module
from Utils.ProMap import ProductsDatasets

class Backprop_Weight_Search:
    def __init__(self, args: argparse.Namespace ):
        self._param_grid = {
            'hidden_layer_sizes' : args.hidden_layers,
            'activation' : ['relu', 'tanh', 'logistic'],
            'alpha' : [0.000_1, 0.001]
        }
        self._dataset = ProductsDatasets.Load_by_name(args.dataset)
        self.best_model = None
        self._scaler = None
        self._transformer = None

        self.metrics = args.metrics # metric to optimize in 

        if args.scale:
            self._scaler = self._dataset.scale_features()

        if args.dimension_reduction:
            self._transformer = self._dataset.reduce_dimensions(method=args.dimension_reduction)

    
    def run(self, iterations: int, seed: int = 42, parallel = None):
        if parallel:
            parallel = -1

        scoring_metrics = [ 'precision_macro', 'precision', 'recall_macro', 'recall', 'f1_macro', 'f1_micro', 'accuracy',]

        grid_search = sklearn.model_selection.GridSearchCV(estimator=sklearn.neural_network.MLPClassifier(
            random_state=seed, verbose=True, max_iter=iterations), param_grid=self._param_grid, cv=5, n_jobs=parallel, refit='f1_macro', scoring=scoring_metrics
            )
        
        grid_search.fit(self._dataset.train_set, self._dataset.train_targets)

        self.best_model = grid_search.best_estimator_

        
    def save_network(self, save_path: str):
        """Saves a NN.
        Args:
            save_path (str): Save path.
        """
        if not self.best_model:
            raise ValueError('Model not trained')

        if not os.path.isdir(save_path):
            os.makedirs(save_path, exist_ok=True)

        with open(os.path.join(save_path,f'evolutionary_neuron_network.model'), mode='wb') as f:
            pickle.dump(self.best_model, f)


    
    def load_model(self, model_path):
        """Loads a sklearn model from given path

        Args:
            model_path (_type_): Path to a model.
        """

        with open(model_path, 'rb') as model:
             self.best_model = pickle.load(model)

    def validate(self, test_set = None, test_targets = None) -> dict[str, float]:
        """Validates NN againsts f1 score (binary, macro, micro and weighted average), precision, recall, accuracy and confusion matrix

        Returns:
            dict[str, float]: Dictionary of metric's name and metric's validation.
        """
        if test_set is None or test_targets is None:   
            pred = self.best_model.predict(self._dataset.test_set)
            test_targets = self._dataset.test_targets
        else:
            pred = self.best_model.predict(test_set)

        return { 
            'f1_score_binary' : sklearn.metrics.f1_score(y_true=test_targets, y_pred=pred, average="binary"),
            'f1_score_macro' : sklearn.metrics.f1_score(y_true=test_targets, y_pred=pred, average="macro"),
            'f1_score_micro' : sklearn.metrics.f1_score(y_true=test_targets, y_pred=pred, average="micro"),
            'f1_score_weighted' : sklearn.metrics.f1_score(y_true=test_targets, y_pred=pred, average="weighted"),
            "precision" : sklearn.metrics.precision_score(y_true=test_targets, y_pred=pred),
            "recall" : sklearn.metrics.recall_score(y_true=test_targets, y_pred=pred),
            'accuracy' : sklearn.metrics.accuracy_score(y_true=test_targets, y_pred=pred),
            'confusion_matrix' : sklearn.metrics.confusion_matrix(y_true=test_targets, y_pred=pred),
        }
    
    def validate_all(self) -> dict[str, float]:
        """Validates against all promap datasets if feature count is the same.

        Returns:
            list[tuple[str, dict[str, float]]]: List of tuples with name of tested dataset and dictionary of metric and metrics value.
        """

        outputs = []
        for name in ProductsDatasets.NAME_MAP:
            dataset= ProductsDatasets.Load_by_name(name)

            if dataset.feature_labels.shape != self._dataset.feature_labels.shape:   
                print(f'Datasets features are different, cannot transform them\nTested dataset name: {dataset.dataset_name} of shape {dataset.feature_labels.shape}\nTrained on {self._dataset.dataset_name} of shape {self._dataset.feature_labels.shape}')
                continue

            if self._scaler:
                dataset.test_set = self._scaler.transform(dataset.test_set)
                
            if self._transformer:
                dataset.test_set = self._transformer.transform(dataset.test_set)

            outputs.append((f"test_{dataset.dataset_name}", self.validate(dataset.test_set, dataset.test_targets)))
        return outputs