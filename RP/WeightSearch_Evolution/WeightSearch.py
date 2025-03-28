import pprint
import sklearn
import numpy as np
import argparse
import pickle
import os
import sys
from functools import partial
from deap import creator, algorithms, base, tools, cma
import sklearn.metrics
import sklearn.neural_network
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..'))) # To load Utils module
from Utils.ProMap import ProductsDatasets

class EvolutionaryNeuronNetwork:
    """
    Wrapper of neural network. Provides simple api changing weight of NN. 
    """
    def __init__(self, args: argparse.Namespace, hidden_layer_size: tuple[int]):
        self._nn = sklearn.neural_network.MLPClassifier(hidden_layer_sizes=hidden_layer_size, max_iter=1, verbose=False) # set max iter to 1, so biases and weight shapes are defined
        self._dataset = ProductsDatasets.Load_by_name(args.dataset)
        self._transformer = None
        self._scaler = None

        if args.scale:
            self._scaler = self._dataset.scale_features()

        if args.dimension_reduction:
            self._transformer = self._dataset.reduce_dimensions(method=args.dimension_reduction)

        self._nn.fit(self._dataset.train_set, self._dataset.train_targets)
        self.n_parameters = self._parameter_count()
        self._metrics = self._choose_metric(args.metrics)

    @staticmethod
    def Get_Metrics() -> dict[str, callable]:
        """Gets all possible callable metrics which can be used as a fitness function in evolution algorithms

        Returns:
            dict[str, callable]: Dictionary of metric's name and a callable function.
        """
        binary = partial(sklearn.metrics.f1_score, average='binary')
        weighted = partial(sklearn.metrics.f1_score, average='weighted')
        macro = partial(sklearn.metrics.f1_score, average='macro')
        micro = partial(sklearn.metrics.f1_score, average='micro')
        return {
            "f1_binary" : binary,
            "f1_weighted" : weighted,
            "f1_macro" : macro,
            "f1_micro" : micro,

            "accuracy" : sklearn.metrics.accuracy_score,
            "precision" : sklearn.metrics.precision_score,
            "recall" : sklearn.metrics.recall_score,  
        }
    
    def _choose_metric(self, metric: str) -> callable:
        """Chooses specific metric which will be used as a fitness metric in evolution algorithm.

        Args:
            metric (str): Name of the metric function

        Raises:
            ValueError: Raises value error if metric doesn't exist. All possible metrics can be obtained from EvolutionaryNeuronNetwork.Get_Metrtics().keys()

        Returns:
            callable: Chosen metric.
        """
        metrics = EvolutionaryNeuronNetwork.Get_Metrics()
        if metric in metrics:
            return metrics[metric]
        raise ValueError(f'Metric{metric} is not in {metrics}')

    def _parameter_count(self) -> int:
        """Counts parameters in neural network
        Returns:
            int: _description_
        """
        params = 0
        for w, b in zip(self._nn.coefs_, self._nn.intercepts_):
            params += w.size + b.size  
        
        return params

    def change_weights(self, weights: list):
        """Changes weights of NN

        Args:
            weights (list): Individual's weight

        Raises:
            AttributeError: _description_
        """
        if len(weights) != self.n_parameters:
            raise AttributeError(f"Individual's lenght:{len(weights)} is not the same as number of parameters{self.n_parameters}")
        start = 0
        new_layers = []
        new_baies = []
        for layer in self._nn.coefs_:
            new_layer = np.array(weights[start: start + layer.size])
            new_layers.append(new_layer.reshape(layer.shape))
            
            start += layer.size

        for bias in self._nn.intercepts_:
            new_bias = np.array(weights[start: start + bias.size])
            new_baies.append(new_bias)

            start += bias.size

        self._nn.coefs_ = new_layers
        self._nn.intercepts_ = new_baies

    def network_accuracy(self) -> float:
        """We want equal weights to all classes, regardless of the class distribution. Thus F1 macro average suits here the best. 

        Returns:
            float: _description_
        """
        pred = self._nn.predict(self._dataset.train_set)
        # f1 = sklearn.metrics.f1_score(y_true= self._dataset.train_targets, y_pred=pred, average='weighted')
        return self._metrics(y_true= self._dataset.train_targets, y_pred=pred)
    
    def validate(self, test_set = None, test_targets = None) -> dict[str, float]:
        """Validates NN againsts f1 score (binary, macro, micro and weighted average), precision, recall, accuracy and confusion matrix

        Returns:
            dict[str, float]: Dictionary of metric's name and metric's validation.
        """
        if test_set is None or test_targets is None:   
            pred = self._nn.predict(self._dataset.test_set)
            test_targets = self._dataset.test_targets
        else:
            pred = self._nn.predict(test_set)

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

            outputs.append((dataset.dataset_name, self.validate(dataset.test_set, dataset.test_targets)))
        return outputs

    def save_network(self, save_path: str):
        """Saves a NN.

        Args:
            save_path (str): Save path.
        """
        if not os.path.isdir(save_path):
            os.makedirs(save_path, exist_ok=True)

        with open(os.path.join(save_path,f'evolutionary_neuron_network.model'), mode='wb') as f:
            pickle.dump(self._nn, f)

    def load_model(self, model_path):
        """Loads a sklearn model from given path

        Args:
            model_path (_type_): Path to a model.
        """

        with open(model_path, 'rb') as model:
             self._nn = pickle.load(model)

class Evo_WeightSearch:
    """Main algorithm for searching weights in NN via evolution algorithms.
    """
    def __init__(self, args: argparse.Namespace):
        self._save_path = args.save
        self._neuron_network = EvolutionaryNeuronNetwork(args, (8,4,2))

        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        self._toolbox = base.Toolbox()
        self._toolbox.register("evaluate", lambda ind: Evo_WeightSearch._fitness(self._neuron_network, ind))
        self._toolbox.register("select", tools.selTournament, tournsize=3)
        
        self._toolbox.register("mutate", tools.mutGaussian, mu=0.0, sigma=0.1, indpb=0.2)

        strategy = cma.Strategy(centroid=[0.1] * self._neuron_network.n_parameters, sigma=0.5, lambda_= 8 * self._neuron_network.n_parameters)
        self._toolbox.register("generate", strategy.generate, creator.Individual)
        self._toolbox.register("update", strategy.update)

        self._hall_of_fame = tools.HallOfFame(1)

        self._stats = tools.Statistics(lambda ind: ind.fitness.values)
        self._stats.register("avg", np.mean)
        self._stats.register("std", np.std)
        self._stats.register("min", np.min)
        self._stats.register("max", np.max)

    @staticmethod
    def _fitness(network: EvolutionaryNeuronNetwork, individual: list) -> tuple[float]:
        """FItness function of evolutionary algorithm.

        Args:
            network (EvolutionaryNeuronNetwork): Neuron network with fixed architecture.
            individual (list): List of weights.

        Returns:
            tuple[float]: Accuracy score.
        """
        network.change_weights(individual)
        fitness = network.network_accuracy(),

        return fitness
    
    def run(self, generations: int):
        """Runs weight search neuroevolution generation.

        Args:
            generations (int): Number of generations in evolution algorithm.
        """
        (pop, stats) = algorithms.eaGenerateUpdate(self._toolbox, ngen=generations, stats=self._stats, halloffame=self._hall_of_fame)
        self._neuron_network.change_weights(weights=self._hall_of_fame[0])
        # pprint.pprint(self._neuron_network.validate_all())
        self._neuron_network.save_network(save_path=self._save_path) 
