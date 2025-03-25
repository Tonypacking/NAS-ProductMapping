import sklearn
import numpy as np
import argparse
import random
import pickle

from deap import creator, algorithms, base, tools, cma
from ProMap import ProductsDatasets
import os

class EvolutionaryNeuronNetwork:

    def __init__(self, args: argparse.Namespace, hidden_layer_size: tuple[int]):
        self._nn = sklearn.neural_network.MLPClassifier(hidden_layer_sizes=hidden_layer_size, max_iter=1) # set max iter to 1, so biases and weight shapes are defined
        self._dataset = ProductsDatasets.Load_by_name(args.dataset)
        self._nn.fit(self._dataset.train_set, self._dataset.train_targets)
        self.n_parameters = self._parameter_count()

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
        pred = self._nn.predict(self._dataset.train_set)
        return sklearn.metrics.f1_score(y_true= self._dataset.train_targets, y_pred=pred) * 1000
    
    def test(self) -> float:
        pred = self._nn.predict(self._dataset.test_set)
        return sklearn.metrics.f1_score(y_true=self._dataset.test_targets, y_pred=pred) 

    def save_network(self, save_path: str):
        if not os.path.isdir(save_path):
            os.makedirs(save_path, exist_ok=True)

        with open(os.path.join(save_path,f'evolutionary_neuron_network.model'), mode='wb') as f:
            pickle.dump(self._nn, f)

    def load_model(self, model_path):
         with open(model_path, 'rb') as model:
             self._nn = pickle.load(model)




class WeightSearch:

    def __init__(self, args: argparse.Namespace):
        np.random.seed(seed=args.seed)
        random.seed(args.seed)
        self.nn = EvolutionaryNeuronNetwork(args, (8,4,2))

        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        self._toolbox = base.Toolbox()
        self._toolbox.register("evaluate", lambda ind: WeightSearch.fitness(nn, ind))
        self._toolbox.register("mutate", tools.mutGaussian, mu=0.0, sigma=0.1, indpb=0.2)
        self._toolbox.register("mutate_polynomial", tools.mutPolynomialBounded, eta=20.0, indpb=0.2)
        self._toolbox.register("mutate_lognormal", tools.mutESLogNormal, mu=0.0, sigma=0.1, indpb=0.2)
        strategy = cma.Strategy(centroid=[0.0] * nn.n_parameters, sigma=0.5, lambda_= 5 * nn.n_parameters)
        self._toolbox.register("generate", strategy.generate, creator.Individual)
        self._toolbox.register("update", strategy.update)

        self._hall_of_fame = tools.HallOfFame(1)

        self._stats = tools.Statistics(lambda ind: ind.fitness.values)
        self._stats.register("avg", np.mean)
        self._stats.register("std", np.std)
        self._stats.register("min", np.min)
        self._stats.register("max", np.max)

    @staticmethod
    def fitness(network: EvolutionaryNeuronNetwork, individual: list) -> tuple[float]:
        network.change_weights(individual)
        fitness = network.network_accuracy(),
        return fitness
    
    def run(self):
       
        _ = algorithms.eaGenerateUpdate(self._toolbox, ngen=25, stats=self._stats, halloffame=self._hall_of_fame)

        nn.change_weights(weights=self._hall_of_fame[0])
        print(nn.test())

        nn.save_network(save_path=args.save) 







