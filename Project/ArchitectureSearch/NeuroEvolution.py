import neat.config
import neat.genes
import numpy as np
import sklearn
import neat
import sklearn.metrics
from typing import Sequence, Optional
import multiprocessing
import sklearn.metrics._base
from Dataset import Dataset
import configparser
import visualize


class Evolution:
    
    def __init__(self, config_path: str, dataset:Dataset = None, scaling : bool= False, dimension_reduction : str = 'raw'):
        self._dataset :Dataset = dataset
        self.dataset_name = self._dataset.dataset_name
        
        self._neat_config = self._create_config(config_path, scaling=scaling, dimension_reduction=dimension_reduction)
        self._population = neat.Population(self._neat_config)
        self.Best_network = None
        self._fitness_scaling = 1_000 

    @staticmethod
    def _binarize_prediction(x: float) -> int:
        """
        Binarize prediction to matching or None Matching
        """
        return 1 if x >= 0.5 else 0
    
    def _eval_genomes(self, genomes, config):
        for id, genome in genomes:
            genome.fitness = self._dataset.train_set.shape[0]

            net = neat.nn.FeedForwardNetwork.create(genome=genome, config=config)

            predicted = np.array([Evolution._binarize_prediction(net.activate(x)[0]) for x in self._dataset.train_set])
            genome.fitness = sklearn.metrics.f1_score(y_pred=predicted, y_true=self._dataset.train_targets) * self._fitness_scaling

    def _create_config(self, config_path, scaling : bool = False, dimension_reduction: str = 'raw') -> neat.Config:
        if scaling:
            self._dataset.scale_features()
        
        if dimension_reduction == 'lda' or dimension_reduction == 'pca':
            # number of input nodes are reduce. Dynamically change neat config also.
            self._dataset.reduce_dimensions(dimension_reduction)
            num_features = self._dataset.train_set.shape[1]
            
            parser = configparser.ConfigParser()
            parser.read(config_path)
            parser.set('DefaultGenome', 'num_inputs', str(num_features))

            with open(config_path, 'w') as f:
                parser.write(f)

        return neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

    def run(self, iterations: int = 50, parralel: bool = False) -> neat.nn.FeedForwardNetwork:

        population = neat.Population(self._neat_config)

        population.add_reporter(neat.StdOutReporter(show_species_detail=True))
        self._statistics = neat.StatisticsReporter()
        population.add_reporter(self._statistics)
        if parralel:
            para_eval = neat.ParallelEvaluator(num_workers=multiprocessing.cpu_count(),eval_function=self._eval_genomes)

            self._winner = population.run(para_eval.eval_function, iterations)
        else:
            self._winner = population.run(self._eval_genomes, iterations)

        self.Best_network = neat.nn.FeedForwardNetwork.create(self._winner, self._neat_config)
        print(f"Winner {self._winner}")
        return self.Best_network
    
    def validation(self, test_set: Optional[Sequence] = None , target_set: Optional[Sequence] = None) -> dict[str, float]:
        if test_set is None and target_set is None:
            predicted = np.array([Evolution._binarize_prediction(self.Best_network.activate(x)[0]) for x in self._dataset.test_set])
            target_set = self._dataset.test_targets
        elif (test_set is None and target_set is not None) or (test_set is not None and target_set is None):
            assert ValueError("Invalid test set or target set")
        else:
            predicted = np.array([Evolution._binarize_prediction(self.Best_network.activate(x)[0]) for x in test_set])

        return {
            'f1_score' : sklearn.metrics.f1_score(y_pred=predicted, y_true=target_set),
            'accuracy' : sklearn.metrics.accuracy_score(y_pred=predicted, y_true=target_set),
            'precision' : sklearn.metrics.precision_score(y_pred=predicted, y_true=target_set),
            'recall' : sklearn.metrics.recall_score(y_pred=predicted, y_true=target_set),
            'confusion_matrix' : sklearn.metrics.confusion_matrix(y_pred=predicted, y_true=target_set),
            'balanced_accuracy': sklearn.metrics.balanced_accuracy_score(y_pred=predicted, y_true=target_set),
        }

    def plot_network(self, save_path :str, view = False ):
        if self.Best_network is None:
            return # nothing to vizualize
        visualize.draw_net(config=self._neat_config,genome=self._winner, view=view, filename=save_path)
        
    def plot_statistics(self, save_path :str, view = False ):
        if self._statistics is None:
            return # nothing to vizualize
        visualize.plot_stats(self._statistics, filename=save_path, view=view)