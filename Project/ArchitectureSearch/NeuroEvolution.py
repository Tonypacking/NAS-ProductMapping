import neat.config
import neat.genes
import numpy as np
import sklearn
import neat
import sklearn.metrics
from typing import Sequence, Optional
from Dataset import Dataset
class Evolution:
    
    def __init__(self, config_path: str, dataset:Dataset = None):
        self._dataset = dataset
        self.Name = self._dataset.dataset_name
        self._neat_config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
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
            genome.fitness = 0.0

            net = neat.nn.FeedForwardNetwork.create(genome=genome, config=config)

            predicted = np.array([Evolution._binarize_prediction(net.activate(x)[0]) for x in self._dataset.train_set])
            genome.fitness = sklearn.metrics.f1_score(y_pred=predicted, y_true=self._dataset.train_targets) * self._fitness_scaling

            
    def run(self, iterations: int = 50) -> neat.nn.FeedForwardNetwork:

        population = neat.Population(self._neat_config)

        population.add_reporter(neat.StdOutReporter(show_species_detail=True))
        stats = neat.StatisticsReporter()
        population.add_reporter(stats)
 
        winner = population.run(self._eval_genomes, iterations)

        self.Best_network = neat.nn.FeedForwardNetwork.create(winner, self._neat_config)
        print(f"Winner {winner}")
        return self.Best_network
    
    def validation(self, test_set: Optional[Sequence] = None , target_set: Optional[Sequence] = None) -> float:
        if test_set is None and target_set is None:
            predicted = np.array([Evolution._binarize_prediction(self.Best_network.activate(x)[0]) for x in self._dataset.test_set])
            target_set = self._dataset.test_targets
        elif (test_set is None and target_set is not None) or (test_set is not None and target_set is None):
            assert ValueError("Invalid test set or target set")
        else:
            predicted = np.array([Evolution._binarize_prediction(self.Best_network.activate(x)[0]) for x in test_set])
        return sklearn.metrics.f1_score(y_pred=predicted, y_true=target_set)
