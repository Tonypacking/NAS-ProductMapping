import neat.config
import neat.genes
import numpy as np
import sklearn
import neat
import sklearn.discriminant_analysis
import sklearn.metrics
from typing import Sequence, Optional
import multiprocessing
import sklearn.metrics._base
import configparser
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..'))) # To load Utils module
from Utils.Dataset import Dataset
import Utils.visualize as visualize
import Utils.ProMap as ProMap
import logging
import HyperNEAT.es_hyperneat
from HyperNEAT.shared import Substrate
from HyperNEAT.es_hyperneat import ESNetwork
from HyperNEAT.shared import draw_net, draw_es
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class NEATEvolution:
    
    def __init__(self, config_path: str, dataset:Dataset = None, scaling : bool= False, dimension_reduction : str = 'raw'):
        self._dataset :Dataset = dataset
        self.dataset_name = self._dataset.dataset_name
        self.Best_network = None
        self._fitness_scaling = 1_000 
        self._transformer = None
        self._scaler = None

        self._neat_config = self._create_config(config_path, scaling=scaling, dimension_reduction=dimension_reduction)
        self._population = neat.Population(self._neat_config)
 
    @staticmethod
    def _binarize_prediction(x: float) -> int:
        """
        Binarize prediction to matching or None Matching
        """
        return 1 if x >= 0.5 else 0
    
    def _eval_genomes(self, genomes, config):
        """Evaluates genomes

        Args:
            genomes (_type_): genomes
            config (_type_): path to config file.
        """
        for id, genome in genomes:
            # genome.fitness = self._dataset.train_set.shape[0]

            net = neat.nn.FeedForwardNetwork.create(genome=genome, config=config)

            predicted = np.array([NEATEvolution._binarize_prediction(net.activate(x)[0]) for x in self._dataset.train_set])
            genome.fitness = sklearn.metrics.f1_score(y_pred=predicted, y_true=self._dataset.train_targets) * self._fitness_scaling

    def _create_config(self, config_path, scaling : bool = False, dimension_reduction: str = 'raw') -> neat.Config:
        """Creates config for genomes. If dataset's dimensions is reduced, it updates neat's input nodes.

        Args:
            config_path (_type_): Path to config file
            scaling (bool, optional): Scales features through StandardScaler before running neat. Defaults to False.
            dimension_reduction (str, optional): Reduces dimensions before running neat. Defaults to 'raw'.

        Returns:
            neat.Config: Neat configuration.
        """
        if scaling:
            self._scaler = self._dataset.scale_features()

        if dimension_reduction == 'lda' or dimension_reduction == 'pca':
            # number of input nodes are reduce. Dynamically change neat config also.
            self._transformer = self._dataset.reduce_dimensions(dimension_reduction)

            num_features = self._dataset.train_set.shape[1]
            parser = configparser.ConfigParser()
            parser.read(config_path)
            parser.set('DefaultGenome', 'num_inputs', str(num_features))

            with open(config_path, 'w') as f:
                parser.write(f)
        else: #
            num_features = self._dataset.train_set.shape[1]
            parser = configparser.ConfigParser()
            parser.read(config_path)
            parser.set('DefaultGenome', 'num_inputs', str(num_features))

            with open(config_path, 'w') as f:
                parser.write(f)
        # create Config for genome
        return neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

    def RunNEAT(self, iterations: int = 50, parralel: bool = False) -> neat.nn.FeedForwardNetwork:
        """Runs the fining algorithm using neat.

        Args:
            iterations (int, optional): Number of generations in neat. Defaults to 50.
            parralel (bool, optional): Parralel evaluation of genomes. Defaults to False.

        Returns:
            neat.nn.FeedForwardNetwork: _description_
        """
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
        # TODO: add logging winner
        #print(f"Winner {self._winner}")
        return self.Best_network
    
    def validate(self, test_set: Optional[Sequence] = None , target_set: Optional[Sequence] = None) -> dict[str, float]:
        """Va;odates best network against unseen data.

        Args:
            test_set (Optional[Sequence], optional): testing set. Defaults to None.
            target_set (Optional[Sequence], optional): testing true outpiut. Defaults to None.

        Returns:
            dict[str, float]: dictionary of name of a metric and metric's value
        """
        if test_set is None and target_set is None:
            predicted = np.array([NEATEvolution._binarize_prediction(self.Best_network.activate(x)[0]) for x in self._dataset.test_set])
            target_set = self._dataset.test_targets

        elif (test_set is None and target_set is not None) or (test_set is not None and target_set is None):
            assert ValueError("Invalid test set or target set")
        else:
            predicted = np.array([NEATEvolution._binarize_prediction(self.Best_network.activate(x)[0]) for x in test_set])

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
        # TODO train accuracy
        for name in ProMap.ProductsDatasets.NAME_MAP:
            tested_dataset= ProMap.ProductsDatasets.Load_by_name(name)

            if tested_dataset.feature_labels.shape < self._dataset.feature_labels.shape:
                #print(tested_dataset.feature_labels.shape, tested_dataset.test_set.shape,tested_dataset.train_set.shape)
               # print(tested_dataset.test_set.shape)
                tested_dataset.extend_dataset(self._dataset)
                # print(tested_dataset.test_set.shape)
                # print(tested_dataset.feature_labels.shape, tested_dataset.test_set.shape,tested_dataset.train_set.shape)

            elif tested_dataset.feature_labels.shape > self._dataset.feature_labels.shape:
                # print(tested_dataset.feature_labels.shape, self._dataset.test_set.shape, tested_dataset.train_set.shape)
                tested_dataset.reduce_dataset(self._dataset)
                # print(tested_dataset.feature_labels.shape, tested_dataset.test_set.shape, tested_dataset.train_set.shape)

            if self._scaler:
                # print(tested_dataset.test_set.shape)
                tested_dataset.test_set = self._scaler.transform(tested_dataset.test_set)
                
            if self._transformer:
                tested_dataset.test_set = self._transformer.transform(tested_dataset.test_set)

            outputs.append((tested_dataset.dataset_name, self.validate(tested_dataset.test_set, tested_dataset.test_targets)))
        return outputs
    
    def plot_network(self, save_path :str, view = False ):
        """Plots the best genome's network

        Args:
            save_path (str): save path
            view (bool, optional): View of a plot durring runtime. Defaults to False.

        Returns:
            None: None
        """
        if self.Best_network is None:
            return # nothing to vizualize
        visualize.draw_net(config=self._neat_config,genome=self._winner, view=view, filename=save_path)
        
    def plot_statistics(self, save_path :str, view = False ):
        """Plots neat staticstics

        Args:
            save_path (str): save path
            view (bool, optional): View of a plot durring runtime. Defaults to False.

        Returns:
            None: None
        """
        if self._statistics is None:
            return # nothing to vizualize
        visualize.plot_stats(self._statistics, filename=save_path, view=view)



class ESHyperNEATEvolution:
    
    def __init__(self, config_path: str, version, dataset:Dataset = None, scaling : bool= False, dimension_reduction : str = 'raw', fitness = 'Acc'):
        self._dataset :Dataset = dataset
        self.dataset_name = self._dataset.dataset_name
        self._best_CPPPN = None
        self._fitness_scaling = 100
        self._transformer = None
        self._scaler = None
        self._substrate = None
        self._neat_config = self._create_config(config_path, scaling=scaling, dimension_reduction=dimension_reduction)
        self._best_network_architecture = None
        num_features = self._dataset.train_set.shape[1]
        #generate points from -1 to 1
        x_coords = np.linspace(start=-1, stop=1, num=num_features, endpoint=True)
        y_coords = np.linspace(start=-1, stop=1, num=num_features, endpoint=True)

        input_coord = np.column_stack((x_coords, y_coords)).tolist()
        self.input_coord = [tuple(coord) for coord in input_coord]
        # output is
        self.output_coord = [( -1.0,1.0)]
        self._substrate = Substrate(input_coordinates=self.input_coord, output_coordinates=self.output_coord)
        # create Config for genome
        self._population = neat.Population(self._neat_config)
        self._params = self._params(version=version)
        self._fitness = sklearn.metrics.accuracy_score if fitness=="Acc" else sklearn.metrics.f1_score


    def _params(self, version):
        """
        ES-HyperNEAT specific parameters.
        """
        return {"initial_depth": 0 if version == "S" else 1 if version == "M" else 2,
                "max_depth": 1 if version == "S" else 2 if version == "M" else 3,
                "variance_threshold": 0.03,
                "band_threshold": 0.3,
                "iteration_level": 5,
                "division_threshold": 0.5,
                "max_weight": 30.0,
                "activation": "sigmoid"}

    def _activate_network(self, network, input, activations):
        network.reset()
        for _ in range(activations):
            output = network.activate(input)      
        return output

    def _eval_genomes(self, genomes, config):
        for id, genome in genomes:
            cppn = neat.nn.FeedForwardNetwork.create(genome, config)
            esnetwork = ESNetwork(self._substrate, cppn, self._params)

            rec_network = esnetwork.create_phenotype_network()
            predictions = np.array([ESHyperNEATEvolution._binarize_prediction(self._activate_network(network=rec_network,input=x, activations=esnetwork.activations)[0]) for x in self._dataset.train_set])
            predictions = np.round(predictions) # * self._fitness_scaling

            genome.fitness = self._fitness(y_true=self._dataset.train_targets, y_pred=predictions) * 1_00
 
    @staticmethod
    def _binarize_prediction(x: float) -> int:
        """
        Binarize prediction to matching or None Matching
        """
        return 1 if x >= 0.5 else 0

    def _create_config(self, config_path, scaling : bool = False, dimension_reduction: str = 'raw') -> neat.Config:

        if scaling:
            self._scaler = self._dataset.scale_features()

        if dimension_reduction == 'lda' or dimension_reduction == 'pca':
            # number of input nodes are reduce. Dynamically change neat config also.
            self._transformer = self._dataset.reduce_dimensions(dimension_reduction)

        return neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

    def RunESHyperNEAT(self, generations: int = 50, parralel: bool = False) -> tuple[neat.nn.FeedForwardNetwork,ESNetwork] :
        """Runs the fining algorithm using HyperNEAT.

        Args:
            iterations (int, optional): Number of generations in neat. Defaults to 50.
            parralel (bool, optional): Parralel evaluation of genomes. Defaults to False.

        Returns:
            neat.nn.FeedForwardNetwork: _description_
        """
        population = neat.Population(self._neat_config)

        population.add_reporter(neat.StdOutReporter(show_species_detail=True))
        self._statistics = neat.StatisticsReporter()
        population.add_reporter(self._statistics)
        
        if parralel:
            para_eval = neat.ParallelEvaluator(num_workers=multiprocessing.cpu_count(),eval_function=self._eval_genomes)
            self._neat_winner = population.run(para_eval.eval_function, generations)
        else:
            self._neat_winner = population.run(self._eval_genomes, generations)

        self._best_CPPPN = neat.nn.FeedForwardNetwork.create(self._neat_winner, self._neat_config)
        self._best_network_architecture = ESNetwork(self._substrate, self._best_CPPPN, self._params)

        return self._best_CPPPN, self._best_network_architecture
    
    def validate(self, test_set: Optional[Sequence] = None , target_set: Optional[Sequence] = None) -> dict[str, float]:
        """Va;odates best network against unseen data.

        Args:
            test_set (Optional[Sequence], optional): testing set. Defaults to None.
            target_set (Optional[Sequence], optional): testing true outpiut. Defaults to None.

        Returns:
            dict[str, float]: dictionary of name of a metric and metric's value
        """
        activations = self._best_network_architecture.activations

        esnetwork = ESNetwork(self._substrate, self._best_CPPPN, self._params)
        network = esnetwork.create_phenotype_network()
        
        if test_set is None and target_set is None:
            predicted = np.array([self._activate_network(network=network,input=x, activations=activations)[0] for x in self._dataset.test_set])
            
            target_set = self._dataset.test_targets

        elif (test_set is None and target_set is not None) or (test_set is not None and target_set is None):
            assert ValueError("Invalid test set or target set")
        else:
            predicted = np.array([self._activate_network(network=network,input=x, activations=activations)[0] for x in test_set])
        predicted = np.round(predicted)
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
                #print(tested_dataset.feature_labels.shape, tested_dataset.test_set.shape,tested_dataset.train_set.shape)
                tested_dataset.extend_dataset(self._dataset)
                # print(tested_dataset.feature_labels.shape, tested_dataset.test_set.shape,tested_dataset.train_set.shape)

            elif tested_dataset.feature_labels.shape > self._dataset.feature_labels.shape:
                # print(tested_dataset.feature_labels.shape, self._dataset.test_set.shape, tested_dataset.train_set.shape)
                tested_dataset.reduce_dataset(self._dataset)
                # print(tested_dataset.feature_labels.shape, tested_dataset.test_set.shape, tested_dataset.train_set.shape)

            if self._scaler:
                tested_dataset.test_set = self._scaler.transform(tested_dataset.test_set)
                
            if self._transformer:
                tested_dataset.test_set = self._transformer.transform(tested_dataset.test_set)

            outputs.append((tested_dataset.dataset_name, self.validate(tested_dataset.test_set, tested_dataset.test_targets)))
        return outputs
    
    def plot_best_network(self, file_path):
        """
        Red edge are inactive black edges are active
        poitns are scatter from -1 to 1
        output node is at  -1 0

        Args:
            file_path str: file output path
        """
        if self._best_CPPPN is None or self._best_network_architecture is None:
            logger.warning('HyperNEAT didnt finish ignoring plot')
            return
        logger.debug(f"Input coordinates{self.input_coord} output coordinates {self.output_coord}. Black edges are active red edges are inactive.")
        esnetwork = ESNetwork(self._substrate, self._best_CPPPN, self._params)
        esnetwork.create_phenotype_network(file_path)
        
    def plot_CPPN_network(self, save_path :str ):
        """Plots the best genome's network

        Args:
            save_path (str): save path
            view (bool, optional): View of a plot durring runtime. Defaults to False.

        Returns:
            None: None
        """
        if self._best_CPPPN is None:
            return # nothing to vizualize
        draw_net(self._best_CPPPN, save_path)
        
    def plot_statistics(self, save_path :str, view = False ):
        """Plots neat staticstics

        Args:
            save_path (str): save path
            view (bool, optional): View of a plot durring runtime. Defaults to False.

        Returns:
            None: None
        """
        if self._statistics is None:
            return # nothing to vizualize
        visualize.plot_stats(self._statistics, filename=save_path, view=view)
        
   # def draw_network(self, save_path):
        #draw_net(self.)


class HyperNEATEvolution:
    
    def __init__(self, config_path: str, hidden_layers, dataset:Dataset = None, scaling : bool= False, dimension_reduction : str = 'raw', fitness = 'F1'):
        self._dataset :Dataset = dataset
        self.dataset_name = self._dataset.dataset_name
        self._best_CPPPN = None
        self._fitness_scaling = 1000
        self._transformer = None
        self._scaler = None
        self._substrate = None
        self._neat_config = self._create_config(config_path, scaling=scaling, dimension_reduction=dimension_reduction)
        self._best_network_architecture = None
        num_features = self._dataset.train_set.shape[1]
        #generate points from -0.5 to 0.5
        x_coords = np.linspace(start=-1, stop=0, num=num_features, endpoint=True)
        y_coords = np.linspace(start=-1, stop=0, num=num_features, endpoint=True)

        input_coord = np.column_stack((x_coords, y_coords)).tolist()
        self.input_coord = [tuple(coord) for coord in input_coord]
        self.hidden_cord = [[(-0.5, 0.5), (0.5, 0.5)], [(-0.5, -0.5), (0.5, -0.5)]]

        self.hidden_cord = self._create_hidden_coordinates(hidden_layers)

        self.output_coord = [( -1,1), (1, -1)]

        self.activations = len(self.hidden_cord) + 2
        self._substrate = Substrate(self.input_coord, self.output_coord, self.hidden_cord)
        # create Config for genome
        self._population = neat.Population(self._neat_config)
        self._fitness = sklearn.metrics.accuracy_score if fitness == "Acc" else sklearn.metrics.f1_score 

    def _create_hidden_coordinates(self, hidden_layers):
        hidden_coordinates = []
        for indx, layer_size in enumerate(hidden_layers):
            logger.debug(f"Layer {indx} size {layer_size}")
            x_cord = np.linspace(start=indx, stop=indx+1, num=layer_size, endpoint=False)
            y_cord = np.linspace(start=indx, stop=indx+1, num=layer_size, endpoint=False)
            input_coord = np.column_stack((x_cord, y_cord)).tolist()
            x = [tuple(coord) for coord in input_coord]
            hidden_coordinates.append(x)

        return hidden_coordinates
    

    def _predict_data(self, network, x, activations):

        network.reset()
        for _ in range(activations):
            output = network.activate(x)
        return np.argmax(output)

    def _eval_genomes(self, genomes, config):
        for id, genome in genomes:
            cppn = neat.nn.FeedForwardNetwork.create(genome, config)
            net = HyperNEAT.hyperneat.create_phenotype_network(cppn=cppn, substrate=self._substrate)

           # predictions = np.array([HyperNEATEvolution._binarize_prediction(self._activate_network(network=rec_network,input=x, activations=esnetwork.activations)[0]) for x in self._dataset.train_set])
            predictions = []
            for x in self._dataset.train_set:
                net.reset()
                for _ in range(self.activations):
                    output = net.activate(x)
                predictions.append(output)

            predictions = np.array([self._predict_data(network=net, x=x, activations=self.activations) for x in self._dataset.train_set])
            #predictions = np.round(predictions)
            genome.fitness = sklearn.metrics.accuracy_score(y_true=self._dataset.train_targets, y_pred=predictions) * 100
            genome.fitness = self._fitness(y_true=self._dataset.train_targets, y_pred=predictions) * 100
 
    def _create_config(self, config_path, scaling : bool = False, dimension_reduction: str = 'raw') -> neat.Config:

        if scaling:
            self._scaler = self._dataset.scale_features()

        if dimension_reduction == 'lda' or dimension_reduction == 'pca':
            # number of input nodes are reduce. Dynamically change neat config also.
            self._transformer = self._dataset.reduce_dimensions(dimension_reduction)

        return neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

    def RunHyperNEAT(self, generations: int = 50, parralel: bool = False) -> tuple[neat.nn.FeedForwardNetwork,ESNetwork] :
        """Runs the fining algorithm using HyperNEAT.

        Args:
            iterations (int, optional): Number of generations in neat. Defaults to 50.
            parralel (bool, optional): Parralel evaluation of genomes. Defaults to False.

        Returns:
            neat.nn.FeedForwardNetwork: _description_
        """
        population = neat.Population(self._neat_config)

        population.add_reporter(neat.StdOutReporter(show_species_detail=True))
        self._statistics = neat.StatisticsReporter()
        population.add_reporter(self._statistics)
        
        if parralel:
            para_eval = neat.ParallelEvaluator(num_workers=multiprocessing.cpu_count(),eval_function=self._eval_genomes)
            self._neat_winner = population.run(para_eval.eval_function, generations)
        else:
            self._neat_winner = population.run(self._eval_genomes, generations)

        self._best_CPPPN = neat.nn.FeedForwardNetwork.create(self._neat_winner, self._neat_config)
        self._best_network_architecture = HyperNEAT.hyperneat.create_phenotype_network(cppn=self._best_CPPPN, substrate=self._substrate)

        return self._best_CPPPN, self._best_network_architecture
    
    def validate(self, test_set: Optional[Sequence] = None , target_set: Optional[Sequence] = None) -> dict[str, float]:
        """Va;odates best network against unseen data.

        Args:
            test_set (Optional[Sequence], optional): testing set. Defaults to None.
            target_set (Optional[Sequence], optional): testing true outpiut. Defaults to None.

        Returns:
            dict[str, float]: dictionary of name of a metric and metric's value
        """
        activations = self.activations

       #  esnetwork = ESNetwork(self._substrate, self._best_CPPPN, self._params)
        network =  self._best_network_architecture
        
        if test_set is None and target_set is None:
            predicted = np.array([self._predict_data(network=network,x=x, activations=activations) for x in self._dataset.test_set])
         #   predicted = (predicted >= 0.5).astype(int)
            target_set = self._dataset.test_targets

        elif (test_set is None and target_set is not None) or (test_set is not None and target_set is None):
            assert ValueError("Invalid test set or target set")
        else:
            predicted = np.array([self._predict_data(network=network,x=x, activations=activations) for x in test_set])
           # predicted = (predicted >= 0.5).astype(int)
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
                #print(tested_dataset.feature_labels.shape, tested_dataset.test_set.shape,tested_dataset.train_set.shape)
                tested_dataset.extend_dataset(self._dataset)
                # print(tested_dataset.feature_labels.shape, tested_dataset.test_set.shape,tested_dataset.train_set.shape)

            elif tested_dataset.feature_labels.shape > self._dataset.feature_labels.shape:
                # print(tested_dataset.feature_labels.shape, self._dataset.test_set.shape, tested_dataset.train_set.shape)
                tested_dataset.reduce_dataset(self._dataset)
                # print(tested_dataset.feature_labels.shape, tested_dataset.test_set.shape, tested_dataset.train_set.shape)

            if self._scaler:
                tested_dataset.test_set = self._scaler.transform(tested_dataset.test_set)
                
            if self._transformer:
                tested_dataset.test_set = self._transformer.transform(tested_dataset.test_set)

            outputs.append((tested_dataset.dataset_name, self.validate(tested_dataset.test_set, tested_dataset.test_targets)))
        return outputs
    
    def plot_best_network(self, file_path):
        """
        Red edge are inactive black edges are active
        poitns are scatter from -1 to 1
        output node is at  -1 0

        Args:
            file_path str: file output path
        """
        if self._best_CPPPN is None or self._best_network_architecture is None:
            logger.warning('HyperNEAT didnt finish ignoring plot')
            return
        logger.debug(f"Input coordinates{self.input_coord} output coordinates {self.output_coord}. Black edges are active red edges are inactive.")
       # esnetwork = HyperNEAT.hyperneat.create_phenotype_network(cppn=self._best_CPPPN, substrate=self._substrate)
        HyperNEAT.shared.draw_net(self._best_network_architecture, filename=file_path)

        
    def plot_CPPN_network(self, save_path :str ):
        """Plots the best genome's network

        Args:
            save_path (str): save path
            view (bool, optional): View of a plot durring runtime. Defaults to False.

        Returns:
            None: None
        """
        if self._best_CPPPN is None:
            return # nothing to vizualize
        draw_net(self._best_CPPPN, save_path)
        
    def plot_statistics(self, save_path :str, view = False ):
        """Plots neat staticstics

        Args:
            save_path (str): save path
            view (bool, optional): View of a plot durring runtime. Defaults to False.

        Returns:
            None: None
        """
        if self._statistics is None:
            return # nothing to vizualize
        visualize.plot_stats(self._statistics, filename=save_path, view=view)
        