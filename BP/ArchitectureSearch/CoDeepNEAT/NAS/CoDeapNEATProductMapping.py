import sys

import sklearn.metrics
sys.path.append("..")

import keras, logging, random, pydot, copy, uuid, os, csv, json
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from enum import Enum, auto
from typing import List
# from keras.utils.vis_utils import plot_model
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestCentroid
from sklearn.preprocessing import scale
from keras import backend as K
# from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from keras import regularizers
# import imp
# kerascodeepneat = imp.load_source("kerascodeepneat", "./base/kerascodeepneat.py")
# kerascodeepneat = imp.load_source("kerascodeepneat", "../base/kerascodeepneat.py")
import importlib.util
import os
from BP.ArchitectureSearch.CoDeepNEAT.CoDeepNeatParser import CoDeepNeatParser
from Utils.ProMap import ProductsDatasets, Dataset
import argparse
import sklearn
CODEEPNEAT_DIRECTORY = "CoDeepNEAT" 
print(os.path.dirname(os.path.abspath(__file__)))
current_script_dir = os.path.dirname(os.path.abspath(__file__))
kerascodeepneat_path = os.path.join(current_script_dir,"..", "base", "kerascodeepneat.py")
spec = importlib.util.spec_from_file_location("kerascodeepneat", kerascodeepneat_path)

# spec = importlib.util.spec_from_file_location("kerascodeepneat", "../base/kerascodeepneat.py")
kerascodeepneat = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kerascodeepneat)

class CoDeepNEAT:
    """Class for neural architecture search with CoDeepNEAT strategy
    """
    def __init__(self):
        self.best_network: keras.models.Model = None    
        self._scaler = None
        self._transformer = None
        self._train_dataset : Dataset|None = None
        self.training_parameters = None

    def RunCoDeepNEAT(self, args: argparse.Namespace, save_path_to_codeepneat ):
        """Runs CoDeepNEAT

        Args:
            args (argparse.Namespace): Users arguments
            save_path_to_codeepneat (str): Save directory
        """
        codeepneat_parser = CoDeepNeatParser()
        codeepneat_parser.load_config(args.input)

        self.training_parameters = codeepneat_parser.parameters
        self._train_dataset = ProductsDatasets.Load_by_name(args.dataset)


        if not os.path.exists(save_path_to_codeepneat):
            os.makedirs(save_path_to_codeepneat, exist_ok=True)
        
        if args.scale:
            self._scaler = self._train_dataset.scale_features()
        
        if args.dimension_reduction != 'raw':
            self._transformer = self._train_dataset.reduce_dimensions(args.dimension_reduction)
        
        generations = codeepneat_parser.generations
        training_epochs = codeepneat_parser.training_epochs
        population_size = codeepneat_parser.population_size

        final_model_training_epochs = codeepneat_parser.final_model_training_epoch
        blueprint_population_size = codeepneat_parser.blueprint_population_size
        module_population_size = codeepneat_parser.module_population_size
        n_blueprint_species = codeepneat_parser.n_blueprint_species
        n_module_species = codeepneat_parser.n_module_species  

        global_configs = codeepneat_parser.Get_global_config()
        self.training_parameters += f";global_configs-{global_configs}"

        input_configs = codeepneat_parser.Get_input_configs()
        self.training_parameters += f";input_configs-{input_configs}"
        output_configs = codeepneat_parser.Get_output_configs()
        self.training_parameters += f";-output_configs{output_configs}"
        possible_components = codeepneat_parser.Get_possible_components()
        self.training_parameters += f";-possible_components{possible_components}"
        possible_inputs = codeepneat_parser.Get_Possible_inputs()
        self.training_parameters += f";possible_inputs-{possible_inputs}"
        possible_outputs = codeepneat_parser.Get_possible_outputs()
        self.training_parameters += f";possible_outputs-{possible_outputs}"
        possible_complementary_components = codeepneat_parser.Get_possible_complementaty_components()
        self.training_parameters += f";possible_complementary_components-{possible_complementary_components}"
        possible_complementary_inputs = codeepneat_parser.Get_possible_complementary_inputs()
        self.training_parameters += f";possible_complementary_inputs-{possible_complementary_inputs}"
        possible_complementary_outputs = codeepneat_parser.Get_possible_complementary_outputs()
        self.training_parameters += f";possible_complementary_outputs-{possible_complementary_outputs}"
        
        num_classes = 2


        # Convert class vectors to binary class matrices.
        y_train = keras.utils.to_categorical(self._train_dataset.train_targets, num_classes)
        y_test = keras.utils.to_categorical(self._train_dataset.test_targets, num_classes)

        x_train = self._train_dataset.train_set
        x_test = self._train_dataset.test_set

        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1, 1))
        x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1, 1))
    
        validation_split = 0.10
        #training
        batch_size = 32

        my_dataset = kerascodeepneat.Datasets(training=[x_train, y_train], test=[x_test, y_test])

        my_dataset.SAMPLE_SIZE = int(x_train.shape[0])
        my_dataset.TEST_SAMPLE_SIZE = int( x_test.shape[0])

        logging.basicConfig(filename='execution.log',
                            filemode='w+', level=logging.INFO,
                            format='%(levelname)s - %(asctime)s: %(message)s')
        
        logging.addLevelName(21, "TOPOLOGY")

        compiler = {"loss":"binary_crossentropy", "optimizer":"keras.optimizers.Adam(lr=0.005)", "metrics":["accuracy", keras.metrics.Precision(name='precision'),
                                                                                                            keras.metrics.F1Score(name='f1'),
                                                                                                            keras.metrics.Recall(name='recall')
                                                                                                                 
                                                                                                                 ]}

        # Set configurations for full training session (final training)
        es = EarlyStopping(monitor='val_acc', mode='auto', verbose=1, patience=15)
        mc = ModelCheckpoint('best_model_checkpoint.h5', monitor='val_accuracy', mode='auto', verbose=1, save_best_only=True)
        csv_logger = CSVLogger('training.csv')
        
        custom_fit_args = {#" generator": datagen.flow(x_train, y_train, batch_size=batch_size),
        "steps_per_epoch": x_train.shape[0] // batch_size,
        "epochs": training_epochs,
        "verbose": 1,
        "validation_data": (x_test,y_test),
        "callbacks": [es, csv_logger]
        }          

        improved_dataset = kerascodeepneat.Datasets(training=[x_train, y_train], test=[x_test, y_test])
        improved_dataset.custom_fit_args = custom_fit_args
        my_dataset.custom_fit_args = None

        # Initiate population
        population = kerascodeepneat.Population(my_dataset, input_shape=x_train.shape[1:], population_size=population_size, compiler=compiler, save_directory= save_path_to_codeepneat)
        
        # Start with random modules
        population.create_module_population(module_population_size, global_configs, possible_components, possible_complementary_components)
        population.create_module_species(n_module_species)

        # Start with random modules
        population.create_blueprint_population(blueprint_population_size,
                                                global_configs, possible_components, possible_complementary_components,
                                                input_configs, possible_inputs, possible_complementary_inputs,
                                                output_configs, possible_outputs, possible_complementary_outputs)
        population.create_blueprint_species(n_blueprint_species)

        # Iterate generating, fitting, scoring, speciating, reproducing and mutating.
        iteration = population.iterate_generations(generations=generations,
                                                    training_epochs=training_epochs,
                                                    validation_split=validation_split,
                                                    mutation_rate= codeepneat_parser.mutation_rate,
                                                    crossover_rate=codeepneat_parser.crossover_rate,
                                                    elitism_rate=codeepneat_parser.elitism_rate,
                                                    possible_components=possible_components,
                                                    possible_complementary_components=possible_complementary_components)

        print("Best fitting: (Individual name, Blueprint mark, Scores[test loss, test acc], History).\n", (iteration))

        # Return the best model
        best_individual = population.return_best_individual()
        self.best_network = best_individual.model

        
        best_model = keras.Model.from_config(best_individual.model.get_config())

        best_model.compile(optimizer='adam', loss = 'categorical_crossentropy',metrics=[
            'accuracy',
             keras.metrics.Precision(),
             keras.metrics.Recall()    ,
                                                                
        print(f"Retraining best network architecture for epochs: {final_model_training_epochs}")       
                                                                           
                                                                                                                                              ])
        history = best_model.fit(x_train, y_train, batch_size=batch_size, epochs=final_model_training_epochs)

        self.best_network = best_model
        # TPDP save nest architecture

        print(f"Best fitting model {best_individual.name}")
        print(f"Best fitting model chosen for retraining: {best_individual.name}")

        best_model_path = os.path.join(save_path_to_codeepneat,f'BestArchitecture_{args.dataset}.keras')
        try:
            best_model.save(best_model_path)
            print(f"Best model saved successfully to: {best_model_path}")
            keras.utils.plot_model(best_model, to_file=os.path.join(save_path_to_codeepneat, f"BestArchitecture_{args.dataset}.png"), show_layer_activations=True)
            import json
            scores = {
                "history" : history,
            }

        except Exception as e:
            print(f"Error saving model to {best_model_path}: {e}")


    def validate(self, testing_dataset: Dataset| None = None):
        """Validates the best network against unseen data.

        Args:
            test_set (Optional[Sequence], optional): testing set. Defaults to None.
            target_set (Optional[Sequence], optional): testing true outpiut. Defaults to None.

        Returns:
            dict[str, float]: dictionary of name of a metric and metric's value
        """

        if testing_dataset is None:
            testing_dataset = self._train_dataset

        test_data = testing_dataset.test_set.reshape(testing_dataset.test_set.shape[0], testing_dataset.test_set.shape[1], 1, 1)
        target_set = testing_dataset.test_targets
        predicted = np.argmax(self.best_network.predict(test_data), axis=1)

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
        for name in ProductsDatasets.NAME_MAP:
            tested_dataset = ProductsDatasets.Load_by_name(name)

            if tested_dataset.feature_labels.shape < self._train_dataset.feature_labels.shape:
                tested_dataset.extend_dataset(self._train_dataset)
            elif tested_dataset.feature_labels.shape > self._train_dataset.feature_labels.shape:
                tested_dataset.reduce_dataset(self._train_dataset)
            
            if self._scaler:
                tested_dataset.test_set = self._scaler.transform(tested_dataset.test_set)
            if self._transformer :
                tested_dataset.test_set = self._transformer.transform(tested_dataset.test_set)
            outputs.append((tested_dataset.dataset_name, self.validate(tested_dataset)))
            
        return outputs

if __name__ == "__main__":

    generations = 2
    training_epochs = 5
    final_model_training_epochs = 2
    population_size = 1
    blueprint_population_size = 10
    module_population_size = 30
    n_blueprint_species = 3
    n_module_species = 3

    def create_dir(dir):
        if not os.path.exists(os.path.dirname(dir)):
            try:
                os.makedirs(os.path.dirname(dir))
            except OSError as exc: # Guard against race condition
                pass
                # if exc.errno != errno.EEXIST:
                #     raise

