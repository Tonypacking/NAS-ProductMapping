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
print(os.path.dirname(os.path.abspath(__file__)))
current_script_dir = os.path.dirname(os.path.abspath(__file__))
kerascodeepneat_path = os.path.join(current_script_dir,"..", "base", "kerascodeepneat.py")
spec = importlib.util.spec_from_file_location("kerascodeepneat", kerascodeepneat_path)

# spec = importlib.util.spec_from_file_location("kerascodeepneat", "../base/kerascodeepneat.py")
kerascodeepneat = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kerascodeepneat)

class CoDeepNEAT:

    def __init__(self):
        self.best_network: keras.models.Model = None    
        self._scaler = None
        self._dimension_reduction = None
        self._train_dataset : Dataset|None = None

    def RunCoDeepNEAT(self, args: argparse.Namespace ):
        
        codeepneat_parser = CoDeepNeatParser()
        codeepneat_parser.load_config(args.input)

        self._train_dataset = ProductsDatasets.Load_by_name(args.dataset)

        save_path_to_codeepneat = os.path.join(args.output, 'CoDeepNEAT')
        if not os.path.exists(save_path_to_codeepneat):
            os.makedirs(save_path_to_codeepneat, exist_ok=True)
        
        if args.scale:
            self._scaler = self._train_dataset.scale_features()
        
        if args.dimension_reduction != 'raw':
            self._dimension_reduction = self._train_dataset.reduce_dimensions(args.dimension_reduction)
        
        generations = codeepneat_parser.generations
        training_epochs = codeepneat_parser.training_epochs
        population_size = codeepneat_parser.population_size

        final_model_training_epochs = codeepneat_parser.final_model_training_epoch
        blueprint_population_size = codeepneat_parser.blueprint_population_size
        module_population_size = codeepneat_parser.module_population_size
        n_blueprint_species = codeepneat_parser.n_blueprint_species
        n_module_species = codeepneat_parser.n_module_species  

        global_configs = codeepneat_parser.Get_global_config()

        input_configs = codeepneat_parser.Get_input_configs()

        output_configs = codeepneat_parser.Get_output_configs()

        possible_components = codeepneat_parser.Get_possible_components()

        possible_inputs = codeepneat_parser.Get_Possible_inputs()

        possible_outputs = codeepneat_parser.Get_possible_outputs()

        possible_complementary_components = codeepneat_parser.Get_possible_complementaty_components()

        possible_complementary_inputs = codeepneat_parser.Get_possible_complementary_inputs()

        possible_complementary_outputs = codeepneat_parser.Get_possible_complementary_outputs()

        
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

        my_dataset.SAMPLE_SIZE = int(x_train.shape[0] * 0.4)
        my_dataset.TEST_SAMPLE_SIZE = int( x_test.shape[0] * 0.2)

        logging.basicConfig(filename='execution.log',
                            filemode='w+', level=logging.INFO,
                            format='%(levelname)s - %(asctime)s: %(message)s')
        
        logging.addLevelName(21, "TOPOLOGY")

        compiler = {"loss":"categorical_crossentropy", "optimizer":"keras.optimizers.Adam(lr=0.005)", "metrics":["accuracy"]}

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
        #print(self._train_dataset.test_set[0])
        # TODO reshape training dataset
        test_data = self._train_dataset.test_set.reshape(self._train_dataset.test_set.shape[0], self._train_dataset.test_set.shape[1], 1, 1)
        predictions = self.best_network.predict(test_data)
        print(np.argmax(predictions))

        print(f"accuracy: {sklearn.metrics.accuracy_score(y_pred=np.argmax(predictions,axis=1), y_true=self._train_dataset.test_targets)}")
    # print(best_model)
        #'Best model')
        # Set data augmentation for full training
        # population.datasets = improved_dataset
    # print("Using data augmentation.")

        # TODO Fix retraining best model architecture
        print(f"Best fitting model {best_individual.name}")
        # retrainign doesnt work
        # try:
        #     print(f"Best fitting model chosen for retraining: {best_model.name}")
        #     print(f"Name {best_model.name} final model {final_model_training_epochs} valsplit {validation_split} custom args")
        #     population.train_full_model(best_model, final_model_training_epochs, validation_split, None)
        # except Exception  as e:
        #     print(e)

        #     population.individuals.remove(best_model)
        #     best_model = population.return_best_individual()
        #     print(f"Best fitting model chosen for retraining: {best_model.name}")
        #     population.train_full_model(best_model, final_model_training_epochs, validation_split, None)
    
        #population.individuals.remove(best_model)
        #best_model = population.return_best_individual()
        # print(f"Best fitting model chosen for retraining: {best_model.name}")
    # population.train_full_model(best_model, final_model_training_epochs, validation_split, None)


    def validate(self):
        pass

    def validate_all(self):

        for dataset_name in ProductsDatasets.NAME_MAP.keys():
            pass

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
                if exc.errno != errno.EEXIST:
                    raise

    create_dir("models/")
    create_dir("images/")
    # RunCoDeepNEAT(generations, training_epochs, population_size, blueprint_population_size, module_population_size, n_blueprint_species, n_module_species, final_model_training_epochs)
    # run_cifar10_full(generations, training_epochs, population_size, blueprint_population_size, module_population_size, n_blueprint_species, n_module_species, final_model_training_epochs)