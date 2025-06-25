import keras
import numpy as np
import random # For generating random numbers
import enum
import sys, os

import sklearn.metrics
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..','..'))) # To load Utils module
from Utils.ProMap import ProductsDatasets, Dataset
import argparse
from RandomSearchStrategy.RandomSearchParser import RandomSearchParser
import logging
import tensorflow as tf
import gc
import sklearn
from typing import Sequence, Optional
import Utils.ProMap as ProMap


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

CLEAR_MEMORY_INTERVAL = 15 # in RUN RANDOM SEARCH every 15th iteration keras memory and heap is cleared

class RandomSearch:
    def __init__(self, args:argparse):
        self.hyper_parameters = RandomSearchParser()
        self.hyper_parameters.parse_config(args.input)
        self._dataset = ProductsDatasets.Load_by_name(args.dataset)
        self.layer_types = ['pooling', 'dense', 'convolution', "dropout"]
        self.layer_type_probab = [
            self.hyper_parameters.pooling_probability,
            self.hyper_parameters.dense_probability,
            self.hyper_parameters.conv_probability,
            self.hyper_parameters.dropout_probability
            ]
        self.best_network = None
        self._scaler = None
        self._transformer = None
        if len(self.layer_type_probab) != len(self.layer_types):
            raise ValueError(f"{len(self.layer_type_probab)} != {len(self.layer_types)}")
        
        if np.sum(self.layer_type_probab) <= 0:
            raise ValueError(f"sum of selected pobabilites is less than or equal to zero")
        
        if np.sum(self.layer_type_probab) != 1:
            logger.warning(f"sum of probabilities: {self.layer_type_probab} isnt 1, normalizing it")
            self.layer_type_probab = self.layer_type_probab / np.sum(self.layer_type_probab)

        if args.scale:
            self._scaler = self._dataset.scale_features()

        if args.dimension_reduction:
            self._transformer = self._dataset.reduce_dimensions(args.dimension_reduction)

        self.training_parameters  = f"dense_prob-{self.hyper_parameters.dense_probability} conv_prob-{self.hyper_parameters.conv_probability} dropout_prob-{self.hyper_parameters.dropout_probability} pooling_prob-{self.hyper_parameters.pooling_probability}"

    def RunRandomSearch(self):
        n_targets = len(np.unique(self._dataset.train_targets))
        # targets = keras.utils.to_categorical(self._dataset.train_targets, num_classes=n_targets)
        targets = self._dataset.train_targets
        for generation in range(self.hyper_parameters.n_sampled_networks):

            logger.info(f"Creating {generation}. netowrk")
            data_shape = self._dataset.train_set.shape[1:]

            if generation != 0 and generation % CLEAR_MEMORY_INTERVAL == 0:    
                logger.debug(msg="Clearing keras memory and calling gc.collect")
                keras.backend.clear_session(free_memory=True)
                # tf.compat.v1.reset_default_graph()
                gc.collect()   
           
            random_model = self._sample_random_network(data_shape, n_targets, network_id=f"Random_model_{generation}")

            random_model.compile(
            loss='binary_crossentropy',
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            metrics=['accuracy', 'recall', 'precision']
            )

            logger.info(f"Fitting {generation}. netowrk")
            logger.info(f"Fitting {random_model.name}")

            random_model.fit(self._dataset.train_set, targets , batch_size=32, epochs=self.hyper_parameters.training_epochs)

            self._compare_models(random_model)
        
        self.best_network.summary()

    def Plot_best_model(self, name, file_type = '.svg'):
        if not self.best_network:
            logger.warning(f"No model to be plotted")
            return
        
        keras.utils.plot_model(self.best_network, to_file=name+file_type, show_layer_names=False, show_layer_activations=True, show_shapes=True)

    def Save_model(self, name):
        self.best_network.save(name)

    def _compare_models(self, sampled_model:keras.models.Sequential):
        """Compares best model with sampled model based on f1 score if sampled model has better f1 score replaces best model with sampled model.

        Args:
            sampled_model (keras.models.Sequential): new model to be compared with the best model
        """
        if not self.best_network: # No best network
            self.best_network = sampled_model
            return
        
        best_network_result = self.validate()
        sampled_network_result = self.validate(model=sampled_model)

        if sampled_network_result['f1_score'] > best_network_result['f1_score']:
            self.best_network = sampled_model
            logger.info(msg=f"New best model found with f1_score{sampled_network_result['f1_score'] }")

    def _sample_random_network(self, input_shape, n_classes, network_id = 'None'):

        n_hidden_layers = np.random.randint(low=self.hyper_parameters.minimul_hidden_layer_size, high=self.hyper_parameters.maximum_hidden_layer_size)
        model = keras.Sequential(name=f"{network_id}")
        model.add(keras.layers.Input(shape=input_shape, name='input'))
        
        previous_layer = None
       
        for i in range(n_hidden_layers):
            select_label =  np.random.choice(self.layer_types, p=self.layer_type_probab)

            logger.info(f"Generating layer {i}")
            if i == 0 and select_label == "dropout": # dont use dropout at start
                select_label = "dense"

           # input(select_label)
            match select_label:
                
                case 'pooling':
                    previous_layer = self._add_pooling_layer(model,i, previous_layer=previous_layer, input_shape=input_shape)

                case "convolution":
                    previous_layer = self._add_conv_layer(model, i,previous_layer=previous_layer, input_shape=input_shape)
                    # = "conv"

                case "dropout":
                    previous_layer = self._add_dropout_layer(model, i)

                case _: # defualt selected layer is dense
                    previous_layer = self._add_dense_layer(model,id=i)


        # create output layer with softmax 
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(1, activation='sigmoid', name='Output'))

        return model
    
    def _add_dense_layer(self,model: keras.models.Sequential, id, layer_size = None) -> str:
        model.add(keras.layers.Flatten())
        layer_size = np.random.choice(self.hyper_parameters.dense_layer_size) if layer_size is None else layer_size
        act_func = np.random.choice(self.hyper_parameters.possible_activations)
        model.add(keras.layers.Dense(layer_size,activation=act_func, name=f"dense_hidden_layer_{id}"))
        logger.debug(f"dense {layer_size=}")
        return "dense"

    def _add_pooling_layer(self, model, id, previous_layer, input_shape):

        pooling_type = np.random.choice(self.hyper_parameters.pool_types)
        strides = (np.random.choice(self.hyper_parameters.pool_strides_choice),)
        size = (np.random.choice(self.hyper_parameters.pool_size_choice),)
        
        if model.layers and model.output_shape[1] <= self.hyper_parameters.resize_small_layer:
            # model is too small expand it by adding dense layer
            logger.debug(msg=f"pooling_layer is too small, extending it to {self.hyper_parameters.resize_layer_value} ")
            previous_layer = "dense"
            self._add_dense_layer(model,f"extending_pooling_layer_with_dense{id}",self.hyper_parameters.resize_layer_value)


        if not model.layers :
            model.add(keras.layers.Reshape((input_shape[0], 1), input_shape=(input_shape,))) # Added input_shape to first layer
        else:
            if previous_layer == 'dense' or previous_layer == "dropout" or previous_layer == "flatten":
                model.add(keras.layers.Reshape((model.output_shape[1], 1), input_shape = (model.output_shape[1], ))) # Added input_shape to first layer
            else:
                logger.debug(f"{model.output_shape=} {previous_layer=}")


        if pooling_type == 'max':
            model.add(keras.layers.MaxPool1D(strides=strides, pool_size=size, name=f"MaxPool_{id}"))
        else: # else average pooling
            model.add(keras.layers.AveragePooling1D(strides=strides, pool_size=size,name=f"AveragePool_{id}"))

        return "pooling"

    def _add_conv_layer(self, model: keras.models.Sequential, id, previous_layer, input_shape):
        filter = np.random.choice(self.hyper_parameters.conv_filters)
        kernel = (np.random.choice(self.hyper_parameters.conv_kernel_sizes),)
        strides = (np.random.choice(self.hyper_parameters.conv_strides),)
        act_function = np.random.choice(self.hyper_parameters.conv_act_functions)
        convolution_type = np.random.choice(self.hyper_parameters.conv_types)

        if model.layers and model.output_shape[1] <= self.hyper_parameters.resize_small_layer:
            # model is too small expand it by adding dense layer
            logger.info(msg=f"pooling_layer is too small, extending it to {self.hyper_parameters.resize_layer_value} ")
            self._add_dense_layer(model,f"extending_conv_layer_with_dense{id}",self.hyper_parameters.resize_layer_value)
            
            previous_layer = "dense"

        if not model.layers :
            model.add(keras.layers.Reshape((input_shape[0], 1), input_shape=(input_shape,))) # Added input_shape to first layer
        else:
            if previous_layer == 'dense' or previous_layer == "dropout" or previous_layer == "flatten":
                model.add(keras.layers.Reshape((model.output_shape[1], 1), input_shape = (model.output_shape[1], ))) # Added input_shape to first layer
            else:
                logger.debug(f"{model.output_shape=} {previous_layer=} {convolution_type=}")


        conv_type = "conv"
        if convolution_type == "separable":
            model.add(keras.layers.SeparableConv1D(filters=filter, kernel_size=kernel, activation=act_function, strides=strides, name=f"SeparableConv1D_{id}"))
        elif convolution_type == "depth":
            model.add(keras.layers.DepthwiseConv1D(kernel_size=kernel, activation=act_function, strides=strides, name=f"DepthwiseConv1D_{id}"))
        elif convolution_type == "transpose":
            model.add(keras.layers.Conv1DTranspose(filters=filter, kernel_size=kernel, activation=act_function, strides=strides, name=f"Conv1DTranspose_{id}"))
        else:
            model.add(keras.layers.Conv1D(filters=filter, kernel_size=kernel, activation=act_function,strides=strides, name=f"Conv1D_{id}"))
        if model.output_shape[1] > 1:
            conv_type = "flatten"
            model.add(keras.layers.Flatten())
        return conv_type
        # print(f"{convolution_type=}, {strides=} {kernel=} {filter=}, {model.output_shape}")
    
    def _add_dropout_layer(self, model: keras.models.Sequential, id) -> str:
        rate = np.random.uniform(low=self.hyper_parameters.dropout_min_rate, high=self.hyper_parameters.dropout_max_rate)
        model.add(keras.layers.Dropout(rate=rate, name=f"Dropout_rate_{rate:.3f}_{id}"))
        logger.debug(f"dropout {model.output_shape=}")
        return "dropout" 
        
    def validate(self, test_set: Optional[Sequence] = None , target_set: Optional[Sequence] = None, model: Optional[keras.models.Sequential| None] = None) -> dict[str, float]:
            """Va;odates best network against unseen data.

            Args:
                test_set (Optional[Sequence], optional): testing set. Defaults to None.
                target_set (Optional[Sequence], optional): testing true outpiut. Defaults to None.

            Returns:
                dict[str, float]: dictionary of name of a metric and metric's value
            """

            if not model:
                model = self.best_network

            if test_set is None and target_set is None:
                predicted = model.predict(self._dataset.test_set)
                target_set = self._dataset.test_targets
                

            elif (test_set is None and target_set is not None) or (test_set is not None and target_set is None):
                assert ValueError("Invalid test set or target set")
            else:
                predicted = model.predict(test_set)
            predicted = np.round(predicted).astype(int)

           # predicted = np.argmax(predicted, axis=1)
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
        

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.WARNING,
        format='- %(name)s - %(levelname)s - %(message)s',
    )

    NEAT_METHOD = 'BasicNEAT'
    ALL = 'all'
    NORMAL_PROMAP = 'promap'
    HYPERNEAT_METHOD = 'HyperNEAT'
    CODEAPNEAT_METHOD = 'CoDeepNEAT'
    METHOD_CHOICES = [ALL,NEAT_METHOD, HYPERNEAT_METHOD, CODEAPNEAT_METHOD]

    # NAS methods directory names
    NEAT_DIRECTORY = 'NEAT'
    HYPERNEAT_DIRECTORY = 'HyperNEAT'
    CODEEPNEAT_DIRECTORY = "CoDeepNEAT"

    CSV_HEADER = ['TRAINING_DATASET', 'TESTING_DATASET','SCALED','DIMENSION_REDUCTION','METHOD','PARAMETERS', 'F1_SCORE', 'ACCURACY', 'PRECISION', 'RECALL', 'BALANCED_ACCURACY']
    GLOBAL_FILE = 'global_results.csv'

    parser = argparse.ArgumentParser()

    # dataset preprocessing arguments
    parser.add_argument('--dimension_reduction', '--dims',default='raw', choices=['raw', 'lda', 'pca'],type=str.lower, help="Specify the dimension reduction technique: 'raw', 'lda', or 'pca'")
    parser.add_argument('--scale', '--s',default=False, action='store_true', help="Standardize data")
    
    # dataset arguments
    available_datasets = list(ProductsDatasets.NAME_MAP.keys()) + [ALL, NORMAL_PROMAP]
    parser.add_argument('--dataset', '--d',default='promapcz', type=str.lower, choices=available_datasets, help='name of promap dataset or path')
    # output arguments
    logger.debug(f"{available_datasets}")
    parser.add_argument('--output', '--o', type=str.lower, default='output', help='Output directory name.')
    parser.add_argument('--validate_all', '--v', action='store_false', default=True, help="Validates input against all possible datasets. If feature count is not same, the testing dataset is extened or reduces to match the training dataset's features")
    parser.add_argument('--kbest', '--k', default=10,type=int, help='prints k best networks')
    parser.add_argument('--remove_global_results','--rg',action='store_true', default=False, help='Removes global results from global_results.csv. If not set, appends to the file.')
    # Config generation
    parser.add_argument('--config_directory', '--dir', default='ConfigGeneration', type=str, help='Directory name in which all generated configs are saved')
    parser.add_argument('--config_generation', '--g', default=True, action='store_false',help='Disables config generation')
    parser.add_argument('--input', '--i', type=str, default='input/input.json', help='Path to config generation input.')
    parser.add_argument('--default','--def', action='store_false', default=True, help='Disables default value generations in config.' )
    parser.add_argument('--all_files','--all', action='store_true', default=False, help='Generates configs from all .neat (.ini) files in config directory set by config_directory argument. ')
    
    parser.add_argument('--NAS_method','--nas', type=str, default=ALL, choices=METHOD_CHOICES, help='Selects the method of NAS.')
    rs = RandomSearch(parser.parse_args())
    rs.RunRandomSearch()
