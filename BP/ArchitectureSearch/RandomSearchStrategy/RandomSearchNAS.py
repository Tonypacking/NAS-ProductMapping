import keras
import numpy as np
import random # For generating random numbers
import enum
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..','..'))) # To load Utils module
from Utils.ProMap import ProductsDatasets, Dataset
import argparse
from RandomSearchParser import RandomSearchParser
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)



class RandomSearch:
    def __init__(self, args:argparse):
        self.hyper_parameters = RandomSearchParser()
        self.hyper_parameters.parse_config(args.input)
        self.dataset = ProductsDatasets.Load_by_name(args.dataset)
        self.layer_types = ['pooling', 'dense']
        self.layer_type_probab = [
            self.hyper_parameters.pooling_probability,
            self.hyper_parameters.dense_probability
            ]
        


        if np.sum(self.layer_type_probab) == 0:
            raise ValueError(f"sum of selected pobabilites is 0")
        
        if np.sum(self.layer_type_probab) != 1:
            logger.warning(f"sum of probabilities: {self.layer_type_probab} isnt 1, normalizing it")
            self.layer_type_probab = self.layer_type_probab / np.sum(self.layer_type_probab)

    def RunRandomSearch(self):
        
        for _ in range(self.hyper_parameters.n_sampled_networks):

            data_shape = self.dataset.train_set.shape[1:]
            random_model = self._sample_random_network(data_shape,np.unique(self.dataset.train_targets.shape).size)

            random_model.fit(self.dataset.train_set, self.dataset.train_targets)
            random_model.summary()


    def _sample_random_network(self, input_shape, n_classes, network_id = 'None'):
        n_hidden_layers = np.random.randint(low=self.hyper_parameters.minimul_hidden_layer_size, high=self.hyper_parameters.maximum_hidden_layer_size)
        model = keras.Sequential(name=network_id)
        model.add(keras.layers.Input(shape=input_shape, name='input'))
        previous_layer = None
        for i in range(3):
            #select_label = np.random.choice(self.layer_types, p=self.layer_type_probab)
            a = ['dense', 'pooling', 'pooling'] # mock list remove and use random choice
            select_label = a[i]
            input(select_label)
            match select_label:
                case 'pooling':

                    pooling_type = np.random.choice(self.hyper_parameters.pool_types)
                    strides = (np.random.choice(self.hyper_parameters.pool_strides_choice),)
                    size = (np.random.choice(self.hyper_parameters.pool_size_choice),)
                    if not model.layers :
                        model.add(keras.layers.Reshape((input_shape[0], 1), input_shape=(input_shape,))) # Added input_shape to first layer
                    else:
                        if previous_layer == 'dense':
                            model.add(keras.layers.Reshape((model.output_shape[1], 1), input_shape = (model.output_shape[1], ))) # Added input_shape to first layer

                    if pooling_type == 'max':
                        model.add(keras.layers.MaxPool1D(strides=strides, pool_size=size))
                    else: # else average pooling
                        model.add(keras.layers.AveragePooling1D(strides=strides, pool_size=size))

                    previous_layer = 'pooling'
                case _: # defualt selected layer is dense
                    model.add(keras.layers.Flatten())
                    layer_size = np.random.choice(self.hyper_parameters.dense_layer_size)
                    act_func = np.random.choice(self.hyper_parameters.possible_activations)
                    model.add(keras.layers.Dense(layer_size,activation=act_func, name=f"hidden_{i}"))
                    print(f"{layer_size=}")
                    previous_layer = 'dense'


        # create output layer with softmax 
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(n_classes, activation='sigmoid', name='Output'))
        model.compile(
            loss='binary_crossentropy',
            optimizer=keras.optimizers.Adam(learning_rate=0.05),
            metrics=['accuracy', 'recall', 'precision']
            )
        return model
    def validate(self):
        pass
    
    def validate_all(self):
        pass

    


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