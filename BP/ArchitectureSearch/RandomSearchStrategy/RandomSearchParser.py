import json
import logging
import enum
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)




class RandomSearchParser:
    def __init__(self):
        self.training_epochs = 10
        self.final_training_epochs = 20
        self.minimul_hidden_layer_size = 1
        self.maximum_hidden_layer_size = 10
        self.n_sampled_networks = 1

        self.RANDOM_SEARCH_KEY = 'RandomSearch'
        self.possible_activations = ['relu']

        self.dense_probability = .8
        self.dense_layer_size = [32,64]

        self.pooling_probability = .05
        self.pool_size_choice = [2]
        self.pool_strides_choice = [1]
        self.pool_types = ['max', 'average']

        self.conv_filters = [1]
        self.conv_type = ['conv','separable', 'depthwise', 'transpose']


        self.dropout_probability = .05
        self.convolution_probability = 0.10


    def _load_args(self, file_content):
        if file_content is None:
            raise ValueError('Config is not loaded')
        # dense layers hyper params
        self.dense_probability = float(file_content["dense_probability"])
        self.dense_layer_size = [int(n) for n in file_content['layer_size']]
        
        # pooling layers hyper params
        self.pooling_probability = float(file_content["pooling_probability"])
        self.pool_size_choice = [int(x) for x in file_content["pool_size_choice"]]
        self.pool_strides_choice = [int(x) for x in file_content["pool_strides_choice"]]
        self.pool_types = file_content["pool_types"]




        # other hyperparams

        self.n_sampled_networks = int(file_content['sampled_networks'])
        print(self.n_sampled_networks)
        self.training_epochs = int(file_content['training_epochs'])
        self.final_training_epochs = int(file_content['final_training_epochs'])
        
        self.minimul_hidden_layer_size = int(file_content['minimum_hidden_layer_size'])
        self.maximum_hidden_layer_size = int(file_content['maximum_hidden_layer_size'])
        if self.minimul_hidden_layer_size > self.maximum_hidden_layer_size:
            self.minimul_hidden_layer_size, self.maximum_hidden_layer_size = self.maximum_hidden_layer_size, self.minimul_hidden_layer_size
            logger.warning(msg=f" min layer size: {self.minimul_hidden_layer_size} is larger than max layer size {self.maximum_hidden_layer_size}, swapping values ")
        self.possible_activations =  file_content['possible_activations']
        
    def parse_config(self, config_path: str) -> None:
        with open(config_path, 'r', encoding='utf-8') as f:
            file_content = json.load(f)[self.RANDOM_SEARCH_KEY]
            self._load_args(file_content=file_content)