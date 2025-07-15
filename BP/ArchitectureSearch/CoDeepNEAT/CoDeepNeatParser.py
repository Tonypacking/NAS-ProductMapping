import json
import keras
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class CoDeepNeatParser:
    """Parser for CoDeepNEAT NAS strategy
    """
    def __init__(self):
        self.generations = None
        self.training_epochs = None
        self.population_size = None
        self.blueprint_population_size = None
        self.module_population_size = None
        self.n_blueprint_species = None
        self.n_module_species = None
        self.final_model_training_epoch = None

        self.crossover_rate = None
        self.mutation_rate = None
        self.elitism_rate = None

        self.parameters = None

        self._file_content = None

    def _load_args(self) :
        """Loads args from JSON file

        Raises:
            ValueError: Config wasnt loaded
        """
        if self._file_content is None:
            raise ValueError('Config is not loaded')
        
        args = self._file_content['args']
        
        self.generations = int(args['generations'])
        self.training_epochs = int(args['training_epochs'])
        self.population_size = int(args['population_size'])
        self.final_model_training_epoch = int(args['final_method_training_epochs'])
        
        self.blueprint_population_size = int(args['blueprint_population_size'])
        self.module_population_size = int(args['module_population_size'])
        self.n_blueprint_species = int(args['n_blueprint_species'])
        self.n_module_species = int(args['n_module_species'])

        self.crossover_rate = float(args['crossover_rate'])
        self.mutation_rate = float(args['mutation_rate'])
        self.elitism_rate = float(args['elitism_rate'])

        self.parameters = f"gen-{self.generations};epochs-{self.training_epochs};pop_size-{self.population_size};mutate-{self.mutation_rate};crossover-{self.crossover_rate};module_size-{self.module_population_size};module_pop-{self.module_population_size}; finalmethodtraining-{self.final_model_training_epoch}"
        

    def load_config(self,config_path: str) -> None:
        """
        Loads the CoDeepNEAT configuration from a specified JSON file.

        Args:
            config_path (str): The file path to the JSON configuration file.
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            self._file_content = json.load(f)['CoDeepNEAT']
        self._load_args()
   
    def _convert_list(self, list: list):
        """Helper function to convert json string list to python list int list

        Args:
            list (list): Json string list

        Returns:
            _type_: list[int]
        """
        # dont convert the input if it is a string for example in padding or data formats channels last
        if any(isinstance(x, str) for x in list):
            return list
        return [int(x) for x in list]
    
    def _parse_json_to_CoDeepNeatConfig(self, input_config: dict|None) -> dict:
        """ Parses JSON input config object

        Args:
            input_config (dict | None): Input json config

        Returns:
            dict: None if json input is null otherwise converts the input config to python dictionary
        """
        output = {}
        if input_config is None:
            return None
        
        for range_str in input_config.keys():
            output[range_str] = (self._convert_list(input_config[range_str][0]), input_config[range_str][-1])
        return output

    def _parse_keras_operation(self, key:str, values):
        """
        Parses a list of values, converting string representations of Keras layers
        into their actual Keras class objects or parsing nested configurations.
        """
        list = []
        for value in values:
            if isinstance(value, str):

                try:
                    list.append(getattr(keras.layers,value))
                except Exception as exception:
                    # print(f"Keras.Layer wasn't recognized {exception=}. Setting {key} to None. ")
                    logger.warning(f"Keras.Layer wasn't recognized {exception=}. Setting {key} to None. ")
                    return None
            elif isinstance(value,dict):
                list.append(self._parse_json_to_CoDeepNeatConfig(value))
                
        return tuple(list)
            #print(f"val {value}")
    
    def Get_global_config(self):
        """Gets global config

        Returns:
            _type_: _description_
        """
        input_confg : dict= self._file_content['global_configs']

        return self._parse_json_to_CoDeepNeatConfig(input_confg)
    
    def Get_input_configs(self):
        """Parses input config and returns it as a python object.

        Returns:
            dict: Returns input config representation
        """
        input_confg : dict= self._file_content['input_configs']        
        if input_confg is None:
            return None
        return self._parse_json_to_CoDeepNeatConfig(input_confg)
    
    def Get_output_configs(self):
        """Parses output config and returns it as a python object.

        Returns:
            dict: Returns output config representation
        """
        input_confg : dict= self._file_content['output_configs']
        return self._parse_json_to_CoDeepNeatConfig(input_confg)

    def Get_possible_components(self):
        """Parses and returns all possible components
        Returns:
            dict: Component dictionary 
        """
        config : dict = self._file_content['possible_components']
        output = {}
        for x, y  in config.items():
            output[x] = self._parse_keras_operation(x, y)
        return output

    def Get_Possible_inputs(self):
        """
        Parses possible inputs and returns possible inputs.

        Returns:
            dict: Returns parsed possible inputs
        """
        config : dict = self._file_content['possible_inputs']
        output = {}
        for x, y  in config.items():
            output[x] = self._parse_keras_operation(x, y)
        return output

    def Get_possible_outputs(self):
        """
        Parses possible outputs and returns possible inputs.

        Returns:
            dict: Returns parsed possible outputs
        """
        config : dict = self._file_content['possible_outputs']
        output = {}

        if config is None:
            return None
        for x, y  in config.items():
            output[x] = self._parse_keras_operation(x, y)
        return output

    def Get_possible_complementaty_components(self):
        """
        Parses possible complementary components and returns possible inputs.

        Returns:
            dict: Returns parsed possible complementary components
        """
        config : dict = self._file_content['possible_complementary_components']
        output = {}

        if config is None:
            return None
        for x, y  in config.items():
            output[x] = self._parse_keras_operation(x, y)
        return output

    def Get_possible_complementary_inputs(self):
        """
        Parses possible complementary inputs and returns possible inputs.

        Returns:
            dict: Returns parsed possible complementary inputs
        """
        config : dict = self._file_content['possible_complementary_inputs']
        output = {}
        if config is None:
            return None
        for x, y  in config.items():
            output[x] = self._parse_keras_operation(x, y)
        return output

    def Get_possible_complementary_outputs(self):
        """
        Parses possible complementary complementary outputs and returns possible inputs.

        Returns:
            dict: Returns parsed possible complementary outputs
        """
        config : dict = self._file_content['possible_complementary_outputs']
        output = {}
        if config is None:
            return None
        for x, y  in config.items():
            output[x] = self._parse_keras_operation(x, y)
        return output